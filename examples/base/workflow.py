"""Abstract runners for AtomicGuard examples."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from atomicguard import (
    ActionPair,
    Artifact,
    DualStateAgent,
    FilesystemArtifactDAG,
    OllamaGenerator,
    PromptTemplate,
    Workflow,
    WorkflowResult,
    WorkflowStatus,
)
from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.infrastructure import GeneratorRegistry

from .config import normalize_base_url
from .console import (
    console,
    print_failure,
    print_provenance,
    print_success,
)
from .guards import build_guard

if TYPE_CHECKING:
    pass


# Type alias for execution results
ExecutionResult = WorkflowResult | Artifact


class BaseRunner(ABC):
    """
    Abstract base class for example runners.

    Supports both multi-step workflows (via Workflow class) and
    single-step execution (via DualStateAgent directly).

    Subclasses implement:
    - build(): Returns either a Workflow or DualStateAgent
    - get_specification(): Returns the input specification/prompt
    """

    def __init__(
        self,
        workflow_config: dict[str, Any],
        prompts: dict[str, PromptTemplate],
        artifact_dag: FilesystemArtifactDAG,
        host: str | None = None,
        model_override: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.workflow_config = workflow_config
        self.prompts = prompts
        self.artifact_dag = artifact_dag
        self.host = host
        self.model_override = model_override
        self.logger = logger or logging.getLogger(__name__)

    @property
    def model(self) -> str:
        """Get effective model name."""
        return self.model_override or self.workflow_config.get(
            "model", "qwen2.5-coder:14b"
        )

    @property
    def rmax(self) -> int:
        """Get max retry attempts."""
        return int(self.workflow_config.get("rmax", 3))

    @property
    def workflow_name(self) -> str:
        """Get workflow/agent name."""
        return str(self.workflow_config.get("name", "Unknown"))

    @property
    def constraints(self) -> str:
        """Get global constraints (Ω)."""
        return str(self.workflow_config.get("constraints", ""))

    @abstractmethod
    def build(self) -> Workflow | DualStateAgent:
        """
        Build the executor instance.

        Returns:
            Configured Workflow (multi-step) or DualStateAgent (single-step)
        """
        pass

    @abstractmethod
    def get_specification(self) -> str:
        """
        Get the specification/prompt for execution.

        Returns:
            Specification string
        """
        pass

    def create_generator(
        self,
        generator_name: str | None = None,
        generator_config: dict[str, Any] | None = None,
    ) -> GeneratorInterface:
        """
        Create a generator instance.

        If generator_name is provided (from schema), uses the registry with
        the schema-provided config as-is (typed config dataclass will validate).

        Otherwise falls back to OllamaGenerator with workflow-level defaults.

        Args:
            generator_name: Generator type from schema (e.g., "OllamaGenerator")
            generator_config: Generator-specific config from schema

        Returns:
            GeneratorInterface instance
        """
        config = generator_config.copy() if generator_config else {}

        if generator_name:
            # Use registry for schema-specified generators
            # Pass config as-is - let the generator's config_class validate
            # Only inject model/base_url for OllamaGenerator (maintains backward compat)
            if generator_name == "OllamaGenerator":
                if "model" not in config:
                    config["model"] = self.model
                if self.host and "base_url" not in config:
                    config["base_url"] = normalize_base_url(self.host)
            return GeneratorRegistry.create(generator_name, **config)
        else:
            # Default: OllamaGenerator with standard config
            config["model"] = self.model
            if self.host:
                config["base_url"] = normalize_base_url(self.host)
            return OllamaGenerator(**config)

    def execute(self) -> tuple[ExecutionResult, float]:
        """
        Execute the workflow or agent.

        Returns:
            Tuple of (WorkflowResult or Artifact, duration_seconds)
        """
        executor = self.build()
        specification = self.get_specification()

        start_time = datetime.now()
        self.logger.info(f"Executing: {self.workflow_name}")

        result = executor.execute(specification)
        duration = (datetime.now() - start_time).total_seconds()

        self._log_result(result, duration)
        return result, duration

    def _log_result(self, result: ExecutionResult, duration: float) -> None:
        """Log execution result."""
        if isinstance(result, WorkflowResult):
            if result.status == WorkflowStatus.SUCCESS:
                self.logger.info(f"Completed successfully in {duration:.2f}s")
            else:
                self.logger.warning(
                    f"Failed at step '{result.failed_step}' after {duration:.2f}s"
                )
        else:
            # Artifact from DualStateAgent
            self.logger.info(
                f"Completed in {duration:.2f}s (attempt {result.attempt_number})"
            )


class WorkflowRunner(BaseRunner):
    """
    Multi-step workflow runner using the Workflow class.

    Use this for workflows with multiple dependent steps
    (e.g., TDD: generate tests → generate implementation).
    """

    def build(self) -> Workflow:
        """Build workflow from action_pairs configuration."""
        # Create a default generator for steps that don't specify one
        default_generator = self.create_generator()
        workflow = Workflow(artifact_dag=self.artifact_dag, rmax=self.rmax)

        action_pairs_config = self.workflow_config["action_pairs"]

        for step_id, ap_config in action_pairs_config.items():
            guard = build_guard(ap_config)
            prompt_template = self.prompts.get(step_id)

            # Use schema-specified generator if present, otherwise default
            generator_name = ap_config.get("generator")
            generator_config = ap_config.get("generator_config")

            if generator_name:
                generator = self.create_generator(generator_name, generator_config)
            else:
                generator = default_generator

            action_pair = ActionPair(
                generator=generator,
                guard=guard,
                prompt_template=prompt_template,
            )

            requires = tuple(ap_config.get("requires", []))

            workflow.add_step(
                guard_id=step_id,
                action_pair=action_pair,
                requires=requires,
                deps=requires,
            )

        return workflow

    def get_specification(self) -> str:
        """Get specification from workflow config."""
        return str(self.workflow_config["specification"])


class AgentRunner(BaseRunner):
    """
    Single-step runner using DualStateAgent directly.

    Use this for single-step execution where you have one complex
    ActionPair (e.g., ADD workflow with custom generator).
    """

    def __init__(
        self,
        workflow_config: dict[str, Any],
        prompts: dict[str, PromptTemplate],
        artifact_dag: FilesystemArtifactDAG,
        action_pair: ActionPair,
        host: str | None = None,
        model_override: str | None = None,
        logger: logging.Logger | None = None,
    ):
        super().__init__(
            workflow_config=workflow_config,
            prompts=prompts,
            artifact_dag=artifact_dag,
            host=host,
            model_override=model_override,
            logger=logger,
        )
        self.action_pair = action_pair

    def build(self) -> DualStateAgent:
        """Build DualStateAgent with the configured ActionPair."""
        return DualStateAgent(
            action_pair=self.action_pair,
            artifact_dag=self.artifact_dag,
            rmax=self.rmax,
            constraints=self.constraints,
        )

    def get_specification(self) -> str:
        """Get specification from workflow config or override."""
        return str(self.workflow_config.get("specification", ""))


# Keep old names as aliases for backwards compatibility
BaseWorkflowRunner = BaseRunner
StandardWorkflowRunner = WorkflowRunner


def save_workflow_results(
    output_path: str,
    workflow_config: dict[str, Any],
    result: ExecutionResult,
    model: str,
    duration: float,
) -> None:
    """
    Save execution results to JSON file.

    Works with both WorkflowResult (multi-step) and Artifact (single-step).

    Args:
        output_path: Path to save results
        workflow_config: Original workflow configuration
        result: WorkflowResult or Artifact from execution
        model: Model name used
        duration: Execution duration in seconds
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if isinstance(result, WorkflowResult):
        data = _build_workflow_result_data(workflow_config, result, model, duration)
    else:
        data = _build_artifact_result_data(workflow_config, result, model, duration)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _build_workflow_result_data(
    workflow_config: dict[str, Any],
    result: WorkflowResult,
    model: str,
    duration: float,
) -> dict[str, Any]:
    """Build result data dict for WorkflowResult."""
    total_attempts = len(result.artifacts) + len(result.provenance)

    data: dict[str, Any] = {
        "workflow_name": workflow_config.get("name", "Unknown"),
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration, 2),
        "success": result.status == WorkflowStatus.SUCCESS,
        "status": result.status.value,
        "failed_step": result.failed_step,
        "total_attempts": total_attempts,
        "artifacts": {},
    }

    for step_id, artifact in result.artifacts.items():
        data["artifacts"][step_id] = {
            "artifact_id": artifact.artifact_id,
            "content": artifact.content,
            "attempt_number": artifact.attempt_number,
            "status": artifact.status.value if artifact.status else None,
        }

    if result.provenance:
        data["provenance"] = [
            {"attempt": i + 1, "content": artifact.content, "feedback": feedback}
            for i, (artifact, feedback) in enumerate(result.provenance)
        ]

    return data


def _build_artifact_result_data(
    workflow_config: dict[str, Any],
    artifact: Artifact,
    model: str,
    duration: float,
) -> dict[str, Any]:
    """Build result data dict for single Artifact."""
    data: dict[str, Any] = {
        "workflow_name": workflow_config.get("name", "Unknown"),
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration, 2),
        "success": True,
        "artifact_id": artifact.artifact_id,
        "attempt_number": artifact.attempt_number,
    }

    # Try to parse content as JSON manifest
    try:
        data["manifest"] = json.loads(artifact.content)
    except json.JSONDecodeError:
        data["content"] = artifact.content

    return data


def display_workflow_result(
    result: ExecutionResult,
    output_path: str,
    artifact_dir: str,
    log_file: str,
    artifact_keys: tuple[str, ...] = ("g_test", "g_impl"),
) -> int:
    """
    Display execution result and return exit code.

    Works with both WorkflowResult (multi-step) and Artifact (single-step).

    Args:
        result: WorkflowResult or Artifact from execution
        output_path: Path where results were saved
        artifact_dir: Directory where artifacts were saved
        log_file: Path to log file
        artifact_keys: Keys of artifacts to display (for WorkflowResult)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if isinstance(result, WorkflowResult):
        return _display_workflow_result(
            result, output_path, artifact_dir, log_file, artifact_keys
        )
    else:
        return _display_artifact_result(result, output_path, artifact_dir, log_file)


def _display_workflow_result(
    result: WorkflowResult,
    output_path: str,
    artifact_dir: str,
    log_file: str,
    artifact_keys: tuple[str, ...],
) -> int:
    """Display WorkflowResult."""
    if result.status == WorkflowStatus.SUCCESS:
        print_success("Workflow Complete")

        for key in artifact_keys:
            if key in result.artifacts:
                console.print(f"\n[bold]--- Generated {key} ---[/bold]")
                console.print(result.artifacts[key].content)

        console.print(f"\nResults saved to: {output_path}")
        console.print(f"Artifacts saved to: {artifact_dir}")
        console.print(f"Log file: {log_file}")
        return 0

    else:
        print_failure(f"Failed at step: {result.failed_step}")

        if result.provenance:
            print_provenance(result.provenance)

        console.print(f"\nResults saved to: {output_path}")
        console.print(f"Log file: {log_file}")
        return 1


def _display_artifact_result(
    artifact: Artifact,
    output_path: str,
    artifact_dir: str,
    log_file: str,
) -> int:
    """Display single Artifact result."""
    print_success("Execution Complete")

    console.print(f"\n[bold]Artifact ID:[/bold] {artifact.artifact_id}")
    console.print(f"[bold]Attempt:[/bold] {artifact.attempt_number}")

    # Try to parse and display as manifest
    try:
        manifest = json.loads(artifact.content)
        if "files" in manifest:
            console.print(
                f"\n[bold]Files created:[/bold] {manifest.get('file_count', len(manifest['files']))}"
            )
            for filepath in manifest.get("files", {}):
                console.print(f"  - {filepath}")
    except json.JSONDecodeError:
        # Show raw content preview
        content = artifact.content
        if len(content) > 500:
            content = content[:500] + "\n... (truncated)"
        console.print(f"\n[bold]Content:[/bold]\n{content}")

    console.print(f"\nResults saved to: {output_path}")
    console.print(f"Artifacts saved to: {artifact_dir}")
    console.print(f"Log file: {log_file}")
    return 0
