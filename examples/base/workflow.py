"""Abstract workflow runner for AtomicGuard examples."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from atomicguard import (
    ActionPair,
    FilesystemArtifactDAG,
    OllamaGenerator,
    PromptTemplate,
    Workflow,
    WorkflowResult,
    WorkflowStatus,
)

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


class BaseWorkflowRunner(ABC):
    """
    Abstract base class for workflow runners.

    Provides common workflow execution infrastructure while allowing
    subclasses to customize workflow building and result handling.
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
        """Get workflow name."""
        return str(self.workflow_config.get("name", "Unknown Workflow"))

    @abstractmethod
    def build_workflow(self) -> Workflow:
        """
        Build the workflow instance.

        Subclasses implement this to customize workflow construction.

        Returns:
            Configured Workflow instance
        """
        pass

    @abstractmethod
    def get_specification(self) -> str:
        """
        Get the specification/prompt for workflow execution.

        Returns:
            Specification string
        """
        pass

    def create_generator(self) -> OllamaGenerator:
        """Create the LLM generator."""
        generator_kwargs: dict[str, Any] = {"model": self.model}
        if self.host:
            generator_kwargs["base_url"] = normalize_base_url(self.host)
        return OllamaGenerator(**generator_kwargs)

    def execute(self) -> tuple[WorkflowResult, float]:
        """
        Execute the workflow.

        Returns:
            Tuple of (WorkflowResult, duration_seconds)
        """
        workflow = self.build_workflow()
        specification = self.get_specification()

        start_time = datetime.now()
        self.logger.info(f"Executing workflow: {self.workflow_name}")

        result = workflow.execute(specification)
        duration = (datetime.now() - start_time).total_seconds()

        if result.status == WorkflowStatus.SUCCESS:
            self.logger.info(f"Workflow completed successfully in {duration:.2f}s")
        else:
            self.logger.warning(
                f"Workflow {result.status.value} at step '{result.failed_step}' "
                f"after {duration:.2f}s"
            )

        return result, duration


class StandardWorkflowRunner(BaseWorkflowRunner):
    """
    Standard workflow runner for TDD-style workflows.

    Handles workflows defined with action_pairs configuration
    (tdd_human_review, tdd_import_guard).
    """

    def build_workflow(self) -> Workflow:
        """Build workflow from action_pairs configuration."""
        generator = self.create_generator()
        workflow = Workflow(artifact_dag=self.artifact_dag, rmax=self.rmax)

        action_pairs_config = self.workflow_config["action_pairs"]

        for step_id, ap_config in action_pairs_config.items():
            guard = build_guard(ap_config)
            prompt_template = self.prompts.get(step_id)

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


def save_workflow_results(
    output_path: str,
    workflow_config: dict[str, Any],
    result: WorkflowResult,
    model: str,
    duration: float,
) -> None:
    """
    Save workflow results to JSON file.

    Args:
        output_path: Path to save results
        workflow_config: Original workflow configuration
        result: WorkflowResult from execution
        model: Model name used
        duration: Execution duration in seconds
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def display_workflow_result(
    result: WorkflowResult,
    output_path: str,
    artifact_dir: str,
    log_file: str,
    artifact_keys: tuple[str, ...] = ("g_test", "g_impl"),
) -> int:
    """
    Display workflow result and return exit code.

    Args:
        result: WorkflowResult from execution
        output_path: Path where results were saved
        artifact_dir: Directory where artifacts were saved
        log_file: Path to log file
        artifact_keys: Keys of artifacts to display on success

    Returns:
        Exit code (0 for success, 1 for failure)
    """
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
