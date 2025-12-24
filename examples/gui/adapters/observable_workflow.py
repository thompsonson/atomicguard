"""Observable workflow wrapper for event emission during execution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
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
from atomicguard.domain.exceptions import EscalationRequired, RmaxExhausted
from examples.base import build_guard, normalize_base_url

from ..state.events import (
    AnyEvent,
    StepCompletedEvent,
    StepStartedEvent,
    WorkflowCompletedEvent,
    WorkflowEvent,
    WorkflowStartedEvent,
)

if TYPE_CHECKING:
    pass


@dataclass
class StepConfig:
    """Configuration for a workflow step."""

    step_id: str
    guard_config: dict[str, Any]
    requires: tuple[str, ...]


class ObservableWorkflow:
    """
    Workflow wrapper that emits events during execution.

    This class wraps the standard Workflow execution and emits
    events at key points for live monitoring in the GUI.
    """

    def __init__(
        self,
        workflow_config: dict[str, Any],
        prompts: dict[str, PromptTemplate],
        artifact_dag: FilesystemArtifactDAG,
        emit_callback: Callable[[AnyEvent], None],
        host: str | None = None,
        model_override: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the observable workflow.

        Args:
            workflow_config: Loaded workflow.json configuration
            prompts: Loaded prompts keyed by step ID
            artifact_dag: Artifact storage
            emit_callback: Function to call with events
            host: Optional Ollama host URL override
            model_override: Optional model name override
            logger: Optional logger instance
        """
        self._config = workflow_config
        self._prompts = prompts
        self._dag = artifact_dag
        self._emit = emit_callback
        self._host = host
        self._model_override = model_override
        self._logger = logger or logging.getLogger(__name__)
        self._stop_requested = False

    @property
    def model(self) -> str:
        """Get effective model name."""
        return self._model_override or self._config.get("model", "qwen2.5-coder:14b")

    @property
    def rmax(self) -> int:
        """Get max retry attempts."""
        return int(self._config.get("rmax", 3))

    @property
    def workflow_name(self) -> str:
        """Get workflow name."""
        return str(self._config.get("name", "Unknown"))

    def request_stop(self) -> None:
        """Request graceful stop of execution."""
        self._stop_requested = True
        self._logger.info("Stop requested - will halt after current step")

    def execute(self) -> tuple[WorkflowResult, float]:
        """
        Execute the workflow with event emission.

        Returns:
            Tuple of (WorkflowResult, duration_seconds)
        """
        self._stop_requested = False
        specification = str(self._config.get("specification", ""))
        action_pairs = self._config.get("action_pairs", {})

        # Log and emit workflow started
        self._logger.info(f"Starting workflow: {self.workflow_name}")
        self._logger.info(f"Model: {self.model}, Max retries: {self.rmax}")
        self._logger.info(f"Steps: {list(action_pairs.keys())}")

        self._emit(
            WorkflowStartedEvent(
                timestamp=WorkflowEvent.now(),
                event_type="workflow_started",
                workflow_name=self.workflow_name,
                step_count=len(action_pairs),
                model=self.model,
                specification_preview=specification[:200],
            )
        )

        start_time = datetime.now()

        try:
            # Build the workflow with wrapped execution
            result = self._execute_with_events(specification, action_pairs)
        except Exception as e:
            # Handle unexpected errors
            self._logger.error(f"Workflow execution failed: {e}")
            duration = (datetime.now() - start_time).total_seconds()
            self._emit(
                WorkflowCompletedEvent(
                    timestamp=WorkflowEvent.now(),
                    event_type="workflow_completed",
                    status="FAILED",
                    failed_step=None,
                    total_duration=duration,
                    total_artifacts=0,
                )
            )
            raise

        duration = (datetime.now() - start_time).total_seconds()

        # Log and emit workflow completed
        self._logger.info(f"Workflow completed: {result.status.value}")
        self._logger.info(
            f"Duration: {duration:.2f}s, Artifacts: {len(result.artifacts)}"
        )
        if result.failed_step:
            self._logger.warning(f"Failed step: {result.failed_step}")

        self._emit(
            WorkflowCompletedEvent(
                timestamp=WorkflowEvent.now(),
                event_type="workflow_completed",
                status=result.status.value,
                failed_step=result.failed_step,
                total_duration=duration,
                total_artifacts=len(result.artifacts),
            )
        )

        return result, duration

    def _execute_with_events(
        self, specification: str, action_pairs: dict[str, dict]
    ) -> WorkflowResult:
        """Execute workflow with event emission at each step."""
        # Create generator
        self._logger.debug(f"Creating OllamaGenerator with model: {self.model}")
        generator_kwargs: dict[str, Any] = {"model": self.model}
        if self._host:
            generator_kwargs["base_url"] = normalize_base_url(self._host)
            self._logger.debug(f"Using host: {self._host}")
        generator = OllamaGenerator(**generator_kwargs)

        # Build workflow
        constraints = str(self._config.get("constraints", ""))
        workflow = Workflow(
            artifact_dag=self._dag,
            rmax=self.rmax,
            constraints=constraints,
        )

        # Track step order for events
        step_configs: list[StepConfig] = []

        for step_id, ap_config in action_pairs.items():
            guard = build_guard(ap_config)
            prompt_template = self._prompts.get(step_id)
            guard_type_str = self._get_guard_type_str(ap_config)

            self._logger.info(f"Adding step '{step_id}' with guard: {guard_type_str}")

            action_pair = ActionPair(
                generator=generator,
                guard=guard,
                prompt_template=prompt_template,
            )

            requires = tuple(ap_config.get("requires", []))
            if requires:
                self._logger.debug(f"  Step '{step_id}' requires: {requires}")

            workflow.add_step(
                guard_id=step_id,
                action_pair=action_pair,
                requires=requires,
                deps=requires,
            )

            step_configs.append(
                StepConfig(
                    step_id=step_id,
                    guard_config=ap_config,
                    requires=requires,
                )
            )

        # Execute with monitoring
        self._logger.info("Beginning workflow execution...")

        result = self._execute_monitored(workflow, specification, step_configs)
        return result

    def _execute_monitored(
        self,
        workflow: Workflow,
        specification: str,
        step_configs: list[StepConfig],
    ) -> WorkflowResult:
        """
        Execute workflow with step-level monitoring.

        This implementation executes steps individually to emit
        granular events. It mirrors the Workflow.execute() logic
        but adds event emission.
        """
        # We need to access workflow internals for step-by-step monitoring
        # Use the standard execute but emit events based on state changes

        # For each step, emit started event before and completed after
        completed_steps: set[str] = set()
        step_start_times: dict[str, datetime] = {}

        def emit_step_events() -> None:
            """Check workflow state and emit events for state changes."""
            for config in step_configs:
                step_id = config.step_id
                is_satisfied = workflow._workflow_state.is_satisfied(step_id)

                if step_id not in completed_steps:
                    if step_id not in step_start_times:
                        # Check if preconditions are met (step could start)
                        preconditions_met = all(
                            workflow._workflow_state.is_satisfied(req)
                            for req in config.requires
                        )
                        if preconditions_met or not config.requires:
                            # Step is starting or about to start
                            step_start_times[step_id] = datetime.now()

                    if is_satisfied:
                        # Step just completed
                        completed_steps.add(step_id)
                        duration = 0.0
                        if step_id in step_start_times:
                            duration = (
                                datetime.now() - step_start_times[step_id]
                            ).total_seconds()

                        artifact = workflow._artifacts.get(step_id)
                        attempts = artifact.attempt_number if artifact else 1

                        self._logger.info(
                            f"Step '{step_id}' completed successfully "
                            f"(attempts: {attempts}, duration: {duration:.2f}s)"
                        )

                        self._emit(
                            StepCompletedEvent(
                                timestamp=WorkflowEvent.now(),
                                event_type="step_completed",
                                step_id=step_id,
                                success=True,
                                artifact_id=artifact.artifact_id if artifact else None,
                                total_attempts=attempts,
                                duration_seconds=duration,
                            )
                        )

        # Emit initial step started events for steps with no requirements
        for config in step_configs:
            if not config.requires:
                guard_type = config.guard_config.get("guard", "unknown")
                if guard_type == "composite":
                    guards = config.guard_config.get("guards", [])
                    guard_type = f"composite({', '.join(guards)})"

                self._logger.info(f"Starting step '{config.step_id}'")

                self._emit(
                    StepStartedEvent(
                        timestamp=WorkflowEvent.now(),
                        event_type="step_started",
                        step_id=config.step_id,
                        guard_type=guard_type,
                        requires=config.requires,
                        attempt=1,
                    )
                )

        # Execute workflow
        try:
            result = workflow.execute(specification)

            # Emit final state events
            emit_step_events()

            # Emit any remaining step completions based on result
            if result.status == WorkflowStatus.FAILED and result.failed_step:
                failed_step = result.failed_step
                if failed_step not in completed_steps:
                    duration = 0.0
                    if failed_step in step_start_times:
                        duration = (
                            datetime.now() - step_start_times[failed_step]
                        ).total_seconds()

                    self._emit(
                        StepCompletedEvent(
                            timestamp=WorkflowEvent.now(),
                            event_type="step_completed",
                            step_id=failed_step,
                            success=False,
                            artifact_id=None,
                            total_attempts=self.rmax,
                            duration_seconds=duration,
                        )
                    )

            return result

        except (EscalationRequired, RmaxExhausted):
            # Re-raise these as they're expected workflow exceptions
            raise

    def _get_guard_type_str(self, config: dict[str, Any]) -> str:
        """Get human-readable guard type string."""
        guard_type = str(config.get("guard", "unknown"))
        if guard_type == "composite":
            guards = config.get("guards", [])
            return f"composite({', '.join(guards)})"
        return guard_type
