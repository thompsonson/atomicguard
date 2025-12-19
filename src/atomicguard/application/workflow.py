"""
Workflow: Orchestrates ActionPair execution across multiple steps.

Owns WorkflowState and infers preconditions from step dependencies.
"""

from dataclasses import dataclass

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.agent import DualStateAgent
from atomicguard.domain.exceptions import EscalationRequired, RmaxExhausted
from atomicguard.domain.interfaces import ArtifactDAGInterface
from atomicguard.domain.models import (
    Artifact,
    WorkflowResult,
    WorkflowState,
    WorkflowStatus,
)


@dataclass(frozen=True)
class WorkflowStep:
    """Internal step representation."""

    guard_id: str
    action_pair: ActionPair
    requires: tuple[str, ...]
    deps: tuple[str, ...]  # Artifacts to pass to guard


class Workflow:
    """
    Orchestrates ActionPair execution.

    Owns WorkflowState and infers preconditions from requires.
    """

    def __init__(
        self,
        artifact_dag: ArtifactDAGInterface | None = None,
        rmax: int = 3,
        constraints: str = "",
    ):
        """
        Args:
            artifact_dag: Repository for storing artifacts (creates InMemory if None)
            rmax: Maximum retries per step
            constraints: Global constraints for the ambient environment
        """
        # Lazy import to avoid circular dependency
        if artifact_dag is None:
            from atomicguard.infrastructure.persistence import InMemoryArtifactDAG

            artifact_dag = InMemoryArtifactDAG()

        self._dag = artifact_dag
        self._rmax = rmax
        self._constraints = constraints
        self._steps: list[WorkflowStep] = []
        self._workflow_state = WorkflowState()
        self._artifacts: dict[str, Artifact] = {}

    def add_step(
        self,
        guard_id: str,
        action_pair: ActionPair,
        requires: tuple[str, ...] = (),
        deps: tuple[str, ...] | None = None,
    ) -> "Workflow":
        """
        Register a step. Precondition inferred from requires.

        Args:
            guard_id: Unique identifier for this step (e.g., 'g_test', 'g_impl')
            action_pair: The generator-guard pair for this step
            requires: Guard IDs that must be satisfied before this step
            deps: Artifact dependencies to pass to guard (defaults to requires)

        Returns:
            Self for fluent chaining
        """
        if deps is None:
            deps = requires
        self._steps.append(WorkflowStep(guard_id, action_pair, requires, deps))
        return self  # Fluent

    def execute(self, specification: str) -> WorkflowResult:
        """
        Execute the workflow until completion or failure.

        Args:
            specification: The task specification

        Returns:
            WorkflowResult with success status and artifacts
        """
        while not self._is_goal_state():
            step = self._find_applicable()

            if step is None:
                return WorkflowResult(
                    status=WorkflowStatus.FAILED,
                    artifacts=self._artifacts,
                    failed_step="No applicable step",
                )

            # Extract dependencies
            dependencies = {
                gid.replace("g_", ""): self._artifacts[gid]
                for gid in step.deps
                if gid in self._artifacts
            }

            # Execute via stateless agent
            agent = DualStateAgent(
                action_pair=step.action_pair,
                artifact_dag=self._dag,
                rmax=self._rmax,
                constraints=self._constraints,
            )

            try:
                artifact = agent.execute(specification, dependencies)
                self._artifacts[step.guard_id] = artifact
                self._workflow_state.satisfy(step.guard_id, artifact.artifact_id)

            except EscalationRequired as e:
                return WorkflowResult(
                    status=WorkflowStatus.ESCALATION,
                    artifacts=self._artifacts,
                    failed_step=step.guard_id,
                    escalation_artifact=e.artifact,
                    escalation_feedback=e.feedback,
                )

            except RmaxExhausted as e:
                return WorkflowResult(
                    status=WorkflowStatus.FAILED,
                    artifacts=self._artifacts,
                    failed_step=step.guard_id,
                    provenance=tuple(e.provenance),
                )

        return WorkflowResult(status=WorkflowStatus.SUCCESS, artifacts=self._artifacts)

    def _precondition_met(self, step: WorkflowStep) -> bool:
        """Precondition: all required guards satisfied."""
        return all(self._workflow_state.is_satisfied(req) for req in step.requires)

    def _find_applicable(self) -> WorkflowStep | None:
        """Find first step not done and with precondition met."""
        for step in self._steps:
            if not self._workflow_state.is_satisfied(
                step.guard_id
            ) and self._precondition_met(step):
                return step
        return None

    def _is_goal_state(self) -> bool:
        """Check if all steps are satisfied."""
        return all(
            self._workflow_state.is_satisfied(step.guard_id) for step in self._steps
        )
