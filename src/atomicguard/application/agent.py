"""
DualStateAgent: Stateless executor for a single ActionPair.

Manages only EnvironmentState (the retry loop).
WorkflowState is managed by Workflow.
"""

from dataclasses import replace

from atomicguard.application.action_pair import ActionPair
from atomicguard.domain.exceptions import EscalationRequired, RmaxExhausted
from atomicguard.domain.interfaces import ArtifactDAGInterface
from atomicguard.domain.models import (
    AmbientEnvironment,
    Artifact,
    ArtifactStatus,
    Context,
    FeedbackEntry,
)


class DualStateAgent:
    """
    Stateless executor for a single ActionPair.

    Manages only EnvironmentState (retry loop).
    WorkflowState is managed by Workflow.

    Dependencies from prior workflow steps are passed to both:
    - Generator (via Context.dependencies) - so it can see prior artifacts
    - Guard (via validate(**deps)) - for validation against prior artifacts
    """

    def __init__(
        self,
        action_pair: ActionPair,
        artifact_dag: ArtifactDAGInterface,
        rmax: int = 3,
        constraints: str = "",
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
    ):
        """
        Args:
            action_pair: The generator-guard pair to execute
            artifact_dag: Repository for storing artifacts
            rmax: Maximum retry attempts (default: 3)
            constraints: Global constraints for the ambient environment
            action_pair_id: Identifier for this action pair (e.g., 'g_test')
            workflow_id: UUID of the workflow execution instance
        """
        self._action_pair = action_pair
        self._artifact_dag = artifact_dag
        self._rmax = rmax
        self._constraints = constraints
        self._action_pair_id = action_pair_id
        self._workflow_id = workflow_id

    def execute(
        self,
        specification: str,
        dependencies: dict[str, Artifact] | None = None,
    ) -> Artifact:
        """
        Execute the action pair with retry logic.

        Args:
            specification: The task specification
            dependencies: Artifacts from prior workflow steps (key -> Artifact)
                         Passed to both generator (via Context) and guard (via validate)

        Returns:
            The accepted artifact

        Raises:
            RmaxExhausted: If all retries fail
        """
        dependencies = dependencies or {}
        context = self._compose_context(specification, dependencies)
        feedback_history: list[tuple[Artifact, str]] = []
        retry_count = 0
        previous_id: str | None = None  # Track chain linkage

        while retry_count <= self._rmax:
            artifact, result = self._action_pair.execute(
                context, dependencies, self._action_pair_id, self._workflow_id
            )

            # Build feedback history for context snapshot
            fb_entries = tuple(
                FeedbackEntry(artifact_id=a.artifact_id, feedback=f)
                for a, f in feedback_history
            )

            # Update artifact with full GuardResult (Extension 08: Composite Guards)
            artifact = replace(
                artifact,
                previous_attempt_id=previous_id,
                status=ArtifactStatus.ACCEPTED
                if result.passed
                else ArtifactStatus.REJECTED,
                guard_result=result,  # Store full GuardResult, not just bool
                context=replace(
                    artifact.context,
                    feedback_history=fb_entries,
                ),
            )
            self._artifact_dag.store(artifact)

            if result.passed:
                return artifact
            elif result.fatal:
                # Non-recoverable failure - escalate immediately
                raise EscalationRequired(artifact, result.feedback)
            else:
                # Recoverable failure - retry
                feedback_history.append((artifact, result.feedback))
                previous_id = artifact.artifact_id  # Track for next iteration
                retry_count += 1
                context = self._refine_context(
                    specification, artifact, feedback_history, dependencies
                )

        raise RmaxExhausted(
            f"Failed after {self._rmax} retries", provenance=feedback_history
        )

    def _compose_context(
        self,
        specification: str,
        dependencies: dict[str, Artifact] | None = None,
    ) -> Context:
        """Compose initial context with dependencies."""
        dependencies = dependencies or {}
        ambient = AmbientEnvironment(
            repository=self._artifact_dag, constraints=self._constraints
        )
        return Context(
            ambient=ambient,
            specification=specification,
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=tuple(
                (k, v.artifact_id) for k, v in dependencies.items()
            ),  # Store IDs, not full artifacts
        )

    def _refine_context(
        self,
        specification: str,
        artifact: Artifact,
        feedback_history: list[tuple[Artifact, str]],
        dependencies: dict[str, Artifact] | None = None,
    ) -> Context:
        """Refine context with feedback from failed attempt."""
        dependencies = dependencies or {}
        ambient = AmbientEnvironment(
            repository=self._artifact_dag, constraints=self._constraints
        )
        return Context(
            ambient=ambient,
            specification=specification,
            current_artifact=artifact.content,
            feedback_history=tuple((a.content, f) for a, f in feedback_history),
            dependency_artifacts=tuple(
                (k, v.artifact_id) for k, v in dependencies.items()
            ),  # Preserve dependencies on retry
        )
