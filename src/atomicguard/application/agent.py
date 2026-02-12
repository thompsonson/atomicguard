"""
DualStateAgent: Stateless executor for a single ActionPair.

Manages only EnvironmentState (the retry loop).
WorkflowState is managed by Workflow.

Extension 09: Supports escalation via informed backtracking when
stagnation is detected (r_patience consecutive similar failures).
Raises StagnationDetected for Level 2 (workflow backtracking) and
EscalationRequired for Level 4 (human intervention).

Supports guard-specific escalation routing via escalation_by_guard
(Definition 45) for composite guards.
"""

import logging
from dataclasses import replace

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.feedback_summarizer import FeedbackSummarizer
from atomicguard.domain.exceptions import (
    EscalationRequired,
    RmaxExhausted,
    StagnationDetected,
)
from atomicguard.domain.interfaces import ArtifactDAGInterface
from atomicguard.domain.models import (
    AmbientEnvironment,
    Artifact,
    ArtifactStatus,
    Context,
    FeedbackEntry,
)

logger = logging.getLogger(__name__)


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
        r_patience: int | None = None,
        e_max: int = 1,
        escalate_feedback_to: list[str] | None = None,
        escalate_feedback_by_guard: dict[str, list[str]] | None = None,
    ):
        """
        Args:
            action_pair: The generator-guard pair to execute
            artifact_dag: Repository for storing artifacts
            rmax: Maximum retry attempts (default: 3)
            constraints: Global constraints for the ambient environment
            action_pair_id: Identifier for this action pair (e.g., 'g_test')
            workflow_id: UUID of the workflow execution instance
            r_patience: Consecutive similar failures before escalation (Extension 09).
                        If None, escalation is disabled. Must be < rmax.
            e_max: Maximum escalation attempts before FAIL (Extension 09, default: 1)
            escalate_feedback_to: Upstream steps to receive failure context on stagnation.
                Cascade invalidation automatically re-runs dependent steps; this controls
                WHICH steps receive the failure summary for informed regeneration.
            escalate_feedback_by_guard: Per-guard feedback routing (Definition 45).
                Maps guard_name -> list of upstream steps to receive failure context.
        """
        self._action_pair = action_pair
        self._artifact_dag = artifact_dag
        self._rmax = rmax
        self._constraints = constraints
        self._action_pair_id = action_pair_id
        self._workflow_id = workflow_id
        # Extension 09: Escalation parameters
        self._r_patience = r_patience
        self._e_max = e_max
        self._escalate_feedback_to = escalate_feedback_to or []
        self._escalate_feedback_by_guard = escalate_feedback_by_guard or {}
        self._feedback_summarizer = FeedbackSummarizer()

    def execute(
        self,
        specification: str,
        dependencies: dict[str, Artifact] | None = None,
        escalation_feedback: tuple[str, ...] = (),
    ) -> Artifact:
        """
        Execute the action pair with retry logic.

        Args:
            specification: The task specification
            dependencies: Artifacts from prior workflow steps (key -> Artifact)
                         Passed to both generator (via Context) and guard (via validate)
            escalation_feedback: Downstream failure summaries from prior escalation cycles

        Returns:
            The accepted artifact

        Raises:
            RmaxExhausted: If all retries fail
            StagnationDetected: If stagnation detected and escalation configured
            EscalationRequired: If guard returns fatal
        """
        dependencies = dependencies or {}

        # Reconstruct prior feedback from DAG for this action pair
        prior_artifacts = self._artifact_dag.get_all_for_action_pair(
            action_pair_id=self._action_pair_id,
            workflow_id=self._workflow_id,
        )
        feedback_history: list[tuple[Artifact, str]] = [
            (a, a.guard_result.feedback)
            for a in prior_artifacts
            if a.status == ArtifactStatus.REJECTED and a.guard_result
        ]

        context = self._compose_context(
            specification, dependencies, escalation_feedback=escalation_feedback
        )
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

                # Extension 09: Check for stagnation and feedback escalation
                stagnation_warning = None
                if self._r_patience is not None:
                    # Try per-guard detection first if escalate_feedback_by_guard configured
                    if self._escalate_feedback_by_guard:
                        stagnation = (
                            self._feedback_summarizer.detect_stagnation_by_guard(
                                feedback_history, self._r_patience
                            )
                        )
                    else:
                        stagnation = self._feedback_summarizer.detect_stagnation(
                            feedback_history, self._r_patience
                        )

                    if stagnation.detected:
                        # Resolve guard-specific targets first, then fallback
                        if (
                            stagnation.stagnant_guard
                            and stagnation.stagnant_guard
                            in self._escalate_feedback_by_guard
                        ):
                            targets = self._escalate_feedback_by_guard[
                                stagnation.stagnant_guard
                            ]
                        else:
                            targets = self._escalate_feedback_to

                        if targets:
                            # Level 2: Workflow backtracking - raise StagnationDetected
                            logger.warning(
                                "[%s] Stagnation detected after %d similar failures. "
                                "Escalating feedback to %s",
                                self._action_pair_id,
                                stagnation.similar_count,
                                targets,
                            )
                            raise StagnationDetected(
                                artifact=artifact,
                                feedback=result.feedback,
                                escalate_to=list(targets),
                                failure_summary=stagnation.failure_summary,
                                stagnant_guard=stagnation.stagnant_guard,
                            )
                        else:
                            # No feedback targets - inject warning and continue retrying
                            stagnation_warning = (
                                f"Stagnation detected: {stagnation.similar_count} similar failures. "
                                f"Pattern: {stagnation.error_signature}"
                            )
                            logger.warning(
                                "[%s] %s (no escalate_feedback_to targets configured)",
                                self._action_pair_id,
                                stagnation_warning,
                            )

                context = self._refine_context(
                    specification,
                    artifact,
                    feedback_history,
                    dependencies,
                    stagnation_warning=stagnation_warning,
                    escalation_feedback=escalation_feedback,
                )

        raise RmaxExhausted(
            f"Failed after {self._rmax} retries", provenance=feedback_history
        )

    def _compose_context(
        self,
        specification: str,
        dependencies: dict[str, Artifact] | None = None,
        escalation_feedback: tuple[str, ...] = (),
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
            escalation_feedback=escalation_feedback,
        )

    def _refine_context(
        self,
        specification: str,
        artifact: Artifact,
        feedback_history: list[tuple[Artifact, str]],
        dependencies: dict[str, Artifact] | None = None,
        stagnation_warning: str | None = None,
        escalation_feedback: tuple[str, ...] = (),
    ) -> Context:
        """Refine context with feedback from failed attempt."""
        dependencies = dependencies or {}

        constraints = self._constraints
        if stagnation_warning:
            constraints = (
                f"{constraints}\n\n"
                f"## STAGNATION WARNING\n{stagnation_warning}\n"
                "Your previous approaches have been producing the same failure. "
                "You MUST try a fundamentally different fix strategy."
            )

        ambient = AmbientEnvironment(
            repository=self._artifact_dag, constraints=constraints
        )
        return Context(
            ambient=ambient,
            specification=specification,
            current_artifact=artifact.content,
            feedback_history=tuple((a.content, f) for a, f in feedback_history),
            dependency_artifacts=tuple(
                (k, v.artifact_id) for k, v in dependencies.items()
            ),  # Preserve dependencies on retry
            escalation_feedback=escalation_feedback,
        )
