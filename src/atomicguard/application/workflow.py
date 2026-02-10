"""
Workflow: Orchestrates ActionPair execution across multiple steps.

Owns WorkflowState and infers preconditions from step dependencies.

Extension 09: Supports escalation via informed backtracking when
stagnation is detected, including cascade invalidation of dependent steps.
Supports guard-specific escalation routing via escalation_by_guard (Definition 45).
"""

import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from types import MappingProxyType

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.agent import DualStateAgent
from atomicguard.domain.exceptions import (
    EscalationRequired,
    RmaxExhausted,
    StagnationDetected,
)
from atomicguard.domain.interfaces import ArtifactDAGInterface
from atomicguard.domain.models import (
    Artifact,
    WorkflowResult,
    WorkflowState,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowStep:
    """Internal step representation."""

    guard_id: str
    action_pair: ActionPair
    requires: tuple[str, ...]
    deps: tuple[str, ...]  # Artifacts to pass to guard
    # Extension 09: Escalation parameters
    r_patience: int | None = None  # Consecutive similar failures before escalation
    e_max: int = 1  # Maximum escalation attempts before FAIL
    escalation: tuple[str, ...] = ()  # Default upstream action_pair_ids to re-invoke
    escalation_by_guard: MappingProxyType[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )  # Definition 45: guard-specific routing


class Workflow:
    """
    Orchestrates ActionPair execution.

    Owns WorkflowState and infers preconditions from requires.
    """

    def __init__(
        self,
        artifact_dag: ArtifactDAGInterface,
        rmax: int = 3,
        constraints: str = "",
    ):
        """
        Args:
            artifact_dag: Repository for storing artifacts
            rmax: Maximum retries per step
            constraints: Global constraints for the ambient environment
        """
        self._dag = artifact_dag
        self._rmax = rmax
        self._constraints = constraints
        self._steps: list[WorkflowStep] = []
        self._workflow_state = WorkflowState()
        self._artifacts: dict[str, Artifact] = {}
        # Extension 09: Escalation tracking
        self._escalation_count: dict[str, int] = defaultdict(int)  # step_id -> count
        self._escalation_context: dict[str, str] = {}  # step_id -> failure_summary

    def add_step(
        self,
        guard_id: str,
        action_pair: ActionPair,
        requires: tuple[str, ...] = (),
        deps: tuple[str, ...] | None = None,
        r_patience: int | None = None,
        e_max: int = 1,
        escalation: tuple[str, ...] = (),
        escalation_by_guard: dict[str, tuple[str, ...]] | None = None,
    ) -> "Workflow":
        """
        Register a step. Precondition inferred from requires.

        Args:
            guard_id: Unique identifier for this step (e.g., 'g_test', 'g_impl')
            action_pair: The generator-guard pair for this step
            requires: Guard IDs that must be satisfied before this step
            deps: Artifact dependencies to pass to guard (defaults to requires)
            r_patience: Consecutive similar failures before escalation (Extension 09).
                        If None, escalation is disabled for this step.
            e_max: Maximum escalation attempts before FAIL (Extension 09, default: 1)
            escalation: Default upstream action_pair_ids to re-invoke on stagnation
            escalation_by_guard: Per-guard escalation targets (Definition 45)

        Returns:
            Self for fluent chaining
        """
        if deps is None:
            deps = requires

        # Extension 09: Validate r_patience < rmax invariant (Definition 44)
        if r_patience is not None and r_patience >= self._rmax:
            raise ValueError(
                f"r_patience ({r_patience}) must be < rmax ({self._rmax}). "
                f"Extension 09 invariant: 1 < r_patience < rmax"
            )

        self._steps.append(
            WorkflowStep(
                guard_id=guard_id,
                action_pair=action_pair,
                requires=requires,
                deps=deps,
                r_patience=r_patience,
                e_max=e_max,
                escalation=escalation,
                escalation_by_guard=MappingProxyType(escalation_by_guard or {}),
            )
        )
        return self  # Fluent

    def execute(self, specification: str) -> WorkflowResult:
        """
        Execute the workflow until completion or failure.

        Extension 09: Supports escalation via informed backtracking.

        Args:
            specification: The task specification

        Returns:
            WorkflowResult with success status and artifacts
        """
        # Generate a unique workflow_id for this execution
        workflow_id = str(uuid.uuid4())

        # Extension 09: Validate escalation targets exist before execution
        step_ids = {s.guard_id for s in self._steps}
        for s in self._steps:
            for target in s.escalation:
                if target not in step_ids:
                    raise ValueError(
                        f"Escalation target '{target}' not found in workflow steps"
                    )

        while not self._is_goal_state():
            step = self._find_applicable()

            if step is None:
                return WorkflowResult(
                    status=WorkflowStatus.FAILED,
                    artifacts=self._artifacts,
                    failed_step="No applicable step",
                )

            # Extract dependencies (keys match action_pair_ids from workflow.json)
            dependencies = {
                gid: self._artifacts[gid] for gid in step.deps if gid in self._artifacts
            }

            # Extension 09: Get any injected failure context
            effective_constraints = self._constraints
            if step.guard_id in self._escalation_context:
                failure_ctx = self._escalation_context.pop(step.guard_id)
                effective_constraints = f"{self._constraints}\n\n{failure_ctx}"
                logger.info(
                    "[%s] Re-running with injected failure context from escalation",
                    step.guard_id,
                )

            # Execute via stateless agent
            agent = DualStateAgent(
                action_pair=step.action_pair,
                artifact_dag=self._dag,
                rmax=self._rmax,
                constraints=effective_constraints,
                action_pair_id=step.guard_id,
                workflow_id=workflow_id,
                r_patience=step.r_patience,
                e_max=step.e_max,
                escalation=list(step.escalation),
                escalation_by_guard={
                    k: list(v) for k, v in step.escalation_by_guard.items()
                },
            )

            try:
                artifact = agent.execute(specification, dependencies)
                self._artifacts[step.guard_id] = artifact
                self._workflow_state.satisfy(step.guard_id, artifact.artifact_id)

            except StagnationDetected as e:
                # Level 2: Workflow backtracking
                if self._escalation_count[step.guard_id] < step.e_max:
                    logger.info(
                        "[%s] Escalation %d/%d triggered. Targets: %s",
                        step.guard_id,
                        self._escalation_count[step.guard_id] + 1,
                        step.e_max,
                        e.escalate_to,
                    )

                    # Definition 47: Cascade Invalidation - invalidate ALL targets
                    for target_id in e.escalate_to:
                        self._invalidate_dependents(target_id)
                        # Definition 48: Context Injection for each target
                        if e.failure_summary:
                            self._inject_failure_context(target_id, e.failure_summary)

                    self._escalation_count[step.guard_id] += 1

                    continue  # Re-run from earliest invalidated step
                else:
                    # e_max exceeded - promote to human escalation
                    return WorkflowResult(
                        status=WorkflowStatus.ESCALATION,
                        artifacts=self._artifacts,
                        failed_step=step.guard_id,
                        escalation_feedback=(
                            f"Automated escalation exhausted ({step.e_max} attempts). "
                            f"Failure summary: {e.failure_summary}"
                        ),
                    )

            except EscalationRequired as e:
                # Level 4: Human intervention (fatal guard)
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

    def _invalidate_dependents(self, target_id: str) -> None:
        """Mark target and all transitive dependents as unsatisfied (Definition 47).

        When escalating to an upstream step, all steps that depend on it
        (directly or transitively) must be invalidated so they re-execute
        with fresh artifacts.

        Args:
            target_id: The action_pair_id to invalidate (and its dependents)
        """
        to_invalidate = self._get_transitive_dependents(target_id)
        to_invalidate.add(target_id)  # Include target itself

        for step_id in to_invalidate:
            if self._workflow_state.is_satisfied(step_id):
                logger.debug("[%s] Cascade invalidation", step_id)
                self._workflow_state.unsatisfy(step_id)
                self._artifacts.pop(step_id, None)

    def _get_transitive_dependents(self, target_id: str) -> set[str]:
        """Find all steps that transitively depend on target_id (Definition 47).

        Uses BFS to find all steps that have target_id in their requires chain.

        Args:
            target_id: The action_pair_id to find dependents for

        Returns:
            Set of guard_ids that depend on target (directly or transitively)
        """
        dependents: set[str] = set()
        queue: deque[str] = deque([target_id])

        while queue:
            current = queue.popleft()
            for step in self._steps:
                if current in step.requires and step.guard_id not in dependents:
                    dependents.add(step.guard_id)
                    queue.append(step.guard_id)

        return dependents

    def _inject_failure_context(self, target_id: str, summary: str) -> None:
        """Add failure summary to constraints for target re-execution (Definition 48).

        The failure summary describes what went wrong in downstream steps,
        enabling the upstream generator to produce output that avoids
        the same failure patterns.

        Args:
            target_id: The action_pair_id to inject context for
            summary: The failure summary to inject
        """
        self._escalation_context[target_id] = summary
        logger.debug(
            "[%s] Failure context injected for re-execution",
            target_id,
        )

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

    def get_workflow_definition(self) -> dict:
        """Build workflow definition dict for W_ref computation.

        Returns a serializable dict representing the workflow structure.
        Used for W_ref computation (Definition 11).

        Returns:
            Dict with steps, rmax, and constraints for hashing.
        """
        return {
            "steps": [
                {
                    "guard_id": step.guard_id,
                    "requires": list(step.requires),
                    "deps": list(step.deps),
                }
                for step in self._steps
            ],
            "rmax": self._rmax,
            "constraints": self._constraints,
        }

    def get_step(self, guard_id: str) -> WorkflowStep:
        """Get a workflow step by guard_id.

        Args:
            guard_id: The identifier of the step to retrieve.

        Returns:
            The WorkflowStep with the given guard_id.

        Raises:
            KeyError: If no step with the given guard_id exists.
        """
        for step in self._steps:
            if step.guard_id == guard_id:
                return step
        raise KeyError(f"Step not found: {guard_id}")
