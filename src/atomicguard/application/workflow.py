"""
Workflow: Orchestrates ActionPair execution across multiple steps.

Owns WorkflowState and infers preconditions from step dependencies.

Extension 09: Supports escalation via informed backtracking when
stagnation is detected, including cascade invalidation of dependent steps.
Supports guard-specific escalation routing via escalation_by_guard (Definition 45).
"""

import logging
import uuid
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.agent import DualStateAgent
from atomicguard.domain.exceptions import (
    EscalationRequired,
    RmaxExhausted,
    StagnationDetected,
)
from atomicguard.domain.interfaces import (
    ArtifactDAGInterface,
    CheckpointDAGInterface,
)
from atomicguard.domain.models import (
    AmendmentType,
    Artifact,
    ArtifactSource,
    ArtifactStatus,
    ContextSnapshot,
    FailureType,
    FeedbackEntry,
    HumanAmendment,
    WorkflowCheckpoint,
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
        Used by CheckpointService for W_ref computation.

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


class ResumableWorkflow(Workflow):
    """
    Workflow with checkpoint and resume support.

    .. deprecated::
        Use Workflow + CheckpointService + WorkflowResumeService instead.
        This class is maintained for backwards compatibility but will be
        removed in a future version.

    Extends Workflow with:
    - Automatic checkpoint creation on failure
    - Resume from checkpoint with human amendment
    - Provenance tracking through amendments
    """

    def __init__(
        self,
        artifact_dag: ArtifactDAGInterface,
        checkpoint_dag: CheckpointDAGInterface,
        rmax: int = 3,
        constraints: str = "",
        auto_checkpoint: bool = True,
    ):
        """
        Args:
            artifact_dag: Repository for storing artifacts
            checkpoint_dag: Repository for storing checkpoints
            rmax: Maximum retries per step
            constraints: Global constraints for the ambient environment
            auto_checkpoint: Create checkpoint on failure (default True)
        """
        warnings.warn(
            "ResumableWorkflow is deprecated. Use Workflow + CheckpointService + "
            "WorkflowResumeService instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(artifact_dag, rmax, constraints)

        self._checkpoint_dag = checkpoint_dag
        self._auto_checkpoint = auto_checkpoint
        self._current_specification: str = ""
        self._current_workflow_id: str = ""

    def execute(self, specification: str) -> WorkflowResult:
        """
        Execute workflow with automatic checkpointing on failure.

        Args:
            specification: The task specification

        Returns:
            WorkflowResult (may include checkpoint if failed)
        """
        self._current_specification = specification
        self._current_workflow_id = str(uuid.uuid4())

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

            # Execute via stateless agent
            agent = DualStateAgent(
                action_pair=step.action_pair,
                artifact_dag=self._dag,
                rmax=self._rmax,
                constraints=self._constraints,
                action_pair_id=step.guard_id,
                workflow_id=self._current_workflow_id,
            )

            try:
                artifact = agent.execute(specification, dependencies)
                self._artifacts[step.guard_id] = artifact
                self._workflow_state.satisfy(step.guard_id, artifact.artifact_id)

            except EscalationRequired as e:
                checkpoint = (
                    self._create_checkpoint(
                        failed_step=step.guard_id,
                        failure_type=FailureType.ESCALATION,
                        feedback=e.feedback,
                        failed_artifact=e.artifact,
                        provenance=[],
                    )
                    if self._auto_checkpoint
                    else None
                )

                return WorkflowResult(
                    status=WorkflowStatus.CHECKPOINT
                    if checkpoint
                    else WorkflowStatus.ESCALATION,
                    artifacts=self._artifacts,
                    failed_step=step.guard_id,
                    escalation_artifact=e.artifact,
                    escalation_feedback=e.feedback,
                    checkpoint=checkpoint,
                )

            except RmaxExhausted as e:
                checkpoint = (
                    self._create_checkpoint(
                        failed_step=step.guard_id,
                        failure_type=FailureType.RMAX_EXHAUSTED,
                        feedback=e.provenance[-1][1] if e.provenance else str(e),
                        failed_artifact=e.provenance[-1][0] if e.provenance else None,
                        provenance=e.provenance,
                    )
                    if self._auto_checkpoint
                    else None
                )

                return WorkflowResult(
                    status=WorkflowStatus.CHECKPOINT
                    if checkpoint
                    else WorkflowStatus.FAILED,
                    artifacts=self._artifacts,
                    failed_step=step.guard_id,
                    provenance=tuple(e.provenance),
                    checkpoint=checkpoint,
                )

        return WorkflowResult(status=WorkflowStatus.SUCCESS, artifacts=self._artifacts)

    def resume(
        self,
        checkpoint_id: str,
        amendment: HumanAmendment,
    ) -> WorkflowResult:
        """
        Resume workflow from checkpoint with human amendment.

        Args:
            checkpoint_id: ID of checkpoint to resume from
            amendment: Human-provided amendment

        Returns:
            WorkflowResult (may include new checkpoint if fails again)
        """
        # 1. Load and validate checkpoint
        checkpoint = self._checkpoint_dag.get_checkpoint(checkpoint_id)

        # 2. Store amendment
        self._checkpoint_dag.store_amendment(amendment)

        # 3. Restore state from checkpoint
        self._restore_state_from_checkpoint(checkpoint)
        self._current_specification = checkpoint.specification
        self._current_workflow_id = checkpoint.workflow_id

        # 4. Find the failed step
        step = self._find_step_by_id(checkpoint.failed_step)
        if step is None:
            raise ValueError(f"Step not found: {checkpoint.failed_step}")

        # 5. Handle amendment based on type
        if amendment.amendment_type == AmendmentType.ARTIFACT:
            # Human provided artifact - validate it directly
            human_artifact = self._create_amendment_artifact(
                amendment, checkpoint, step
            )
            self._dag.store(human_artifact)

            # Run guard on human artifact
            dependencies = {
                gid: self._artifacts[gid] for gid in step.deps if gid in self._artifacts
            }

            result = step.action_pair.guard.validate(human_artifact, **dependencies)

            if result.passed:
                # Update artifact status and proceed
                updated_artifact = replace(
                    human_artifact,
                    status=ArtifactStatus.ACCEPTED,
                    guard_result=result,  # Store full GuardResult
                )
                self._dag.store(updated_artifact)
                self._artifacts[step.guard_id] = updated_artifact
                self._workflow_state.satisfy(
                    step.guard_id, updated_artifact.artifact_id
                )
            else:
                # Guard failed human artifact - create new checkpoint
                new_checkpoint = (
                    self._create_checkpoint(
                        failed_step=step.guard_id,
                        failure_type=FailureType.ESCALATION,
                        feedback=result.feedback,
                        failed_artifact=human_artifact,
                        provenance=[],
                    )
                    if self._auto_checkpoint
                    else None
                )

                return WorkflowResult(
                    status=WorkflowStatus.CHECKPOINT
                    if new_checkpoint
                    else WorkflowStatus.FAILED,
                    artifacts=self._artifacts,
                    failed_step=step.guard_id,
                    escalation_feedback=result.feedback,
                    checkpoint=new_checkpoint,
                )

        elif amendment.amendment_type == AmendmentType.FEEDBACK:
            # Human provided feedback - inject into context and retry with agent
            # Use additional_rmax from amendment
            effective_rmax = self._rmax + amendment.additional_rmax

            dependencies = {
                gid: self._artifacts[gid] for gid in step.deps if gid in self._artifacts
            }

            agent = DualStateAgent(
                action_pair=step.action_pair,
                artifact_dag=self._dag,
                rmax=effective_rmax,
                constraints=self._constraints + "\n\n" + amendment.content,
                action_pair_id=step.guard_id,
                workflow_id=self._current_workflow_id,
            )

            try:
                artifact = agent.execute(
                    self._current_specification
                    + "\n\n[Human Feedback]: "
                    + amendment.content,
                    dependencies,
                )
                self._artifacts[step.guard_id] = artifact
                self._workflow_state.satisfy(step.guard_id, artifact.artifact_id)

            except (EscalationRequired, RmaxExhausted) as e:
                # Create new checkpoint for the new failure
                if isinstance(e, EscalationRequired):
                    new_checkpoint = (
                        self._create_checkpoint(
                            failed_step=step.guard_id,
                            failure_type=FailureType.ESCALATION,
                            feedback=e.feedback,
                            failed_artifact=e.artifact,
                            provenance=[],
                        )
                        if self._auto_checkpoint
                        else None
                    )
                else:
                    new_checkpoint = (
                        self._create_checkpoint(
                            failed_step=step.guard_id,
                            failure_type=FailureType.RMAX_EXHAUSTED,
                            feedback=e.provenance[-1][1] if e.provenance else str(e),
                            failed_artifact=e.provenance[-1][0]
                            if e.provenance
                            else None,
                            provenance=e.provenance,
                        )
                        if self._auto_checkpoint
                        else None
                    )

                return WorkflowResult(
                    status=WorkflowStatus.CHECKPOINT
                    if new_checkpoint
                    else WorkflowStatus.FAILED,
                    artifacts=self._artifacts,
                    failed_step=step.guard_id,
                    checkpoint=new_checkpoint,
                )

        # 6. Continue executing remaining steps
        return self._continue_execution()

    def _create_checkpoint(
        self,
        failed_step: str,
        failure_type: FailureType,
        feedback: str,
        failed_artifact: Artifact | None,
        provenance: list[tuple[Artifact, str]],
    ) -> WorkflowCheckpoint:
        """Create and persist a checkpoint."""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            workflow_id=self._current_workflow_id,
            created_at=datetime.now(UTC).isoformat(),
            specification=self._current_specification,
            constraints=self._constraints,
            rmax=self._rmax,
            completed_steps=tuple(
                step.guard_id
                for step in self._steps
                if self._workflow_state.is_satisfied(step.guard_id)
            ),
            artifact_ids=tuple(
                (gid, art.artifact_id) for gid, art in self._artifacts.items()
            ),
            failure_type=failure_type,
            failed_step=failed_step,
            failed_artifact_id=failed_artifact.artifact_id if failed_artifact else None,
            failure_feedback=feedback,
            provenance_ids=tuple(a.artifact_id for a, _ in provenance),
        )
        self._checkpoint_dag.store_checkpoint(checkpoint)
        return checkpoint

    def _restore_state_from_checkpoint(self, checkpoint: WorkflowCheckpoint) -> None:
        """Restore workflow state from checkpoint."""
        # Restore satisfied guards
        for guard_id in checkpoint.completed_steps:
            self._workflow_state.guards[guard_id] = True

        # Restore artifact references
        for guard_id, artifact_id in checkpoint.artifact_ids:
            artifact = self._dag.get_artifact(artifact_id)
            self._artifacts[guard_id] = artifact
            self._workflow_state.artifact_ids[guard_id] = artifact_id

    def _find_step_by_id(self, guard_id: str) -> WorkflowStep | None:
        """Find a step by its guard_id."""
        for step in self._steps:
            if step.guard_id == guard_id:
                return step
        return None

    def _create_amendment_artifact(
        self,
        amendment: HumanAmendment,
        checkpoint: WorkflowCheckpoint,
        step: WorkflowStep,
    ) -> Artifact:
        """Create an artifact from human amendment."""
        # Determine attempt number
        latest = self._dag.get_latest_for_action_pair(
            step.guard_id, checkpoint.workflow_id
        )
        attempt_number = (latest.attempt_number + 1) if latest else 1

        # Build context snapshot
        context = ContextSnapshot(
            workflow_id=checkpoint.workflow_id,
            specification=checkpoint.specification,
            constraints=checkpoint.constraints,
            feedback_history=tuple(
                FeedbackEntry(artifact_id=aid, feedback=checkpoint.failure_feedback)
                for aid in checkpoint.provenance_ids
            ),
            dependency_artifacts=tuple(
                (gid, art.artifact_id) for gid, art in self._artifacts.items()
            ),
        )

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            workflow_id=checkpoint.workflow_id,
            content=amendment.content,
            previous_attempt_id=checkpoint.failed_artifact_id,
            parent_action_pair_id=None,
            action_pair_id=step.guard_id,
            created_at=datetime.now(UTC).isoformat(),
            attempt_number=attempt_number,
            status=ArtifactStatus.PENDING,
            guard_result=None,  # Guard result set after validation
            context=context,
            source=ArtifactSource.HUMAN,
        )

    def _continue_execution(self) -> WorkflowResult:
        """Continue executing remaining steps after resume."""
        while not self._is_goal_state():
            step = self._find_applicable()

            if step is None:
                return WorkflowResult(
                    status=WorkflowStatus.FAILED,
                    artifacts=self._artifacts,
                    failed_step="No applicable step",
                )

            dependencies = {
                gid: self._artifacts[gid] for gid in step.deps if gid in self._artifacts
            }

            agent = DualStateAgent(
                action_pair=step.action_pair,
                artifact_dag=self._dag,
                rmax=self._rmax,
                constraints=self._constraints,
                action_pair_id=step.guard_id,
                workflow_id=self._current_workflow_id,
            )

            try:
                artifact = agent.execute(self._current_specification, dependencies)
                self._artifacts[step.guard_id] = artifact
                self._workflow_state.satisfy(step.guard_id, artifact.artifact_id)

            except EscalationRequired as e:
                checkpoint = (
                    self._create_checkpoint(
                        failed_step=step.guard_id,
                        failure_type=FailureType.ESCALATION,
                        feedback=e.feedback,
                        failed_artifact=e.artifact,
                        provenance=[],
                    )
                    if self._auto_checkpoint
                    else None
                )

                return WorkflowResult(
                    status=WorkflowStatus.CHECKPOINT
                    if checkpoint
                    else WorkflowStatus.ESCALATION,
                    artifacts=self._artifacts,
                    failed_step=step.guard_id,
                    escalation_artifact=e.artifact,
                    escalation_feedback=e.feedback,
                    checkpoint=checkpoint,
                )

            except RmaxExhausted as e:
                checkpoint = (
                    self._create_checkpoint(
                        failed_step=step.guard_id,
                        failure_type=FailureType.RMAX_EXHAUSTED,
                        feedback=e.provenance[-1][1] if e.provenance else str(e),
                        failed_artifact=e.provenance[-1][0] if e.provenance else None,
                        provenance=e.provenance,
                    )
                    if self._auto_checkpoint
                    else None
                )

                return WorkflowResult(
                    status=WorkflowStatus.CHECKPOINT
                    if checkpoint
                    else WorkflowStatus.FAILED,
                    artifacts=self._artifacts,
                    failed_step=step.guard_id,
                    provenance=tuple(e.provenance),
                    checkpoint=checkpoint,
                )

        return WorkflowResult(status=WorkflowStatus.SUCCESS, artifacts=self._artifacts)
