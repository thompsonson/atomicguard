"""
Domain models for the Dual-State Framework.

These are pure data structures aligned with paper Definitions 4-6.
All models are immutable (frozen dataclasses) to ensure referential transparency.
"""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import ArtifactDAGInterface


# =============================================================================
# GUARD RESULT (moved before Artifact for forward reference)
# =============================================================================


@dataclass(frozen=True)
class SubGuardOutcome:
    """Outcome from a single sub-guard within a composite (Definition 43).

    Captures individual guard result for attribution in composite validations.
    """

    guard_name: str  # Class name of the sub-guard
    passed: bool  # Whether this sub-guard passed
    feedback: str  # Feedback from this sub-guard
    execution_time_ms: float = 0.0  # Time spent in this sub-guard


@dataclass(frozen=True)
class GuardResult:
    """Immutable guard validation outcome (Definition 6).

    G(α, C) → {v, φ} where:
    - v = passed (⊤ or ⊥)
    - φ = feedback signal
    - fatal = ⊥_fatal (non-recoverable, requires escalation)

    Note: Stagnation detection is done by the agent (which sees feedback history),
    not by guards (which are stateless and validate single artifacts).
    """

    passed: bool
    feedback: str = ""
    fatal: bool = False  # ⊥_fatal - skip retry, escalate to human
    guard_name: str | None = None  # Name of the guard that produced this result
    sub_results: tuple[SubGuardOutcome, ...] = ()  # For composite guards (Extension 08)


# =============================================================================
# ARTIFACT MODEL (Definition 4-6)
# =============================================================================


class ArtifactStatus(Enum):
    """Status of an artifact in the DAG."""

    PENDING = "pending"  # Generated, not yet validated
    REJECTED = "rejected"  # Guard returned ⊥
    ACCEPTED = "accepted"  # Guard returned ⊤, final for this step
    SUPERSEDED = "superseded"  # Guard returned ⊤, but later attempt also passed


class ArtifactSource(Enum):
    """Origin of artifact content."""

    GENERATED = "generated"  # LLM-generated
    HUMAN = "human"  # Human-provided during amendment
    IMPORTED = "imported"  # Imported from external source


@dataclass(frozen=True)
class FeedbackEntry:
    """Single entry in feedback history H."""

    artifact_id: str  # Reference to the rejected artifact
    feedback: str  # Guard's rejection message φ


@dataclass(frozen=True)
class ContextSnapshot:
    """Immutable context C that conditioned generation (Definition 5)."""

    workflow_id: str  # UUID of the workflow execution instance
    specification: str  # Ψ - static specification
    constraints: str  # Ω - global constraints
    feedback_history: tuple[FeedbackEntry, ...]  # H - accumulated rejections
    dependency_artifacts: tuple[
        tuple[str, str], ...
    ] = ()  # (action_pair_id, artifact_id) - matches schema


@dataclass(frozen=True)
class Artifact:
    """
    Immutable node in the Versioned Repository DAG (Definition 4).

    Represents a single generation attempt with full provenance tracking.
    """

    # Identity
    artifact_id: str  # Unique identifier (UUID)
    workflow_id: str  # UUID of the workflow execution instance
    content: str  # The generated code/text

    # DAG Structure
    previous_attempt_id: str | None  # Retry chain within same action pair
    parent_action_pair_id: str | None  # Parent hierarchy for composite generators
    # Cross-step deps are in context.dependency_artifacts

    # Action Pair Coupling (Definition 6: A = ⟨ρ, a_gen, G⟩)
    action_pair_id: str  # Which action pair produced this

    # Metadata
    created_at: str  # ISO timestamp
    attempt_number: int  # Attempt within this action pair context
    status: ArtifactStatus  # pending/rejected/accepted/superseded
    guard_result: GuardResult | None  # Full guard result (None if pending)
    context: ContextSnapshot  # Full context snapshot at generation time
    source: ArtifactSource = ArtifactSource.GENERATED  # Origin of content

    # Extension 01: Versioned Environment (Definition 10)
    workflow_ref: str | None = None  # W_ref: Content-addressed workflow hash (Def 11)

    # Extension 07: Incremental Execution (Definition 33)
    config_ref: str | None = (
        None  # Ψ_ref: Configuration fingerprint for change detection
    )

    metadata: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )  # Immutable metadata dict

    def __post_init__(self) -> None:
        """Convert metadata dict to immutable MappingProxyType if needed."""
        # Handle case where metadata is passed as a regular dict
        if isinstance(object.__getattribute__(self, "metadata"), dict):
            object.__setattr__(self, "metadata", MappingProxyType(self.metadata))


# =============================================================================
# CONTEXT AND ENVIRONMENT
# =============================================================================


@dataclass(frozen=True)
class AmbientEnvironment:
    """Ambient Environment E = ⟨R, Ω⟩"""

    repository: "ArtifactDAGInterface"
    constraints: str = ""


@dataclass(frozen=True)
class Context:
    """Immutable hierarchical context composition (Definition 5)."""

    ambient: AmbientEnvironment
    specification: str
    current_artifact: str | None = None
    feedback_history: tuple[tuple[str, str], ...] = ()
    dependency_artifacts: tuple[
        tuple[str, str], ...
    ] = ()  # (action_pair_id, artifact_id) - matches schema
    workflow_id: str | None = None  # Extension 03: Agent workflow identifier (Def 19)

    def get_dependency(self, action_pair_id: str) -> str | None:
        """Look up artifact_id by action_pair_id."""
        for key, artifact_id in self.dependency_artifacts:
            if key == action_pair_id:
                return artifact_id
        return None

    def amend(self, delta_spec: str = "", delta_constraints: str = "") -> "Context":
        """Monotonic configuration amendment (⊕ operator, Definition 12).

        Creates a new Context with appended specification and/or constraints.
        Original Context remains unchanged (immutability preserved).

        Args:
            delta_spec: Additional specification to append to current specification.
            delta_constraints: Additional constraints to append to ambient constraints.

        Returns:
            New Context with amended specification and constraints.
        """
        new_spec = self.specification
        if delta_spec:
            new_spec = f"{self.specification}\n{delta_spec}"

        new_constraints = self.ambient.constraints
        if delta_constraints:
            new_constraints = f"{self.ambient.constraints}\n{delta_constraints}"

        new_ambient = AmbientEnvironment(
            repository=self.ambient.repository,
            constraints=new_constraints,
        )

        return Context(
            ambient=new_ambient,
            specification=new_spec,
            current_artifact=self.current_artifact,
            feedback_history=self.feedback_history,
            dependency_artifacts=self.dependency_artifacts,
            workflow_id=self.workflow_id,
        )


# =============================================================================
# WORKFLOW STATE
# =============================================================================


class WorkflowStatus(Enum):
    """Workflow execution outcome."""

    SUCCESS = "success"  # All steps completed
    FAILED = "failed"  # Rmax exhausted on a step
    ESCALATION = "escalation"  # Fatal guard triggered
    CHECKPOINT = "checkpoint"  # Workflow paused, checkpoint created for resume


@dataclass
class WorkflowState:
    """Mutable workflow state tracking guard satisfaction."""

    guards: dict[str, bool] = field(default_factory=dict)
    artifact_ids: dict[str, str] = field(default_factory=dict)

    def is_satisfied(self, guard_id: str) -> bool:
        return self.guards.get(guard_id, False)

    def satisfy(self, guard_id: str, artifact_id: str) -> None:
        self.guards[guard_id] = True
        self.artifact_ids[guard_id] = artifact_id

    def unsatisfy(self, guard_id: str) -> None:
        """Mark a guard as unsatisfied (Extension 09: Cascade Invalidation)."""
        self.guards[guard_id] = False
        self.artifact_ids.pop(guard_id, None)

    def get_artifact_id(self, guard_id: str) -> str | None:
        return self.artifact_ids.get(guard_id)

    def get_satisfied_guards(self) -> tuple[str, ...]:
        """Return tuple of all currently satisfied guard IDs."""
        return tuple(gid for gid, satisfied in self.guards.items() if satisfied)


@dataclass(frozen=True)
class WorkflowResult:
    """Result of workflow execution."""

    status: WorkflowStatus
    artifacts: dict[str, Artifact]
    failed_step: str | None = None
    provenance: tuple[tuple[Artifact, str], ...] = ()
    escalation_artifact: Artifact | None = None  # Artifact that triggered escalation
    escalation_feedback: str = ""  # Fatal feedback message
    checkpoint: "WorkflowCheckpoint | None" = None  # For CHECKPOINT status


# =============================================================================
# CHECKPOINT AND HUMAN AMENDMENT (Resumable Workflow Support)
# =============================================================================


class FailureType(Enum):
    """Type of workflow failure that triggered checkpoint."""

    ESCALATION = "escalation"  # Guard returned ⊥_fatal
    RMAX_EXHAUSTED = "rmax_exhausted"  # Retry budget exhausted


@dataclass(frozen=True)
class WorkflowCheckpoint:
    """
    Immutable checkpoint capturing workflow state at failure.

    Enables resumption after human amendment by preserving:
    - Original workflow context and configuration
    - Completed steps and their artifacts
    - Failure details for human review
    """

    # Identity
    checkpoint_id: str  # UUID
    workflow_id: str  # Original workflow execution ID
    created_at: str  # ISO timestamp

    # Workflow Context
    specification: str  # Original Ψ
    constraints: str  # Original Ω
    rmax: int  # Original retry budget

    # Completed State
    completed_steps: tuple[str, ...]  # guard_ids that passed
    artifact_ids: tuple[tuple[str, str], ...]  # (guard_id, artifact_id) pairs

    # Failure Details
    failure_type: FailureType
    failed_step: str  # guard_id where failure occurred
    failed_artifact_id: str | None  # Last artifact before failure
    failure_feedback: str  # Error/feedback message
    provenance_ids: tuple[str, ...]  # Artifact IDs of all failed attempts

    # Extension 01: Versioned Environment (Definition 11)
    workflow_ref: str | None = None  # W_ref: Content-addressed workflow hash


class AmendmentType(Enum):
    """Type of human amendment."""

    ARTIFACT = "artifact"  # Human provides new artifact content
    FEEDBACK = "feedback"  # Human provides additional guidance for LLM retry
    SKIP = "skip"  # Human approves skipping this step (for optional steps)


@dataclass(frozen=True)
class HumanAmendment:
    """
    Immutable record of human intervention in a workflow.

    Creates a link in the DAG provenance chain from the failed artifact
    to the human-provided amendment.
    """

    # Identity
    amendment_id: str  # UUID
    checkpoint_id: str  # Links to WorkflowCheckpoint
    amendment_type: AmendmentType
    created_at: str  # ISO timestamp
    created_by: str  # Human identifier (e.g., username, "cli")

    # Content
    content: str  # Human-provided artifact or feedback
    context: str = ""  # Additional context/clarification

    # Provenance
    parent_artifact_id: str | None = None  # Links to failed artifact in DAG

    # Resume Options
    additional_rmax: int = 0  # Extra retries beyond original budget
