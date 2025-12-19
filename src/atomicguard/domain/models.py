"""
Domain models for the Dual-State Framework.

These are pure data structures aligned with paper Definitions 4-6.
All models are immutable (frozen dataclasses) to ensure referential transparency.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import ArtifactDAGInterface


# =============================================================================
# ARTIFACT MODEL (Definition 4-6)
# =============================================================================


class ArtifactStatus(Enum):
    """Status of an artifact in the DAG."""

    PENDING = "pending"  # Generated, not yet validated
    REJECTED = "rejected"  # Guard returned ⊥
    ACCEPTED = "accepted"  # Guard returned ⊤, final for this step
    SUPERSEDED = "superseded"  # Guard returned ⊤, but later attempt also passed


@dataclass(frozen=True)
class FeedbackEntry:
    """Single entry in feedback history H."""

    artifact_id: str  # Reference to the rejected artifact
    feedback: str  # Guard's rejection message φ


@dataclass(frozen=True)
class ContextSnapshot:
    """Immutable context C that conditioned generation (Definition 5)."""

    specification: str  # Ψ - static specification
    constraints: str  # Ω - global constraints
    feedback_history: tuple[FeedbackEntry, ...]  # H - accumulated rejections
    dependency_ids: tuple[str, ...]  # Artifact IDs from prior workflow steps


@dataclass(frozen=True)
class Artifact:
    """
    Immutable node in the Versioned Repository DAG (Definition 4).

    Represents a single generation attempt with full provenance tracking.
    """

    # Identity
    artifact_id: str  # Unique identifier (UUID)
    content: str  # The generated code/text

    # DAG Structure
    previous_attempt_id: str | None  # Retry chain within same action pair
    # Cross-step deps are in context.dependency_ids

    # Action Pair Coupling (Definition 6: A = ⟨ρ, a_gen, G⟩)
    action_pair_id: str  # Which action pair produced this

    # Metadata
    created_at: str  # ISO timestamp
    attempt_number: int  # Attempt within this action pair context
    status: ArtifactStatus  # pending/rejected/accepted/superseded
    guard_result: bool | None  # ⊤ or ⊥ (None if pending)
    feedback: str  # φ - guard feedback (empty if passed)
    context: ContextSnapshot  # Full context snapshot at generation time


# =============================================================================
# GUARD RESULT
# =============================================================================


@dataclass(frozen=True)
class GuardResult:
    """Immutable guard validation outcome."""

    passed: bool
    feedback: str = ""
    fatal: bool = False  # ⊥_fatal - skip retry, escalate to human


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
    dependencies: tuple[
        tuple[str, "Artifact"], ...
    ] = ()  # (key, artifact) pairs from prior steps


# =============================================================================
# WORKFLOW STATE
# =============================================================================


class WorkflowStatus(Enum):
    """Workflow execution outcome."""

    SUCCESS = "success"  # All steps completed
    FAILED = "failed"  # Rmax exhausted on a step
    ESCALATION = "escalation"  # Fatal guard triggered


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

    def get_artifact_id(self, guard_id: str) -> str | None:
        return self.artifact_ids.get(guard_id)


@dataclass(frozen=True)
class WorkflowResult:
    """Result of workflow execution."""

    status: WorkflowStatus
    artifacts: dict[str, Artifact]
    failed_step: str | None = None
    provenance: tuple[tuple[Artifact, str], ...] = ()
    escalation_artifact: Artifact | None = None  # Artifact that triggered escalation
    escalation_feedback: str = ""  # Fatal feedback message
