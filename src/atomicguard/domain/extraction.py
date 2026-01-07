"""
Artifact Extraction (Extension 02: Definitions 17-18).

Implements:
- Predicate base class (Φ: r → {⊤, ⊥})
- Concrete predicates: StatusPredicate, ActionPairPredicate, WorkflowPredicate, SourcePredicate
- Compound predicates: AndPredicate, OrPredicate, NotPredicate
- Extraction function: E: ℛ × Φ → 2^ℛ
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from atomicguard.domain.models import Artifact, ArtifactSource, ArtifactStatus

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import ArtifactDAGInterface


# =============================================================================
# PREDICATE BASE CLASS (Definition 18)
# =============================================================================


class Predicate(ABC):
    """
    Abstract base class for filter predicates Φ: r → {⊤, ⊥} (Definition 18).

    Predicates are boolean functions over repository items that determine
    whether an artifact should be included in an extraction result.
    """

    @abstractmethod
    def matches(self, artifact: Artifact) -> bool:
        """Evaluate predicate on artifact.

        Args:
            artifact: The artifact to test.

        Returns:
            True if artifact matches predicate, False otherwise.
        """
        pass

    def __call__(self, artifact: Artifact) -> bool:
        """Allow predicate(artifact) syntax.

        Enables predicates to be used as Φ(r) per the formal notation.
        """
        return self.matches(artifact)


# =============================================================================
# CONCRETE PREDICATES
# =============================================================================


class StatusPredicate(Predicate):
    """Filter by artifact status (Φ_status).

    Matches artifacts whose status is in the specified set of statuses.
    """

    def __init__(self, *statuses: ArtifactStatus) -> None:
        """Initialize with one or more statuses to match.

        Args:
            *statuses: ArtifactStatus values to match.
        """
        self._statuses = set(statuses)

    def matches(self, artifact: Artifact) -> bool:
        """Check if artifact status is in the specified set."""
        return artifact.status in self._statuses


class ActionPairPredicate(Predicate):
    """Filter by action pair ID (Φ_action_pair).

    Matches artifacts produced by a specific action pair.
    """

    def __init__(self, action_pair_id: str) -> None:
        """Initialize with action pair ID to match.

        Args:
            action_pair_id: The action_pair_id to match.
        """
        self._action_pair_id = action_pair_id

    def matches(self, artifact: Artifact) -> bool:
        """Check if artifact's action_pair_id matches."""
        return artifact.action_pair_id == self._action_pair_id


class WorkflowPredicate(Predicate):
    """Filter by workflow ID (Φ_workflow).

    Matches artifacts from a specific workflow execution.
    """

    def __init__(self, workflow_id: str) -> None:
        """Initialize with workflow ID to match.

        Args:
            workflow_id: The workflow_id to match.
        """
        self._workflow_id = workflow_id

    def matches(self, artifact: Artifact) -> bool:
        """Check if artifact's workflow_id matches."""
        return artifact.workflow_id == self._workflow_id


class SourcePredicate(Predicate):
    """Filter by artifact source (Φ_source).

    Matches artifacts with a specific source (GENERATED, HUMAN, IMPORTED).
    """

    def __init__(self, source: ArtifactSource) -> None:
        """Initialize with source to match.

        Args:
            source: The ArtifactSource to match.
        """
        self._source = source

    def matches(self, artifact: Artifact) -> bool:
        """Check if artifact's source matches."""
        return artifact.source == self._source


# =============================================================================
# COMPOUND PREDICATES
# =============================================================================


class AndPredicate(Predicate):
    """Logical AND of two predicates (Φ₁ ∧ Φ₂).

    Matches only if both predicates match.
    """

    def __init__(self, p1: Predicate, p2: Predicate) -> None:
        """Initialize with two predicates to AND together.

        Args:
            p1: First predicate.
            p2: Second predicate.
        """
        self._p1 = p1
        self._p2 = p2

    def matches(self, artifact: Artifact) -> bool:
        """Check if both predicates match."""
        return self._p1.matches(artifact) and self._p2.matches(artifact)


class OrPredicate(Predicate):
    """Logical OR of two predicates (Φ₁ ∨ Φ₂).

    Matches if either predicate matches.
    """

    def __init__(self, p1: Predicate, p2: Predicate) -> None:
        """Initialize with two predicates to OR together.

        Args:
            p1: First predicate.
            p2: Second predicate.
        """
        self._p1 = p1
        self._p2 = p2

    def matches(self, artifact: Artifact) -> bool:
        """Check if either predicate matches."""
        return self._p1.matches(artifact) or self._p2.matches(artifact)


class NotPredicate(Predicate):
    """Logical NOT of a predicate (¬Φ).

    Inverts the result of the wrapped predicate.
    """

    def __init__(self, predicate: Predicate) -> None:
        """Initialize with predicate to invert.

        Args:
            predicate: The predicate to negate.
        """
        self._predicate = predicate

    def matches(self, artifact: Artifact) -> bool:
        """Return inverse of wrapped predicate."""
        return not self._predicate.matches(artifact)


# =============================================================================
# EXTRACTION FUNCTION (Definition 17)
# =============================================================================


def extract(
    dag: ArtifactDAGInterface,
    predicate: Predicate | None = None,
    limit: int | None = None,
    offset: int | None = None,
    order_by: str | None = None,
) -> list[Artifact]:
    """Extract artifacts from repository matching predicate (Definition 17).

    E: ℛ × Φ → 2^ℛ

    This is a read-only operation that does not modify the repository.
    Implements Theorem 3: Extraction Invariance - extraction is idempotent
    and referentially transparent.

    Args:
        dag: The artifact repository to extract from.
        predicate: Filter predicate Φ. If None, matches all artifacts.
        limit: Maximum number of artifacts to return.
        offset: Number of artifacts to skip before returning.
        order_by: Field name to sort by. Prefix with '-' for descending order.

    Returns:
        List of artifacts matching the predicate, with pagination applied.
    """
    # Get all artifacts from DAG via interface method
    all_artifacts = dag.get_all()

    # Apply predicate filter
    if predicate is not None:
        all_artifacts = [a for a in all_artifacts if predicate.matches(a)]

    # Apply ordering
    if order_by is not None:
        descending = order_by.startswith("-")
        field_name = order_by.lstrip("-")

        def get_sort_key(artifact: Artifact) -> str:
            return getattr(artifact, field_name, "")

        all_artifacts.sort(key=get_sort_key, reverse=descending)

    # Apply offset
    if offset is not None and offset > 0:
        all_artifacts = all_artifacts[offset:]

    # Apply limit
    if limit is not None and limit > 0:
        all_artifacts = all_artifacts[:limit]

    return all_artifacts
