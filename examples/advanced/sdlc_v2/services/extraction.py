"""
Artifact Extraction Service (Extension 02: Definitions 17-18).

Provides a high-level service interface for querying artifacts
from the repository using predicates and compound filters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atomicguard.domain.extraction import (
    ActionPairPredicate,
    AndPredicate,
    NotPredicate,
    OrPredicate,
    Predicate,
    SourcePredicate,
    StatusPredicate,
    WorkflowPredicate,
    extract,
)
from atomicguard.domain.models import Artifact, ArtifactSource, ArtifactStatus

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import ArtifactDAGInterface


class ArtifactExtractionService:
    """
    Service for querying artifacts from the repository (Extension 02).

    Provides convenient methods for common query patterns while
    exposing the full predicate system for advanced queries.

    The extraction function E: ℛ × Φ → 2^ℛ is a pure, read-only operation
    that preserves system dynamics (Theorem 3: Extraction Invariance).
    """

    def __init__(self, artifact_dag: ArtifactDAGInterface) -> None:
        """Initialize with artifact DAG.

        Args:
            artifact_dag: Repository for artifact storage and retrieval.
        """
        self._artifact_dag = artifact_dag

    def query(
        self,
        predicate: Predicate | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,
    ) -> list[Artifact]:
        """Execute a query with optional predicate and pagination.

        Args:
            predicate: Filter predicate. If None, returns all artifacts.
            limit: Maximum number of results.
            offset: Number of results to skip.
            order_by: Field to sort by (prefix with '-' for descending).

        Returns:
            List of matching artifacts.
        """
        return extract(
            self._artifact_dag,
            predicate=predicate,
            limit=limit,
            offset=offset,
            order_by=order_by,
        )

    def get_all(self) -> list[Artifact]:
        """Get all artifacts in the repository.

        Returns:
            List of all artifacts.
        """
        return extract(self._artifact_dag)

    def get_by_workflow(self, workflow_id: str) -> list[Artifact]:
        """Get all artifacts for a specific workflow.

        Args:
            workflow_id: Workflow ID to filter by.

        Returns:
            List of artifacts from the workflow.
        """
        return extract(
            self._artifact_dag,
            predicate=WorkflowPredicate(workflow_id),
            order_by="created_at",
        )

    def get_by_status(self, status: ArtifactStatus) -> list[Artifact]:
        """Get artifacts with a specific status.

        Args:
            status: Status to filter by.

        Returns:
            List of artifacts with the status.
        """
        return extract(
            self._artifact_dag,
            predicate=StatusPredicate(status),
        )

    def get_by_action_pair(self, action_pair_id: str) -> list[Artifact]:
        """Get artifacts from a specific action pair.

        Args:
            action_pair_id: Action pair ID to filter by.

        Returns:
            List of artifacts from the action pair.
        """
        return extract(
            self._artifact_dag,
            predicate=ActionPairPredicate(action_pair_id),
        )

    def get_by_source(self, source: ArtifactSource) -> list[Artifact]:
        """Get artifacts with a specific source.

        Args:
            source: Source to filter by (GENERATED, HUMAN, IMPORTED).

        Returns:
            List of artifacts with the source.
        """
        return extract(
            self._artifact_dag,
            predicate=SourcePredicate(source),
        )

    def get_accepted(self) -> list[Artifact]:
        """Get all accepted artifacts.

        Returns:
            List of accepted artifacts.
        """
        return self.get_by_status(ArtifactStatus.ACCEPTED)

    def get_rejected(self) -> list[Artifact]:
        """Get all rejected artifacts.

        Returns:
            List of rejected artifacts.
        """
        return self.get_by_status(ArtifactStatus.REJECTED)

    def get_human_artifacts(self) -> list[Artifact]:
        """Get all human-provided artifacts.

        Returns:
            List of artifacts with HUMAN source.
        """
        return self.get_by_source(ArtifactSource.HUMAN)

    def get_generated_artifacts(self) -> list[Artifact]:
        """Get all LLM-generated artifacts.

        Returns:
            List of artifacts with GENERATED source.
        """
        return self.get_by_source(ArtifactSource.GENERATED)

    def get_accepted_by_workflow(self, workflow_id: str) -> list[Artifact]:
        """Get accepted artifacts for a specific workflow.

        Args:
            workflow_id: Workflow ID to filter by.

        Returns:
            List of accepted artifacts from the workflow.
        """
        return extract(
            self._artifact_dag,
            predicate=AndPredicate(
                WorkflowPredicate(workflow_id),
                StatusPredicate(ArtifactStatus.ACCEPTED),
            ),
            order_by="created_at",
        )

    def get_accepted_by_action_pair(
        self,
        action_pair_id: str,
        workflow_id: str | None = None,
    ) -> list[Artifact]:
        """Get accepted artifacts for a specific action pair.

        Args:
            action_pair_id: Action pair ID to filter by.
            workflow_id: Optional workflow ID to further filter.

        Returns:
            List of accepted artifacts from the action pair.
        """
        predicate: Predicate = AndPredicate(
            ActionPairPredicate(action_pair_id),
            StatusPredicate(ArtifactStatus.ACCEPTED),
        )

        if workflow_id:
            predicate = AndPredicate(
                predicate,
                WorkflowPredicate(workflow_id),
            )

        return extract(
            self._artifact_dag,
            predicate=predicate,
            order_by="-created_at",
            limit=1,
        )

    def get_latest_accepted(self, action_pair_id: str) -> Artifact | None:
        """Get the most recent accepted artifact for an action pair.

        Args:
            action_pair_id: Action pair ID to filter by.

        Returns:
            Most recent accepted artifact, or None if none exist.
        """
        results = extract(
            self._artifact_dag,
            predicate=AndPredicate(
                ActionPairPredicate(action_pair_id),
                StatusPredicate(ArtifactStatus.ACCEPTED),
            ),
            order_by="-created_at",
            limit=1,
        )
        return results[0] if results else None

    def count_by_status(self) -> dict[ArtifactStatus, int]:
        """Count artifacts grouped by status.

        Returns:
            Dictionary mapping status to count.
        """
        all_artifacts = self.get_all()
        counts: dict[ArtifactStatus, int] = {}
        for artifact in all_artifacts:
            counts[artifact.status] = counts.get(artifact.status, 0) + 1
        return counts

    def count_by_action_pair(self) -> dict[str, int]:
        """Count artifacts grouped by action pair.

        Returns:
            Dictionary mapping action_pair_id to count.
        """
        all_artifacts = self.get_all()
        counts: dict[str, int] = {}
        for artifact in all_artifacts:
            ap_id = artifact.action_pair_id or "unknown"
            counts[ap_id] = counts.get(ap_id, 0) + 1
        return counts

    # Compound predicate builders for advanced queries

    @staticmethod
    def and_predicates(*predicates: Predicate) -> Predicate:
        """Build an AND predicate from multiple predicates.

        Args:
            *predicates: Predicates to AND together.

        Returns:
            Combined AND predicate.
        """
        if not predicates:
            raise ValueError("At least one predicate required")
        result = predicates[0]
        for p in predicates[1:]:
            result = AndPredicate(result, p)
        return result

    @staticmethod
    def or_predicates(*predicates: Predicate) -> Predicate:
        """Build an OR predicate from multiple predicates.

        Args:
            *predicates: Predicates to OR together.

        Returns:
            Combined OR predicate.
        """
        if not predicates:
            raise ValueError("At least one predicate required")
        result = predicates[0]
        for p in predicates[1:]:
            result = OrPredicate(result, p)
        return result

    @staticmethod
    def not_predicate(predicate: Predicate) -> Predicate:
        """Build a NOT predicate.

        Args:
            predicate: Predicate to negate.

        Returns:
            Negated predicate.
        """
        return NotPredicate(predicate)
