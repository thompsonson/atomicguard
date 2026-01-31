"""Application service for checkpoint operations.

Provides clean separation of checkpoint creation from workflow execution,
with Extension 01 W_ref support for workflow integrity verification.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from atomicguard.domain.models import FailureType, WorkflowCheckpoint
from atomicguard.domain.workflow import compute_workflow_ref

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import (
        ArtifactDAGInterface,
        CheckpointDAGInterface,
    )


class CheckpointService:
    """Application service for creating and managing checkpoints.

    Separates checkpoint creation concerns from Workflow execution,
    enabling cleaner DDD architecture with proper layering.
    """

    def __init__(
        self,
        checkpoint_dag: CheckpointDAGInterface,
        artifact_dag: ArtifactDAGInterface | None = None,
    ) -> None:
        """Initialize checkpoint service.

        Args:
            checkpoint_dag: DAG for storing checkpoints.
            artifact_dag: DAG for storing artifacts (optional).
        """
        self._checkpoint_dag = checkpoint_dag
        self._artifact_dag = artifact_dag

    def create_checkpoint(
        self,
        workflow_definition: dict,
        workflow_id: str,
        specification: str,
        constraints: str,
        rmax: int,
        completed_steps: tuple[str, ...],
        artifact_ids: tuple[tuple[str, str], ...],
        failure_type: FailureType,
        failed_step: str,
        failed_artifact_id: str | None,
        failure_feedback: str,
        provenance_ids: tuple[str, ...],
    ) -> WorkflowCheckpoint:
        """Create checkpoint with W_ref for integrity verification.

        Computes the W_ref (content-addressed workflow hash) from the
        workflow definition and stores it with the checkpoint. This
        enables resume-time verification that the workflow hasn't changed.

        Args:
            workflow_definition: Dict representing workflow structure for W_ref computation.
            workflow_id: UUID of the workflow execution instance.
            specification: Original task specification (Ψ).
            constraints: Global constraints (Ω).
            rmax: Original retry budget.
            completed_steps: Guard IDs that completed successfully.
            artifact_ids: (guard_id, artifact_id) pairs for completed artifacts.
            failure_type: Type of failure (ESCALATION or RMAX_EXHAUSTED).
            failed_step: Guard ID where failure occurred.
            failed_artifact_id: ID of the last artifact before failure.
            failure_feedback: Error/feedback message from the guard.
            provenance_ids: Artifact IDs of all failed attempts.

        Returns:
            WorkflowCheckpoint with W_ref for integrity verification.
        """
        # Compute W_ref using domain layer (stores in registry for later resolution)
        w_ref = compute_workflow_ref(workflow_definition, store=True)

        checkpoint = WorkflowCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            created_at=datetime.now(UTC).isoformat(),
            specification=specification,
            constraints=constraints,
            rmax=rmax,
            completed_steps=completed_steps,
            artifact_ids=artifact_ids,
            failure_type=failure_type,
            failed_step=failed_step,
            failed_artifact_id=failed_artifact_id,
            failure_feedback=failure_feedback,
            provenance_ids=provenance_ids,
            workflow_ref=w_ref,  # Extension 01: W_ref for integrity
        )

        self._checkpoint_dag.store_checkpoint(checkpoint)
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint:
        """Retrieve a checkpoint by ID.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            The WorkflowCheckpoint.

        Raises:
            KeyError: If checkpoint not found.
        """
        return self._checkpoint_dag.get_checkpoint(checkpoint_id)

    def list_checkpoints(
        self, workflow_id: str | None = None
    ) -> list[WorkflowCheckpoint]:
        """List checkpoints, optionally filtered by workflow_id.

        Args:
            workflow_id: Optional filter by workflow.

        Returns:
            List of matching checkpoints, newest first.
        """
        return self._checkpoint_dag.list_checkpoints(workflow_id)
