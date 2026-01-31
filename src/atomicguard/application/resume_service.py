"""Application service for workflow resume operations.

Provides clean separation of resume logic from workflow execution,
with Extension 01 W_ref support for workflow integrity verification.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from atomicguard.domain.models import (
    AmendmentType,
    Artifact,
    ArtifactStatus,
    GuardResult,
    HumanAmendment,
)
from atomicguard.domain.workflow import (
    HumanAmendmentProcessor,
    WorkflowResumer,
    compute_workflow_ref,
)

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import (
        ArtifactDAGInterface,
        CheckpointDAGInterface,
        GuardInterface,
    )


class ResumeResult:
    """Result of resume operation.

    Attributes:
        success: Whether the resume completed without errors.
        artifact: The artifact created from the amendment (if any).
        guard_result: The guard validation result (if validation occurred).
        needs_retry: Whether the workflow needs a retry (feedback amendment or guard failure).
        error: Error message if success is False.
        amended_context: The amended context for retry (feedback amendment only).
    """

    def __init__(
        self,
        success: bool,
        artifact: Artifact | None = None,
        guard_result: GuardResult | None = None,
        needs_retry: bool = False,
        error: str | None = None,
        amended_context: object | None = None,
    ) -> None:
        self.success = success
        self.artifact = artifact
        self.guard_result = guard_result
        self.needs_retry = needs_retry
        self.error = error
        self.amended_context = amended_context


class WorkflowResumeService:
    """Application service for resuming workflows from checkpoints.

    Delegates to domain layer components (WorkflowResumer, HumanAmendmentProcessor)
    while providing a clean interface for the presentation layer.

    Extension 01 compliant: Verifies W_ref integrity on resume.
    """

    def __init__(
        self,
        checkpoint_dag: CheckpointDAGInterface,
        artifact_dag: ArtifactDAGInterface,
    ) -> None:
        """Initialize resume service.

        Args:
            checkpoint_dag: DAG for storing checkpoints and amendments.
            artifact_dag: DAG for storing artifacts.
        """
        self._checkpoint_dag = checkpoint_dag
        self._artifact_dag = artifact_dag

        # Initialize domain layer components
        self._resumer = WorkflowResumer(checkpoint_dag, artifact_dag)
        self._processor = HumanAmendmentProcessor(checkpoint_dag, artifact_dag)

    def resume(
        self,
        checkpoint_id: str,
        amendment: HumanAmendment,
        current_workflow_definition: dict,
        guard: GuardInterface,
        dependencies: dict[str, Artifact] | None = None,
    ) -> ResumeResult:
        """Resume workflow from checkpoint with W_ref verification.

        Args:
            checkpoint_id: ID of checkpoint to resume from.
            amendment: Human-provided amendment.
            current_workflow_definition: Current workflow dict for W_ref verification.
            guard: Guard to validate the human artifact.
            dependencies: Artifact dependencies for guard validation.

        Returns:
            ResumeResult with success status and artifact.

        Raises:
            WorkflowIntegrityError: If workflow changed since checkpoint.
        """
        dependencies = dependencies or {}

        # 1. Store amendment (required by HumanAmendmentProcessor)
        self._checkpoint_dag.store_amendment(amendment)

        # 2. Verify W_ref integrity using domain layer
        current_w_ref = compute_workflow_ref(current_workflow_definition, store=False)
        resume_result = self._resumer.resume(checkpoint_id, current_w_ref)

        if not resume_result.success:
            return ResumeResult(success=False, error=resume_result.error)

        # 3. Process amendment based on type
        if amendment.amendment_type == AmendmentType.ARTIFACT:
            return self._handle_artifact_amendment(amendment, guard, dependencies)

        elif amendment.amendment_type == AmendmentType.FEEDBACK:
            # Return amended context for agent retry
            amended_context = self._processor.apply_amendment_to_context(
                amendment.amendment_id
            )
            return ResumeResult(
                success=True,
                needs_retry=True,
                amended_context=amended_context,
            )

        return ResumeResult(
            success=False, error=f"Unknown amendment type: {amendment.amendment_type}"
        )

    def _handle_artifact_amendment(
        self,
        amendment: HumanAmendment,
        guard: GuardInterface,
        dependencies: dict[str, Artifact],
    ) -> ResumeResult:
        """Process artifact amendment using domain layer.

        Args:
            amendment: The human amendment with artifact content.
            guard: Guard to validate the artifact.
            dependencies: Artifact dependencies for guard validation.

        Returns:
            ResumeResult with artifact and guard result.
        """
        try:
            # Create artifact from amendment using domain layer
            human_artifact = self._processor.create_artifact_from_amendment(
                amendment.amendment_id
            )
        except KeyError as e:
            return ResumeResult(
                success=False,
                error=f"Failed to create artifact from amendment: {e}",
            )
        except ValueError as e:
            return ResumeResult(
                success=False,
                error=f"Invalid amendment: {e}",
            )

        # Validate with guard
        guard_result = guard.validate(human_artifact, **dependencies)

        if guard_result.passed:
            # Update artifact status
            accepted_artifact = replace(
                human_artifact,
                status=ArtifactStatus.ACCEPTED,
                guard_result=guard_result,  # Store full GuardResult
            )
            self._artifact_dag.store(accepted_artifact)

            return ResumeResult(
                success=True,
                artifact=accepted_artifact,
                guard_result=guard_result,
            )
        else:
            # Guard failed - return for new checkpoint
            return ResumeResult(
                success=True,
                artifact=human_artifact,
                guard_result=guard_result,
                needs_retry=True,
            )

    def get_restored_state(self, checkpoint_id: str):
        """Get restored workflow state from checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore from.

        Returns:
            WorkflowState with completed steps and artifact IDs.
        """
        return self._resumer.restore_state(checkpoint_id)

    def get_reconstructed_context(self, checkpoint_id: str):
        """Get reconstructed context from checkpoint.

        Args:
            checkpoint_id: ID of checkpoint.

        Returns:
            Reconstructed Context.
        """
        return self._resumer.reconstruct_context(checkpoint_id)

    def verify_workflow_integrity(
        self, checkpoint_id: str, current_workflow_definition: dict
    ) -> bool:
        """Verify workflow integrity without resuming.

        Args:
            checkpoint_id: ID of checkpoint to verify against.
            current_workflow_definition: Current workflow dict.

        Returns:
            True if workflow matches checkpoint, False otherwise.

        Raises:
            WorkflowIntegrityError: If workflow changed since checkpoint.
        """
        current_w_ref = compute_workflow_ref(current_workflow_definition, store=False)
        result = self._resumer.resume(checkpoint_id, current_w_ref)
        return result.success
