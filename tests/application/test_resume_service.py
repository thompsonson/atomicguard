"""Tests for WorkflowResumeService - resume with W_ref verification."""

import uuid
from datetime import UTC, datetime

import pytest

from atomicguard.application.resume_service import ResumeResult, WorkflowResumeService
from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import (
    AmendmentType,
    Artifact,
    ArtifactSource,
    ArtifactStatus,
    ContextSnapshot,
    FailureType,
    GuardResult,
    HumanAmendment,
    WorkflowCheckpoint,
)
from atomicguard.domain.workflow import WorkflowIntegrityError, compute_workflow_ref
from atomicguard.infrastructure.persistence.checkpoint import InMemoryCheckpointDAG
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


class AlwaysPassGuard(GuardInterface):
    """Guard that always passes."""

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=True, feedback="Passed")


class AlwaysFailGuard(GuardInterface):
    """Guard that always fails."""

    def __init__(self, feedback: str = "Always fails"):
        self._feedback = feedback

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=False, feedback=self._feedback)


class TestWorkflowResumeServiceInit:
    """Tests for WorkflowResumeService initialization."""

    def test_init_with_dags(self) -> None:
        """WorkflowResumeService initializes with DAGs."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = WorkflowResumeService(checkpoint_dag, artifact_dag)

        assert service._checkpoint_dag is checkpoint_dag
        assert service._artifact_dag is artifact_dag
        assert service._resumer is not None
        assert service._processor is not None


def create_test_checkpoint(
    checkpoint_dag: InMemoryCheckpointDAG,
    workflow_definition: dict,
    workflow_id: str = "test-workflow",
    failed_step: str = "g_test",
) -> WorkflowCheckpoint:
    """Helper to create and store a test checkpoint."""
    w_ref = compute_workflow_ref(workflow_definition, store=True)

    checkpoint = WorkflowCheckpoint(
        checkpoint_id=str(uuid.uuid4()),
        workflow_id=workflow_id,
        created_at=datetime.now(UTC).isoformat(),
        specification="Test specification",
        constraints="Test constraints",
        rmax=3,
        completed_steps=(),
        artifact_ids=(),
        failure_type=FailureType.RMAX_EXHAUSTED,
        failed_step=failed_step,
        failed_artifact_id=None,
        failure_feedback="Test failed",
        provenance_ids=(),
        workflow_ref=w_ref,
    )
    checkpoint_dag.store_checkpoint(checkpoint)
    return checkpoint


def create_test_amendment(
    checkpoint_id: str,
    content: str = "def fixed(): pass",
) -> HumanAmendment:
    """Helper to create a test amendment."""
    return HumanAmendment(
        amendment_id=str(uuid.uuid4()),
        checkpoint_id=checkpoint_id,
        amendment_type=AmendmentType.ARTIFACT,
        created_at=datetime.now(UTC).isoformat(),
        created_by="test",
        content=content,
        context="Test context",
        parent_artifact_id=None,
        additional_rmax=0,
    )


class TestWorkflowResumeServiceResume:
    """Tests for resume method."""

    def test_resume_verifies_w_ref(self) -> None:
        """resume() verifies W_ref matches before proceeding."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = WorkflowResumeService(checkpoint_dag, artifact_dag)

        workflow_definition = {
            "steps": [{"guard_id": "g_test", "requires": [], "deps": []}],
            "rmax": 3,
            "constraints": "",
        }

        checkpoint = create_test_checkpoint(checkpoint_dag, workflow_definition)
        amendment = create_test_amendment(checkpoint.checkpoint_id)

        # Resume with matching workflow - should succeed
        result = service.resume(
            checkpoint_id=checkpoint.checkpoint_id,
            amendment=amendment,
            current_workflow_definition=workflow_definition,
            guard=AlwaysPassGuard(),
        )

        assert result.success is True

    def test_resume_raises_on_w_ref_mismatch(self) -> None:
        """resume() raises WorkflowIntegrityError on W_ref mismatch."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = WorkflowResumeService(checkpoint_dag, artifact_dag)

        original_workflow = {
            "steps": [{"guard_id": "g_test", "requires": [], "deps": []}],
            "rmax": 3,
            "constraints": "",
        }

        checkpoint = create_test_checkpoint(checkpoint_dag, original_workflow)
        amendment = create_test_amendment(checkpoint.checkpoint_id)

        # Modified workflow (different rmax)
        modified_workflow = {
            "steps": [{"guard_id": "g_test", "requires": [], "deps": []}],
            "rmax": 5,  # Changed!
            "constraints": "",
        }

        # Resume with modified workflow - should raise
        with pytest.raises(WorkflowIntegrityError) as exc_info:
            service.resume(
                checkpoint_id=checkpoint.checkpoint_id,
                amendment=amendment,
                current_workflow_definition=modified_workflow,
                guard=AlwaysPassGuard(),
            )

        assert "Workflow changed since checkpoint" in str(exc_info.value)

    def test_resume_artifact_amendment_passes_guard(self) -> None:
        """resume() with artifact amendment that passes guard."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = WorkflowResumeService(checkpoint_dag, artifact_dag)

        workflow_definition = {
            "steps": [{"guard_id": "g_test", "requires": [], "deps": []}],
            "rmax": 3,
            "constraints": "",
        }

        checkpoint = create_test_checkpoint(checkpoint_dag, workflow_definition)
        amendment = create_test_amendment(
            checkpoint.checkpoint_id, "def correct(): return True"
        )

        result = service.resume(
            checkpoint_id=checkpoint.checkpoint_id,
            amendment=amendment,
            current_workflow_definition=workflow_definition,
            guard=AlwaysPassGuard(),
        )

        assert result.success is True
        assert result.artifact is not None
        assert result.artifact.status == ArtifactStatus.ACCEPTED
        assert result.guard_result is not None
        assert result.guard_result.passed is True
        assert result.needs_retry is False

    def test_resume_artifact_amendment_fails_guard(self) -> None:
        """resume() with artifact amendment that fails guard."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = WorkflowResumeService(checkpoint_dag, artifact_dag)

        workflow_definition = {
            "steps": [{"guard_id": "g_test", "requires": [], "deps": []}],
            "rmax": 3,
            "constraints": "",
        }

        checkpoint = create_test_checkpoint(checkpoint_dag, workflow_definition)
        amendment = create_test_amendment(checkpoint.checkpoint_id, "def wrong(): pass")

        result = service.resume(
            checkpoint_id=checkpoint.checkpoint_id,
            amendment=amendment,
            current_workflow_definition=workflow_definition,
            guard=AlwaysFailGuard("Still wrong"),
        )

        assert result.success is True  # Resume operation succeeded
        assert result.artifact is not None
        assert result.guard_result is not None
        assert result.guard_result.passed is False
        assert result.needs_retry is True  # Needs another attempt

    def test_resume_feedback_amendment(self) -> None:
        """resume() with feedback amendment returns needs_retry=True."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = WorkflowResumeService(checkpoint_dag, artifact_dag)

        workflow_definition = {
            "steps": [{"guard_id": "g_test", "requires": [], "deps": []}],
            "rmax": 3,
            "constraints": "",
        }

        checkpoint = create_test_checkpoint(checkpoint_dag, workflow_definition)

        # Feedback amendment instead of artifact
        amendment = HumanAmendment(
            amendment_id=str(uuid.uuid4()),
            checkpoint_id=checkpoint.checkpoint_id,
            amendment_type=AmendmentType.FEEDBACK,
            created_at=datetime.now(UTC).isoformat(),
            created_by="test",
            content="Try using a different algorithm",
            context="Guidance for retry",
            parent_artifact_id=None,
            additional_rmax=2,
        )

        result = service.resume(
            checkpoint_id=checkpoint.checkpoint_id,
            amendment=amendment,
            current_workflow_definition=workflow_definition,
            guard=AlwaysPassGuard(),
        )

        assert result.success is True
        assert result.needs_retry is True
        assert result.amended_context is not None

    def test_resume_stores_amendment(self) -> None:
        """resume() stores the amendment in checkpoint DAG."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = WorkflowResumeService(checkpoint_dag, artifact_dag)

        workflow_definition = {
            "steps": [{"guard_id": "g_test", "requires": [], "deps": []}],
            "rmax": 3,
            "constraints": "",
        }

        checkpoint = create_test_checkpoint(checkpoint_dag, workflow_definition)
        amendment = create_test_amendment(checkpoint.checkpoint_id)

        service.resume(
            checkpoint_id=checkpoint.checkpoint_id,
            amendment=amendment,
            current_workflow_definition=workflow_definition,
            guard=AlwaysPassGuard(),
        )

        # Amendment should be stored
        stored_amendment = checkpoint_dag.get_amendment(amendment.amendment_id)
        assert stored_amendment is not None
        assert stored_amendment.amendment_id == amendment.amendment_id


class TestWorkflowResumeServiceHelpers:
    """Tests for helper methods."""

    def test_get_restored_state(self) -> None:
        """get_restored_state returns workflow state from checkpoint."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = WorkflowResumeService(checkpoint_dag, artifact_dag)

        # Create checkpoint with completed steps
        w_ref = compute_workflow_ref(
            {"steps": [], "rmax": 1, "constraints": ""}, store=True
        )
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            workflow_id="wf-test",
            created_at=datetime.now(UTC).isoformat(),
            specification="spec",
            constraints="",
            rmax=1,
            completed_steps=("step1", "step2"),
            artifact_ids=(("step1", "a1"), ("step2", "a2")),
            failure_type=FailureType.RMAX_EXHAUSTED,
            failed_step="step3",
            failed_artifact_id=None,
            failure_feedback="error",
            provenance_ids=(),
            workflow_ref=w_ref,
        )
        checkpoint_dag.store_checkpoint(checkpoint)

        state = service.get_restored_state(checkpoint.checkpoint_id)

        assert state.completed_steps == ("step1", "step2")
        assert state.artifact_ids == {"step1": "a1", "step2": "a2"}
        assert state.next_step == "step3"

    def test_verify_workflow_integrity_matching(self) -> None:
        """verify_workflow_integrity returns True for matching workflow."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = WorkflowResumeService(checkpoint_dag, artifact_dag)

        workflow_definition = {"steps": [], "rmax": 1, "constraints": ""}
        checkpoint = create_test_checkpoint(checkpoint_dag, workflow_definition)

        result = service.verify_workflow_integrity(
            checkpoint.checkpoint_id,
            workflow_definition,
        )

        assert result is True

    def test_verify_workflow_integrity_mismatch(self) -> None:
        """verify_workflow_integrity raises on mismatch."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = WorkflowResumeService(checkpoint_dag, artifact_dag)

        original = {"steps": [], "rmax": 1, "constraints": ""}
        modified = {"steps": [], "rmax": 2, "constraints": ""}

        checkpoint = create_test_checkpoint(checkpoint_dag, original)

        with pytest.raises(WorkflowIntegrityError):
            service.verify_workflow_integrity(checkpoint.checkpoint_id, modified)


class TestResumeResult:
    """Tests for ResumeResult dataclass."""

    def test_resume_result_defaults(self) -> None:
        """ResumeResult has correct defaults."""
        result = ResumeResult(success=True)

        assert result.success is True
        assert result.artifact is None
        assert result.guard_result is None
        assert result.needs_retry is False
        assert result.error is None
        assert result.amended_context is None

    def test_resume_result_with_all_fields(self) -> None:
        """ResumeResult can be created with all fields."""
        artifact = Artifact(
            artifact_id="test-artifact",
            workflow_id="test-workflow",
            content="content",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_test",
            created_at=datetime.now(UTC).isoformat(),
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=True,
            feedback="",
            context=ContextSnapshot(
                workflow_id="test-workflow",
                specification="spec",
                constraints="",
                feedback_history=(),
            ),
            source=ArtifactSource.HUMAN,
        )
        guard_result = GuardResult(passed=True, feedback="OK")

        result = ResumeResult(
            success=True,
            artifact=artifact,
            guard_result=guard_result,
            needs_retry=False,
            error=None,
            amended_context={"key": "value"},
        )

        assert result.success is True
        assert result.artifact is artifact
        assert result.guard_result is guard_result
        assert result.needs_retry is False
        assert result.amended_context == {"key": "value"}
