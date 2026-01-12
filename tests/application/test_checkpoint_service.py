"""Tests for CheckpointService - checkpoint creation with W_ref support."""

from atomicguard.application.checkpoint_service import CheckpointService
from atomicguard.domain.models import FailureType
from atomicguard.domain.workflow import compute_workflow_ref
from atomicguard.infrastructure.persistence.checkpoint import InMemoryCheckpointDAG
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


class TestCheckpointServiceInit:
    """Tests for CheckpointService initialization."""

    def test_init_with_checkpoint_dag(self) -> None:
        """CheckpointService initializes with provided DAG."""
        checkpoint_dag = InMemoryCheckpointDAG()
        service = CheckpointService(checkpoint_dag)

        assert service._checkpoint_dag is checkpoint_dag

    def test_init_with_artifact_dag(self) -> None:
        """CheckpointService accepts optional artifact DAG."""
        checkpoint_dag = InMemoryCheckpointDAG()
        artifact_dag = InMemoryArtifactDAG()
        service = CheckpointService(checkpoint_dag, artifact_dag)

        assert service._checkpoint_dag is checkpoint_dag
        assert service._artifact_dag is artifact_dag


class TestCheckpointServiceCreateCheckpoint:
    """Tests for create_checkpoint method."""

    def test_create_checkpoint_stores_w_ref(self) -> None:
        """create_checkpoint computes and stores W_ref from workflow definition."""
        checkpoint_dag = InMemoryCheckpointDAG()
        service = CheckpointService(checkpoint_dag)

        workflow_definition = {
            "steps": [
                {"guard_id": "g_test", "requires": [], "deps": []},
                {"guard_id": "g_impl", "requires": ["g_test"], "deps": ["g_test"]},
            ],
            "rmax": 3,
            "constraints": "Test constraints",
        }

        checkpoint = service.create_checkpoint(
            workflow_definition=workflow_definition,
            workflow_id="test-workflow-123",
            specification="Test specification",
            constraints="Test constraints",
            rmax=3,
            completed_steps=("g_test",),
            artifact_ids=(("g_test", "artifact-1"),),
            failure_type=FailureType.RMAX_EXHAUSTED,
            failed_step="g_impl",
            failed_artifact_id="artifact-2",
            failure_feedback="Test failed",
            provenance_ids=("artifact-2",),
        )

        # Verify W_ref is computed correctly
        expected_w_ref = compute_workflow_ref(workflow_definition, store=False)
        assert checkpoint.workflow_ref == expected_w_ref
        assert checkpoint.workflow_ref is not None
        assert len(checkpoint.workflow_ref) == 64  # SHA-256 hex digest

    def test_create_checkpoint_persists_to_dag(self) -> None:
        """create_checkpoint stores checkpoint in DAG."""
        checkpoint_dag = InMemoryCheckpointDAG()
        service = CheckpointService(checkpoint_dag)

        workflow_definition = {
            "steps": [{"guard_id": "g_test", "requires": [], "deps": []}],
            "rmax": 2,
            "constraints": "",
        }

        checkpoint = service.create_checkpoint(
            workflow_definition=workflow_definition,
            workflow_id="test-workflow",
            specification="Test spec",
            constraints="",
            rmax=2,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.ESCALATION,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="Fatal error",
            provenance_ids=(),
        )

        # Retrieve from DAG
        retrieved = checkpoint_dag.get_checkpoint(checkpoint.checkpoint_id)
        assert retrieved.checkpoint_id == checkpoint.checkpoint_id
        assert retrieved.workflow_ref == checkpoint.workflow_ref

    def test_create_checkpoint_preserves_all_fields(self) -> None:
        """create_checkpoint preserves all provided fields."""
        checkpoint_dag = InMemoryCheckpointDAG()
        service = CheckpointService(checkpoint_dag)

        workflow_definition = {"steps": [], "rmax": 5, "constraints": "Global"}

        checkpoint = service.create_checkpoint(
            workflow_definition=workflow_definition,
            workflow_id="wf-456",
            specification="Multi-step task",
            constraints="Must be fast",
            rmax=5,
            completed_steps=("step1", "step2"),
            artifact_ids=(("step1", "a1"), ("step2", "a2")),
            failure_type=FailureType.RMAX_EXHAUSTED,
            failed_step="step3",
            failed_artifact_id="a3",
            failure_feedback="Validation failed",
            provenance_ids=("a3", "a3-retry"),
        )

        assert checkpoint.workflow_id == "wf-456"
        assert checkpoint.specification == "Multi-step task"
        assert checkpoint.constraints == "Must be fast"
        assert checkpoint.rmax == 5
        assert checkpoint.completed_steps == ("step1", "step2")
        assert checkpoint.artifact_ids == (("step1", "a1"), ("step2", "a2"))
        assert checkpoint.failure_type == FailureType.RMAX_EXHAUSTED
        assert checkpoint.failed_step == "step3"
        assert checkpoint.failed_artifact_id == "a3"
        assert checkpoint.failure_feedback == "Validation failed"
        assert checkpoint.provenance_ids == ("a3", "a3-retry")

    def test_w_ref_deterministic(self) -> None:
        """Same workflow definition produces same W_ref."""
        checkpoint_dag = InMemoryCheckpointDAG()
        service = CheckpointService(checkpoint_dag)

        workflow_definition = {
            "steps": [{"guard_id": "g_test", "requires": [], "deps": []}],
            "rmax": 2,
            "constraints": "test",
        }

        checkpoint1 = service.create_checkpoint(
            workflow_definition=workflow_definition,
            workflow_id="wf-1",
            specification="spec",
            constraints="test",
            rmax=2,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.ESCALATION,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="error",
            provenance_ids=(),
        )

        checkpoint2 = service.create_checkpoint(
            workflow_definition=workflow_definition,
            workflow_id="wf-2",  # Different workflow_id
            specification="spec",
            constraints="test",
            rmax=2,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.ESCALATION,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="error",
            provenance_ids=(),
        )

        # Same workflow definition = same W_ref
        assert checkpoint1.workflow_ref == checkpoint2.workflow_ref

    def test_different_workflow_produces_different_w_ref(self) -> None:
        """Different workflow definitions produce different W_ref."""
        checkpoint_dag = InMemoryCheckpointDAG()
        service = CheckpointService(checkpoint_dag)

        workflow_def_1 = {
            "steps": [{"guard_id": "g_test", "requires": [], "deps": []}],
            "rmax": 2,
            "constraints": "",
        }

        workflow_def_2 = {
            "steps": [{"guard_id": "g_impl", "requires": [], "deps": []}],
            "rmax": 3,  # Different
            "constraints": "",
        }

        checkpoint1 = service.create_checkpoint(
            workflow_definition=workflow_def_1,
            workflow_id="wf-1",
            specification="spec",
            constraints="",
            rmax=2,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.ESCALATION,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="error",
            provenance_ids=(),
        )

        checkpoint2 = service.create_checkpoint(
            workflow_definition=workflow_def_2,
            workflow_id="wf-2",
            specification="spec",
            constraints="",
            rmax=3,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.ESCALATION,
            failed_step="g_impl",
            failed_artifact_id=None,
            failure_feedback="error",
            provenance_ids=(),
        )

        # Different workflow definition = different W_ref
        assert checkpoint1.workflow_ref != checkpoint2.workflow_ref


class TestCheckpointServiceGetAndList:
    """Tests for get_checkpoint and list_checkpoints methods."""

    def test_get_checkpoint(self) -> None:
        """get_checkpoint retrieves checkpoint by ID."""
        checkpoint_dag = InMemoryCheckpointDAG()
        service = CheckpointService(checkpoint_dag)

        workflow_definition = {"steps": [], "rmax": 1, "constraints": ""}

        checkpoint = service.create_checkpoint(
            workflow_definition=workflow_definition,
            workflow_id="wf-test",
            specification="spec",
            constraints="",
            rmax=1,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.ESCALATION,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="error",
            provenance_ids=(),
        )

        retrieved = service.get_checkpoint(checkpoint.checkpoint_id)
        assert retrieved.checkpoint_id == checkpoint.checkpoint_id

    def test_list_checkpoints_all(self) -> None:
        """list_checkpoints returns all checkpoints."""
        checkpoint_dag = InMemoryCheckpointDAG()
        service = CheckpointService(checkpoint_dag)

        workflow_definition = {"steps": [], "rmax": 1, "constraints": ""}

        # Create two checkpoints
        service.create_checkpoint(
            workflow_definition=workflow_definition,
            workflow_id="wf-1",
            specification="spec1",
            constraints="",
            rmax=1,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.ESCALATION,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="error",
            provenance_ids=(),
        )
        service.create_checkpoint(
            workflow_definition=workflow_definition,
            workflow_id="wf-2",
            specification="spec2",
            constraints="",
            rmax=1,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.ESCALATION,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="error",
            provenance_ids=(),
        )

        all_checkpoints = service.list_checkpoints()
        assert len(all_checkpoints) == 2

    def test_list_checkpoints_filtered_by_workflow_id(self) -> None:
        """list_checkpoints filters by workflow_id."""
        checkpoint_dag = InMemoryCheckpointDAG()
        service = CheckpointService(checkpoint_dag)

        workflow_definition = {"steps": [], "rmax": 1, "constraints": ""}

        # Create checkpoints for different workflows
        service.create_checkpoint(
            workflow_definition=workflow_definition,
            workflow_id="wf-alpha",
            specification="spec",
            constraints="",
            rmax=1,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.ESCALATION,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="error",
            provenance_ids=(),
        )
        service.create_checkpoint(
            workflow_definition=workflow_definition,
            workflow_id="wf-beta",
            specification="spec",
            constraints="",
            rmax=1,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.ESCALATION,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="error",
            provenance_ids=(),
        )

        alpha_checkpoints = service.list_checkpoints(workflow_id="wf-alpha")
        assert len(alpha_checkpoints) == 1
        assert alpha_checkpoints[0].workflow_id == "wf-alpha"
