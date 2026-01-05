"""Tests for checkpoint persistence - FilesystemCheckpointDAG and InMemoryCheckpointDAG."""

import json

import pytest

from atomicguard.domain.models import (
    AmendmentType,
    FailureType,
    HumanAmendment,
    WorkflowCheckpoint,
)
from atomicguard.infrastructure.persistence.checkpoint import (
    FilesystemCheckpointDAG,
    InMemoryCheckpointDAG,
)

# =============================================================================
# FilesystemCheckpointDAG Tests
# =============================================================================


class TestFilesystemCheckpointDAGInit:
    """Tests for FilesystemCheckpointDAG initialization."""

    def test_creates_directories(self, tmp_path) -> None:
        """Initialization creates checkpoints/ and amendments/ directories."""
        _dag = FilesystemCheckpointDAG(str(tmp_path / "checkpoints"))

        assert (tmp_path / "checkpoints").exists()
        assert (tmp_path / "checkpoints" / "checkpoints").exists()
        assert (tmp_path / "checkpoints" / "amendments").exists()

    def test_creates_index_file(self, tmp_path) -> None:
        """Initialization creates checkpoint_index.json with default structure."""
        dag = FilesystemCheckpointDAG(str(tmp_path / "checkpoints"))

        assert dag._index["version"] == "1.0"
        assert dag._index["checkpoints"] == {}
        assert dag._index["amendments"] == {}
        assert dag._index["by_workflow"] == {}
        assert dag._index["by_checkpoint"] == {}

    def test_loads_existing_index(self, tmp_path) -> None:
        """Initialization loads existing index file."""
        base_dir = tmp_path / "checkpoints"
        base_dir.mkdir(parents=True)
        (base_dir / "checkpoints").mkdir()
        (base_dir / "amendments").mkdir()

        # Create existing index
        index = {
            "version": "1.0",
            "checkpoints": {
                "existing-chk": {"path": "checkpoints/ex/existing-chk.json"}
            },
            "amendments": {},
            "by_workflow": {},
            "by_checkpoint": {},
        }
        with open(base_dir / "checkpoint_index.json", "w") as f:
            json.dump(index, f)

        dag = FilesystemCheckpointDAG(str(base_dir))

        assert "existing-chk" in dag._index["checkpoints"]


class TestFilesystemCheckpointDAGStoreCheckpoint:
    """Tests for FilesystemCheckpointDAG.store_checkpoint() method."""

    def test_store_creates_json_file(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_checkpoint: WorkflowCheckpoint,
        tmp_path,
    ) -> None:
        """store_checkpoint() creates JSON file in checkpoints directory."""
        fs_checkpoint_dag.store_checkpoint(sample_checkpoint)

        # Object path uses first 2 chars as prefix
        prefix = sample_checkpoint.checkpoint_id[:2]
        object_path = (
            tmp_path
            / "checkpoints"
            / "checkpoints"
            / prefix
            / f"{sample_checkpoint.checkpoint_id}.json"
        )
        assert object_path.exists()

    def test_store_updates_index(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_checkpoint: WorkflowCheckpoint,
    ) -> None:
        """store_checkpoint() updates the index."""
        fs_checkpoint_dag.store_checkpoint(sample_checkpoint)

        assert (
            sample_checkpoint.checkpoint_id in fs_checkpoint_dag._index["checkpoints"]
        )

    def test_store_tracks_by_workflow(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_checkpoint: WorkflowCheckpoint,
    ) -> None:
        """store_checkpoint() tracks checkpoints by workflow_id."""
        fs_checkpoint_dag.store_checkpoint(sample_checkpoint)

        assert sample_checkpoint.workflow_id in fs_checkpoint_dag._index["by_workflow"]
        assert (
            sample_checkpoint.checkpoint_id
            in fs_checkpoint_dag._index["by_workflow"][sample_checkpoint.workflow_id]
        )

    def test_store_returns_checkpoint_id(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_checkpoint: WorkflowCheckpoint,
    ) -> None:
        """store_checkpoint() returns the checkpoint_id."""
        result = fs_checkpoint_dag.store_checkpoint(sample_checkpoint)

        assert result == sample_checkpoint.checkpoint_id


class TestFilesystemCheckpointDAGGetCheckpoint:
    """Tests for FilesystemCheckpointDAG.get_checkpoint() method."""

    def test_get_from_cache(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_checkpoint: WorkflowCheckpoint,
    ) -> None:
        """get_checkpoint() returns cached checkpoint."""
        fs_checkpoint_dag.store_checkpoint(sample_checkpoint)

        result = fs_checkpoint_dag.get_checkpoint(sample_checkpoint.checkpoint_id)

        assert result.checkpoint_id == sample_checkpoint.checkpoint_id
        assert result.workflow_id == sample_checkpoint.workflow_id

    def test_get_from_filesystem(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_checkpoint: WorkflowCheckpoint,
    ) -> None:
        """get_checkpoint() loads from filesystem when not cached."""
        fs_checkpoint_dag.store_checkpoint(sample_checkpoint)

        # Clear cache to force filesystem read
        fs_checkpoint_dag._cache_checkpoints.clear()

        result = fs_checkpoint_dag.get_checkpoint(sample_checkpoint.checkpoint_id)

        assert result.checkpoint_id == sample_checkpoint.checkpoint_id
        assert result.specification == sample_checkpoint.specification

    def test_get_missing_raises_keyerror(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """get_checkpoint() raises KeyError for unknown ID."""
        with pytest.raises(KeyError, match="Checkpoint not found"):
            fs_checkpoint_dag.get_checkpoint("nonexistent-id")


class TestFilesystemCheckpointDAGStoreAmendment:
    """Tests for FilesystemCheckpointDAG.store_amendment() method."""

    def test_store_amendment_creates_json(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_amendment: HumanAmendment,
        tmp_path,
    ) -> None:
        """store_amendment() creates JSON file in amendments directory."""
        fs_checkpoint_dag.store_amendment(sample_amendment)

        prefix = sample_amendment.amendment_id[:2]
        object_path = (
            tmp_path
            / "checkpoints"
            / "amendments"
            / prefix
            / f"{sample_amendment.amendment_id}.json"
        )
        assert object_path.exists()

    def test_store_amendment_updates_index(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_amendment: HumanAmendment,
    ) -> None:
        """store_amendment() updates the index."""
        fs_checkpoint_dag.store_amendment(sample_amendment)

        assert sample_amendment.amendment_id in fs_checkpoint_dag._index["amendments"]

    def test_store_amendment_tracks_by_checkpoint(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_amendment: HumanAmendment,
    ) -> None:
        """store_amendment() tracks amendments by checkpoint_id."""
        fs_checkpoint_dag.store_amendment(sample_amendment)

        assert (
            sample_amendment.checkpoint_id in fs_checkpoint_dag._index["by_checkpoint"]
        )
        assert (
            sample_amendment.amendment_id
            in fs_checkpoint_dag._index["by_checkpoint"][sample_amendment.checkpoint_id]
        )


class TestFilesystemCheckpointDAGGetAmendment:
    """Tests for FilesystemCheckpointDAG.get_amendment() method."""

    def test_get_amendment_success(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_amendment: HumanAmendment,
    ) -> None:
        """get_amendment() retrieves stored amendment."""
        fs_checkpoint_dag.store_amendment(sample_amendment)

        result = fs_checkpoint_dag.get_amendment(sample_amendment.amendment_id)

        assert result.amendment_id == sample_amendment.amendment_id
        assert result.content == sample_amendment.content

    def test_get_amendment_missing_raises_keyerror(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """get_amendment() raises KeyError for unknown ID."""
        with pytest.raises(KeyError, match="Amendment not found"):
            fs_checkpoint_dag.get_amendment("nonexistent-id")


class TestFilesystemCheckpointDAGListCheckpoints:
    """Tests for FilesystemCheckpointDAG.list_checkpoints() method."""

    def test_list_all_checkpoints(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_checkpoint: WorkflowCheckpoint,
        sample_checkpoint_escalation: WorkflowCheckpoint,
    ) -> None:
        """list_checkpoints() returns all checkpoints."""
        fs_checkpoint_dag.store_checkpoint(sample_checkpoint)
        fs_checkpoint_dag.store_checkpoint(sample_checkpoint_escalation)

        result = fs_checkpoint_dag.list_checkpoints()

        assert len(result) == 2
        checkpoint_ids = [c.checkpoint_id for c in result]
        assert sample_checkpoint.checkpoint_id in checkpoint_ids
        assert sample_checkpoint_escalation.checkpoint_id in checkpoint_ids

    def test_list_by_workflow_id(
        self,
        fs_checkpoint_dag: FilesystemCheckpointDAG,
        sample_checkpoint: WorkflowCheckpoint,
        sample_checkpoint_escalation: WorkflowCheckpoint,
    ) -> None:
        """list_checkpoints() filters by workflow_id."""
        fs_checkpoint_dag.store_checkpoint(sample_checkpoint)
        fs_checkpoint_dag.store_checkpoint(sample_checkpoint_escalation)

        result = fs_checkpoint_dag.list_checkpoints(
            workflow_id=sample_checkpoint.workflow_id
        )

        assert len(result) == 1
        assert result[0].checkpoint_id == sample_checkpoint.checkpoint_id

    def test_list_empty_returns_empty(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """list_checkpoints() returns empty list when no checkpoints."""
        result = fs_checkpoint_dag.list_checkpoints()

        assert result == []

    def test_list_sorted_by_created_at(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """list_checkpoints() returns newest first."""
        older = WorkflowCheckpoint(
            checkpoint_id="chk-older",
            workflow_id="wf-sort",
            created_at="2025-01-05T09:00:00Z",
            specification="older",
            constraints="",
            rmax=3,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.RMAX_EXHAUSTED,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="",
            provenance_ids=(),
        )
        newer = WorkflowCheckpoint(
            checkpoint_id="chk-newer",
            workflow_id="wf-sort",
            created_at="2025-01-05T10:00:00Z",
            specification="newer",
            constraints="",
            rmax=3,
            completed_steps=(),
            artifact_ids=(),
            failure_type=FailureType.RMAX_EXHAUSTED,
            failed_step="g_test",
            failed_artifact_id=None,
            failure_feedback="",
            provenance_ids=(),
        )

        fs_checkpoint_dag.store_checkpoint(older)
        fs_checkpoint_dag.store_checkpoint(newer)

        result = fs_checkpoint_dag.list_checkpoints()

        assert result[0].checkpoint_id == "chk-newer"
        assert result[1].checkpoint_id == "chk-older"


class TestFilesystemCheckpointDAGSerialization:
    """Tests for checkpoint/amendment serialization round-trips."""

    def test_checkpoint_round_trip(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """All checkpoint fields preserved through JSON serialization."""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="chk-round-trip",
            workflow_id="wf-round-trip",
            created_at="2025-01-05T10:00:00Z",
            specification="Full specification\nwith newlines",
            constraints="Must use stdlib only",
            rmax=5,
            completed_steps=("g_config", "g_test", "g_impl"),
            artifact_ids=(
                ("g_config", "art-1"),
                ("g_test", "art-2"),
                ("g_impl", "art-3"),
            ),
            failure_type=FailureType.ESCALATION,
            failed_step="g_deploy",
            failed_artifact_id="art-4",
            failure_feedback="Security issue: credentials exposed",
            provenance_ids=("art-4", "art-5", "art-6"),
        )

        fs_checkpoint_dag.store_checkpoint(checkpoint)
        fs_checkpoint_dag._cache_checkpoints.clear()

        loaded = fs_checkpoint_dag.get_checkpoint("chk-round-trip")

        assert loaded.checkpoint_id == checkpoint.checkpoint_id
        assert loaded.workflow_id == checkpoint.workflow_id
        assert loaded.created_at == checkpoint.created_at
        assert loaded.specification == checkpoint.specification
        assert loaded.constraints == checkpoint.constraints
        assert loaded.rmax == checkpoint.rmax
        assert loaded.completed_steps == checkpoint.completed_steps
        assert loaded.artifact_ids == checkpoint.artifact_ids
        assert loaded.failure_type == checkpoint.failure_type
        assert loaded.failed_step == checkpoint.failed_step
        assert loaded.failed_artifact_id == checkpoint.failed_artifact_id
        assert loaded.failure_feedback == checkpoint.failure_feedback
        assert loaded.provenance_ids == checkpoint.provenance_ids

    def test_amendment_round_trip(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """All amendment fields preserved through JSON serialization."""
        amendment = HumanAmendment(
            amendment_id="amd-round-trip",
            checkpoint_id="chk-001",
            amendment_type=AmendmentType.FEEDBACK,
            created_at="2025-01-05T10:05:00Z",
            created_by="test-admin",
            content="Use asyncio instead of threading",
            context="Previous attempt had race conditions",
            parent_artifact_id="art-failed",
            additional_rmax=3,
        )

        fs_checkpoint_dag.store_amendment(amendment)
        fs_checkpoint_dag._cache_amendments.clear()

        loaded = fs_checkpoint_dag.get_amendment("amd-round-trip")

        assert loaded.amendment_id == amendment.amendment_id
        assert loaded.checkpoint_id == amendment.checkpoint_id
        assert loaded.amendment_type == amendment.amendment_type
        assert loaded.created_at == amendment.created_at
        assert loaded.created_by == amendment.created_by
        assert loaded.content == amendment.content
        assert loaded.context == amendment.context
        assert loaded.parent_artifact_id == amendment.parent_artifact_id
        assert loaded.additional_rmax == amendment.additional_rmax

    def test_failure_type_enum_serialization(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """FailureType enum serializes to/from string correctly."""
        for failure_type in FailureType:
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=f"chk-enum-{failure_type.value}",
                workflow_id="wf-enum",
                created_at="2025-01-05T10:00:00Z",
                specification="test",
                constraints="",
                rmax=3,
                completed_steps=(),
                artifact_ids=(),
                failure_type=failure_type,
                failed_step="g_test",
                failed_artifact_id=None,
                failure_feedback="",
                provenance_ids=(),
            )

            fs_checkpoint_dag.store_checkpoint(checkpoint)
            fs_checkpoint_dag._cache_checkpoints.clear()

            loaded = fs_checkpoint_dag.get_checkpoint(checkpoint.checkpoint_id)
            assert loaded.failure_type == failure_type

    def test_amendment_type_enum_serialization(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """AmendmentType enum serializes to/from string correctly."""
        for amendment_type in AmendmentType:
            amendment = HumanAmendment(
                amendment_id=f"amd-enum-{amendment_type.value}",
                checkpoint_id="chk-001",
                amendment_type=amendment_type,
                created_at="2025-01-05T10:00:00Z",
                created_by="test",
                content="test content",
            )

            fs_checkpoint_dag.store_amendment(amendment)
            fs_checkpoint_dag._cache_amendments.clear()

            loaded = fs_checkpoint_dag.get_amendment(amendment.amendment_id)
            assert loaded.amendment_type == amendment_type

    def test_tuple_fields_preserved(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """Tuple fields (completed_steps, artifact_ids, provenance_ids) preserved."""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="chk-tuples",
            workflow_id="wf-tuples",
            created_at="2025-01-05T10:00:00Z",
            specification="test",
            constraints="",
            rmax=3,
            completed_steps=("step1", "step2", "step3"),
            artifact_ids=(("step1", "art1"), ("step2", "art2")),
            failure_type=FailureType.RMAX_EXHAUSTED,
            failed_step="step4",
            failed_artifact_id="art-fail",
            failure_feedback="",
            provenance_ids=("art-a", "art-b", "art-c"),
        )

        fs_checkpoint_dag.store_checkpoint(checkpoint)
        fs_checkpoint_dag._cache_checkpoints.clear()

        loaded = fs_checkpoint_dag.get_checkpoint("chk-tuples")

        # Verify tuples are restored correctly
        assert isinstance(loaded.completed_steps, tuple)
        assert loaded.completed_steps == ("step1", "step2", "step3")

        assert isinstance(loaded.artifact_ids, tuple)
        assert loaded.artifact_ids == (("step1", "art1"), ("step2", "art2"))

        assert isinstance(loaded.provenance_ids, tuple)
        assert loaded.provenance_ids == ("art-a", "art-b", "art-c")


class TestFilesystemCheckpointDAGGetAmendmentsForCheckpoint:
    """Tests for FilesystemCheckpointDAG.get_amendments_for_checkpoint()."""

    def test_returns_linked_amendments(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """Returns all amendments linked to a checkpoint."""
        amd1 = HumanAmendment(
            amendment_id="amd-linked-1",
            checkpoint_id="chk-target",
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="user1",
            content="fix1",
        )
        amd2 = HumanAmendment(
            amendment_id="amd-linked-2",
            checkpoint_id="chk-target",
            amendment_type=AmendmentType.FEEDBACK,
            created_at="2025-01-05T10:05:00Z",
            created_by="user2",
            content="fix2",
        )
        amd_other = HumanAmendment(
            amendment_id="amd-other",
            checkpoint_id="chk-other",
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:10:00Z",
            created_by="user3",
            content="other",
        )

        fs_checkpoint_dag.store_amendment(amd1)
        fs_checkpoint_dag.store_amendment(amd2)
        fs_checkpoint_dag.store_amendment(amd_other)

        result = fs_checkpoint_dag.get_amendments_for_checkpoint("chk-target")

        assert len(result) == 2
        amendment_ids = [a.amendment_id for a in result]
        assert "amd-linked-1" in amendment_ids
        assert "amd-linked-2" in amendment_ids
        assert "amd-other" not in amendment_ids

    def test_returns_empty_for_unknown_checkpoint(
        self, fs_checkpoint_dag: FilesystemCheckpointDAG
    ) -> None:
        """Returns empty list for unknown checkpoint."""
        result = fs_checkpoint_dag.get_amendments_for_checkpoint("unknown-chk")

        assert result == []


# =============================================================================
# InMemoryCheckpointDAG Tests
# =============================================================================


class TestInMemoryCheckpointDAG:
    """Tests for InMemoryCheckpointDAG."""

    def test_store_and_get_checkpoint(
        self,
        memory_checkpoint_dag: InMemoryCheckpointDAG,
        sample_checkpoint: WorkflowCheckpoint,
    ) -> None:
        """Basic checkpoint store/retrieve works."""
        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)

        result = memory_checkpoint_dag.get_checkpoint(sample_checkpoint.checkpoint_id)

        assert result.checkpoint_id == sample_checkpoint.checkpoint_id
        assert result.workflow_id == sample_checkpoint.workflow_id

    def test_store_and_get_amendment(
        self,
        memory_checkpoint_dag: InMemoryCheckpointDAG,
        sample_amendment: HumanAmendment,
    ) -> None:
        """Basic amendment store/retrieve works."""
        memory_checkpoint_dag.store_amendment(sample_amendment)

        result = memory_checkpoint_dag.get_amendment(sample_amendment.amendment_id)

        assert result.amendment_id == sample_amendment.amendment_id
        assert result.content == sample_amendment.content

    def test_list_checkpoints(
        self,
        memory_checkpoint_dag: InMemoryCheckpointDAG,
        sample_checkpoint: WorkflowCheckpoint,
        sample_checkpoint_escalation: WorkflowCheckpoint,
    ) -> None:
        """list_checkpoints() returns all and filters correctly."""
        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)
        memory_checkpoint_dag.store_checkpoint(sample_checkpoint_escalation)

        # List all
        all_results = memory_checkpoint_dag.list_checkpoints()
        assert len(all_results) == 2

        # Filter by workflow
        filtered = memory_checkpoint_dag.list_checkpoints(
            workflow_id=sample_checkpoint.workflow_id
        )
        assert len(filtered) == 1
        assert filtered[0].checkpoint_id == sample_checkpoint.checkpoint_id

    def test_get_amendments_for_checkpoint(
        self, memory_checkpoint_dag: InMemoryCheckpointDAG
    ) -> None:
        """get_amendments_for_checkpoint() returns linked amendments."""
        amd1 = HumanAmendment(
            amendment_id="amd-mem-1",
            checkpoint_id="chk-mem",
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="user",
            content="fix",
        )
        amd2 = HumanAmendment(
            amendment_id="amd-mem-2",
            checkpoint_id="chk-mem",
            amendment_type=AmendmentType.FEEDBACK,
            created_at="2025-01-05T10:05:00Z",
            created_by="user",
            content="hint",
        )

        memory_checkpoint_dag.store_amendment(amd1)
        memory_checkpoint_dag.store_amendment(amd2)

        result = memory_checkpoint_dag.get_amendments_for_checkpoint("chk-mem")

        assert len(result) == 2

    def test_get_checkpoint_missing_raises_keyerror(
        self, memory_checkpoint_dag: InMemoryCheckpointDAG
    ) -> None:
        """get_checkpoint() raises KeyError for unknown ID."""
        with pytest.raises(KeyError, match="Checkpoint not found"):
            memory_checkpoint_dag.get_checkpoint("missing")

    def test_get_amendment_missing_raises_keyerror(
        self, memory_checkpoint_dag: InMemoryCheckpointDAG
    ) -> None:
        """get_amendment() raises KeyError for unknown ID."""
        with pytest.raises(KeyError, match="Amendment not found"):
            memory_checkpoint_dag.get_amendment("missing")
