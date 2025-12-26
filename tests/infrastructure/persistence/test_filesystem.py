"""Tests for FilesystemArtifactDAG - persistent artifact storage."""

import json

import pytest

from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    ContextSnapshot,
    FeedbackEntry,
)
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG


@pytest.fixture
def fs_dag(tmp_path):  # noqa: ANN001
    """Create a FilesystemArtifactDAG with temporary directory."""
    return FilesystemArtifactDAG(str(tmp_path / "artifacts"))


@pytest.fixture
def sample_fs_artifact() -> Artifact:
    """Create a sample artifact for filesystem tests."""
    return Artifact(
        artifact_id="abc123def456",
        content="def add(a, b):\n    return a + b",
        previous_attempt_id=None,
        action_pair_id="ap-001",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        feedback="",
        context=ContextSnapshot(
            specification="Write an add function",
            constraints="Pure Python",
            feedback_history=(),
            dependency_artifacts=(),
        ),
    )


class TestFilesystemArtifactDAGInit:
    """Tests for FilesystemArtifactDAG initialization."""

    def test_init_creates_directories(self, tmp_path) -> None:  # noqa: ANN001
        """Initialization creates base and objects directories."""
        _dag = FilesystemArtifactDAG(str(tmp_path / "artifacts"))

        assert (tmp_path / "artifacts").exists()
        assert (tmp_path / "artifacts" / "objects").exists()

    def test_init_creates_empty_index(self, tmp_path) -> None:  # noqa: ANN001
        """Initialization creates index with default structure."""
        dag = FilesystemArtifactDAG(str(tmp_path / "artifacts"))

        assert dag._index["version"] == "1.0"
        assert dag._index["artifacts"] == {}
        assert dag._index["action_pairs"] == {}

    def test_init_loads_existing_index(self, tmp_path) -> None:  # noqa: ANN001
        """Initialization loads existing index file."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "objects").mkdir()

        # Create existing index
        index = {
            "version": "1.0",
            "artifacts": {"existing-id": {"path": "objects/ex/existing-id.json"}},
            "action_pairs": {},
        }
        with open(artifacts_dir / "index.json", "w") as f:
            json.dump(index, f)

        dag = FilesystemArtifactDAG(str(artifacts_dir))

        assert "existing-id" in dag._index["artifacts"]


class TestFilesystemArtifactDAGStore:
    """Tests for FilesystemArtifactDAG.store() method."""

    def test_store_returns_artifact_id(
        self, fs_dag: FilesystemArtifactDAG, sample_fs_artifact: Artifact
    ) -> None:
        """store() returns the artifact ID."""
        result = fs_dag.store(sample_fs_artifact)

        assert result == sample_fs_artifact.artifact_id

    def test_store_creates_object_file(
        self,
        fs_dag: FilesystemArtifactDAG,
        sample_fs_artifact: Artifact,
        tmp_path,  # noqa: ANN001
    ) -> None:
        """store() creates JSON file in objects directory."""
        fs_dag.store(sample_fs_artifact)

        # Object path uses first 2 chars as prefix
        prefix = sample_fs_artifact.artifact_id[:2]
        object_path = (
            tmp_path
            / "artifacts"
            / "objects"
            / prefix
            / f"{sample_fs_artifact.artifact_id}.json"
        )
        assert object_path.exists()

    def test_store_updates_index(
        self, fs_dag: FilesystemArtifactDAG, sample_fs_artifact: Artifact
    ) -> None:
        """store() updates the index."""
        fs_dag.store(sample_fs_artifact)

        assert sample_fs_artifact.artifact_id in fs_dag._index["artifacts"]

    def test_store_tracks_action_pair(
        self, fs_dag: FilesystemArtifactDAG, sample_fs_artifact: Artifact
    ) -> None:
        """store() tracks artifacts by action pair."""
        fs_dag.store(sample_fs_artifact)

        assert sample_fs_artifact.action_pair_id in fs_dag._index["action_pairs"]
        assert (
            sample_fs_artifact.artifact_id
            in fs_dag._index["action_pairs"][sample_fs_artifact.action_pair_id]
        )

    def test_store_adds_to_cache(
        self, fs_dag: FilesystemArtifactDAG, sample_fs_artifact: Artifact
    ) -> None:
        """store() adds artifact to cache."""
        fs_dag.store(sample_fs_artifact)

        assert sample_fs_artifact.artifact_id in fs_dag._cache

    def test_store_persists_index_file(
        self,
        fs_dag: FilesystemArtifactDAG,
        sample_fs_artifact: Artifact,
        tmp_path,  # noqa: ANN001
    ) -> None:
        """store() persists index to filesystem."""
        fs_dag.store(sample_fs_artifact)

        index_path = tmp_path / "artifacts" / "index.json"
        assert index_path.exists()

        with open(index_path) as f:
            persisted_index = json.load(f)

        assert sample_fs_artifact.artifact_id in persisted_index["artifacts"]


class TestFilesystemArtifactDAGGetArtifact:
    """Tests for FilesystemArtifactDAG.get_artifact() method."""

    def test_get_artifact_from_cache(
        self, fs_dag: FilesystemArtifactDAG, sample_fs_artifact: Artifact
    ) -> None:
        """get_artifact() returns cached artifact."""
        fs_dag.store(sample_fs_artifact)

        # Clear any filesystem state but keep cache
        result = fs_dag.get_artifact(sample_fs_artifact.artifact_id)

        assert result.artifact_id == sample_fs_artifact.artifact_id
        assert result.content == sample_fs_artifact.content

    def test_get_artifact_from_filesystem(
        self, fs_dag: FilesystemArtifactDAG, sample_fs_artifact: Artifact
    ) -> None:
        """get_artifact() loads from filesystem when not cached."""
        fs_dag.store(sample_fs_artifact)

        # Clear cache to force filesystem read
        fs_dag._cache.clear()

        result = fs_dag.get_artifact(sample_fs_artifact.artifact_id)

        assert result.artifact_id == sample_fs_artifact.artifact_id
        assert result.content == sample_fs_artifact.content

    def test_get_artifact_not_found(self, fs_dag: FilesystemArtifactDAG) -> None:
        """get_artifact() raises KeyError for missing artifact."""
        with pytest.raises(KeyError, match="Artifact not found"):
            fs_dag.get_artifact("nonexistent-id")

    def test_get_artifact_caches_after_load(
        self, fs_dag: FilesystemArtifactDAG, sample_fs_artifact: Artifact
    ) -> None:
        """get_artifact() caches artifact after loading from filesystem."""
        fs_dag.store(sample_fs_artifact)
        fs_dag._cache.clear()

        fs_dag.get_artifact(sample_fs_artifact.artifact_id)

        assert sample_fs_artifact.artifact_id in fs_dag._cache


class TestFilesystemArtifactDAGGetProvenance:
    """Tests for FilesystemArtifactDAG.get_provenance() method."""

    def test_get_provenance_single_artifact(
        self, fs_dag: FilesystemArtifactDAG, sample_fs_artifact: Artifact
    ) -> None:
        """get_provenance() returns list with single artifact."""
        fs_dag.store(sample_fs_artifact)

        provenance = fs_dag.get_provenance(sample_fs_artifact.artifact_id)

        assert len(provenance) == 1
        assert provenance[0].artifact_id == sample_fs_artifact.artifact_id

    def test_get_provenance_chain(self, fs_dag: FilesystemArtifactDAG) -> None:
        """get_provenance() traces full retry chain."""
        context = ContextSnapshot(
            specification="test",
            constraints="",
            feedback_history=(),
            dependency_artifacts=(),
        )

        # Create chain: artifact1 -> artifact2 -> artifact3
        artifact1 = Artifact(
            artifact_id="chain-001",
            content="v1",
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=None,
            feedback="",
            context=context,
        )
        artifact2 = Artifact(
            artifact_id="chain-002",
            content="v2",
            previous_attempt_id="chain-001",
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:01Z",
            attempt_number=2,
            status=ArtifactStatus.REJECTED,
            guard_result=None,
            feedback="",
            context=context,
        )
        artifact3 = Artifact(
            artifact_id="chain-003",
            content="v3",
            previous_attempt_id="chain-002",
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:02Z",
            attempt_number=3,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            feedback="",
            context=context,
        )

        fs_dag.store(artifact1)
        fs_dag.store(artifact2)
        fs_dag.store(artifact3)

        provenance = fs_dag.get_provenance("chain-003")

        assert len(provenance) == 3
        assert provenance[0].artifact_id == "chain-001"
        assert provenance[1].artifact_id == "chain-002"
        assert provenance[2].artifact_id == "chain-003"


class TestFilesystemArtifactDAGGetByActionPair:
    """Tests for FilesystemArtifactDAG.get_by_action_pair() method."""

    def test_get_by_action_pair_returns_all(
        self, fs_dag: FilesystemArtifactDAG
    ) -> None:
        """get_by_action_pair() returns all artifacts for action pair."""
        context = ContextSnapshot(
            specification="test",
            constraints="",
            feedback_history=(),
            dependency_artifacts=(),
        )

        artifact1 = Artifact(
            artifact_id="ap-art-001",
            content="v1",
            previous_attempt_id=None,
            action_pair_id="ap-same",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=context,
        )
        artifact2 = Artifact(
            artifact_id="ap-art-002",
            content="v2",
            previous_attempt_id=None,
            action_pair_id="ap-same",
            created_at="2025-01-01T00:00:01Z",
            attempt_number=2,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=context,
        )

        fs_dag.store(artifact1)
        fs_dag.store(artifact2)

        results = fs_dag.get_by_action_pair("ap-same")

        assert len(results) == 2

    def test_get_by_action_pair_empty(self, fs_dag: FilesystemArtifactDAG) -> None:
        """get_by_action_pair() returns empty list for unknown action pair."""
        results = fs_dag.get_by_action_pair("unknown-ap")

        assert results == []


class TestFilesystemArtifactDAGGetAccepted:
    """Tests for FilesystemArtifactDAG.get_accepted() method."""

    def test_get_accepted_returns_accepted_artifact(
        self, fs_dag: FilesystemArtifactDAG
    ) -> None:
        """get_accepted() returns artifact with ACCEPTED status."""
        context = ContextSnapshot(
            specification="test",
            constraints="",
            feedback_history=(),
            dependency_artifacts=(),
        )

        rejected = Artifact(
            artifact_id="acc-001",
            content="bad",
            previous_attempt_id=None,
            action_pair_id="ap-acc",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=None,
            feedback="",
            context=context,
        )
        accepted = Artifact(
            artifact_id="acc-002",
            content="good",
            previous_attempt_id=None,
            action_pair_id="ap-acc",
            created_at="2025-01-01T00:00:01Z",
            attempt_number=2,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            feedback="",
            context=context,
        )

        fs_dag.store(rejected)
        fs_dag.store(accepted)

        result = fs_dag.get_accepted("ap-acc")

        assert result is not None
        assert result.artifact_id == "acc-002"
        assert result.status == ArtifactStatus.ACCEPTED

    def test_get_accepted_returns_none_when_no_accepted(
        self, fs_dag: FilesystemArtifactDAG
    ) -> None:
        """get_accepted() returns None when no accepted artifact."""
        context = ContextSnapshot(
            specification="test",
            constraints="",
            feedback_history=(),
            dependency_artifacts=(),
        )

        pending = Artifact(
            artifact_id="pend-001",
            content="pending",
            previous_attempt_id=None,
            action_pair_id="ap-pend",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=context,
        )

        fs_dag.store(pending)

        result = fs_dag.get_accepted("ap-pend")

        assert result is None


class TestFilesystemArtifactDAGUpdateStatus:
    """Tests for FilesystemArtifactDAG.update_status() method."""

    def test_update_status_changes_status(
        self, fs_dag: FilesystemArtifactDAG, sample_fs_artifact: Artifact
    ) -> None:
        """update_status() changes artifact status."""
        fs_dag.store(sample_fs_artifact)

        fs_dag.update_status(sample_fs_artifact.artifact_id, ArtifactStatus.ACCEPTED)

        updated = fs_dag.get_artifact(sample_fs_artifact.artifact_id)
        assert updated.status == ArtifactStatus.ACCEPTED

    def test_update_status_persists_to_filesystem(
        self,
        fs_dag: FilesystemArtifactDAG,
        sample_fs_artifact: Artifact,
    ) -> None:
        """update_status() persists change to filesystem."""
        fs_dag.store(sample_fs_artifact)
        fs_dag.update_status(sample_fs_artifact.artifact_id, ArtifactStatus.ACCEPTED)

        # Clear cache and reload
        fs_dag._cache.clear()
        reloaded = fs_dag.get_artifact(sample_fs_artifact.artifact_id)

        assert reloaded.status == ArtifactStatus.ACCEPTED

    def test_update_status_updates_index(
        self, fs_dag: FilesystemArtifactDAG, sample_fs_artifact: Artifact
    ) -> None:
        """update_status() updates index."""
        fs_dag.store(sample_fs_artifact)
        fs_dag.update_status(sample_fs_artifact.artifact_id, ArtifactStatus.SUPERSEDED)

        assert (
            fs_dag._index["artifacts"][sample_fs_artifact.artifact_id]["status"]
            == "superseded"
        )


class TestFilesystemArtifactDAGSerialization:
    """Tests for artifact serialization/deserialization."""

    def test_round_trip_preserves_all_fields(
        self, fs_dag: FilesystemArtifactDAG
    ) -> None:
        """Stored artifact can be retrieved with all fields intact."""
        context = ContextSnapshot(
            specification="Test spec",
            constraints="Test constraints",
            feedback_history=(
                FeedbackEntry(artifact_id="prev-001", feedback="Bad code"),
            ),
            dependency_artifacts=(("dep-key-1", "dep-001"), ("dep-key-2", "dep-002")),
        )

        artifact = Artifact(
            artifact_id="round-trip-001",
            content="def foo(): pass",
            previous_attempt_id="prev-attempt",
            action_pair_id="ap-round",
            created_at="2025-06-15T10:30:00Z",
            attempt_number=3,
            status=ArtifactStatus.ACCEPTED,
            guard_result=True,
            feedback="Looks good",
            context=context,
        )

        fs_dag.store(artifact)
        fs_dag._cache.clear()  # Force filesystem read

        loaded = fs_dag.get_artifact("round-trip-001")

        assert loaded.artifact_id == artifact.artifact_id
        assert loaded.content == artifact.content
        assert loaded.previous_attempt_id == artifact.previous_attempt_id
        assert loaded.action_pair_id == artifact.action_pair_id
        assert loaded.created_at == artifact.created_at
        assert loaded.attempt_number == artifact.attempt_number
        assert loaded.status == artifact.status
        assert loaded.guard_result == artifact.guard_result
        assert loaded.feedback == artifact.feedback
        assert loaded.context.specification == artifact.context.specification
        assert loaded.context.constraints == artifact.context.constraints
        assert len(loaded.context.feedback_history) == 1
        assert loaded.context.feedback_history[0].feedback == "Bad code"
        assert loaded.context.dependency_artifacts == (
            ("dep-key-1", "dep-001"),
            ("dep-key-2", "dep-002"),
        )
