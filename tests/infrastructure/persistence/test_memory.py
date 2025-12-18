"""Tests for InMemoryArtifactDAG."""

import pytest

from atomicguard.domain.models import Artifact, ArtifactStatus
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


class TestInMemoryArtifactDAG:
    """Tests for in-memory artifact storage."""

    def test_store_and_retrieve(self, memory_dag, sample_artifact):
        """Store and retrieve an artifact."""
        artifact_id = memory_dag.store(sample_artifact)
        assert artifact_id == sample_artifact.artifact_id

        retrieved = memory_dag.get_artifact(artifact_id)
        assert retrieved.content == sample_artifact.content
        assert retrieved.artifact_id == sample_artifact.artifact_id

    def test_store_with_metadata(self, memory_dag, sample_artifact):
        """Store artifact with metadata."""
        artifact_id = memory_dag.store(sample_artifact, metadata="test metadata")
        assert artifact_id == sample_artifact.artifact_id

    def test_get_nonexistent(self, memory_dag):
        """Getting nonexistent artifact raises KeyError."""
        with pytest.raises(KeyError, match="Artifact not found"):
            memory_dag.get_artifact("nonexistent-id")

    def test_provenance_single(self, memory_dag, sample_artifact):
        """Provenance of single artifact is just itself."""
        memory_dag.store(sample_artifact)
        provenance = memory_dag.get_provenance(sample_artifact.artifact_id)
        assert len(provenance) == 1
        assert provenance[0].artifact_id == sample_artifact.artifact_id

    def test_provenance_chain(self, memory_dag, sample_context_snapshot):
        """Provenance follows previous_attempt_id chain."""
        # Create a chain of artifacts: first -> second -> third
        first = Artifact(
            artifact_id="first-001",
            content="def add(a, b): pass",
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=False,
            feedback="Incomplete",
            context=sample_context_snapshot,
        )
        second = Artifact(
            artifact_id="second-001",
            content="def add(a, b): return a",
            previous_attempt_id="first-001",
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:01Z",
            attempt_number=2,
            status=ArtifactStatus.REJECTED,
            guard_result=False,
            feedback="Wrong result",
            context=sample_context_snapshot,
        )
        third = Artifact(
            artifact_id="third-001",
            content="def add(a, b): return a + b",
            previous_attempt_id="second-001",
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:02Z",
            attempt_number=3,
            status=ArtifactStatus.ACCEPTED,
            guard_result=True,
            feedback="",
            context=sample_context_snapshot,
        )

        memory_dag.store(first)
        memory_dag.store(second)
        memory_dag.store(third)

        provenance = memory_dag.get_provenance("third-001")
        assert len(provenance) == 3
        assert provenance[0].artifact_id == "first-001"
        assert provenance[1].artifact_id == "second-001"
        assert provenance[2].artifact_id == "third-001"

    def test_multiple_artifacts(self, memory_dag, sample_context_snapshot):
        """Store and retrieve multiple independent artifacts."""
        artifacts = [
            Artifact(
                artifact_id=f"art-{i:03d}",
                content=f"# Code version {i}",
                previous_attempt_id=None,
                action_pair_id=f"ap-{i:03d}",
                created_at="2025-01-01T00:00:00Z",
                attempt_number=1,
                status=ArtifactStatus.PENDING,
                guard_result=None,
                feedback="",
                context=sample_context_snapshot,
            )
            for i in range(5)
        ]

        for artifact in artifacts:
            memory_dag.store(artifact)

        for i, _artifact in enumerate(artifacts):
            retrieved = memory_dag.get_artifact(f"art-{i:03d}")
            assert f"version {i}" in retrieved.content

    def test_store_overwrites(self, memory_dag, sample_context_snapshot):
        """Storing with same ID overwrites previous artifact."""
        original = Artifact(
            artifact_id="overwrite-001",
            content="original content",
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )
        updated = Artifact(
            artifact_id="overwrite-001",
            content="updated content",
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:01Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=True,
            feedback="",
            context=sample_context_snapshot,
        )

        memory_dag.store(original)
        memory_dag.store(updated)

        retrieved = memory_dag.get_artifact("overwrite-001")
        assert retrieved.content == "updated content"
        assert retrieved.status == ArtifactStatus.ACCEPTED

    def test_empty_dag(self):
        """Fresh DAG should be empty."""
        dag = InMemoryArtifactDAG()
        with pytest.raises(KeyError):
            dag.get_artifact("any-id")

    def test_provenance_missing_parent(self, memory_dag, sample_context_snapshot):
        """Provenance handles missing parent gracefully."""
        # Create artifact with reference to non-existent parent
        orphan = Artifact(
            artifact_id="orphan-001",
            content="orphaned artifact",
            previous_attempt_id="missing-parent",
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=2,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )
        memory_dag.store(orphan)

        # Should return just the orphan, not crash
        provenance = memory_dag.get_provenance("orphan-001")
        assert len(provenance) == 1
        assert provenance[0].artifact_id == "orphan-001"
