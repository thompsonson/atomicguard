"""
In-memory implementation of the Artifact DAG.

Useful for testing and ephemeral workflows.
"""

from atomicguard.domain.interfaces import ArtifactDAGInterface
from atomicguard.domain.models import Artifact


class InMemoryArtifactDAG(ArtifactDAGInterface):
    """Simple in-memory DAG for testing."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._metadata: dict[str, str] = {}

    def store(self, artifact: Artifact, metadata: str = "") -> str:
        self._artifacts[artifact.artifact_id] = artifact
        self._metadata[artifact.artifact_id] = metadata
        return artifact.artifact_id

    def get_artifact(self, artifact_id: str) -> Artifact:
        if artifact_id not in self._artifacts:
            raise KeyError(f"Artifact not found: {artifact_id}")
        return self._artifacts[artifact_id]

    def get_provenance(self, artifact_id: str) -> list[Artifact]:
        result = []
        current = self._artifacts.get(artifact_id)
        while current:
            result.append(current)
            if (
                not hasattr(current, "previous_attempt_id")
                or current.previous_attempt_id is None
            ):
                break
            current = self._artifacts.get(current.previous_attempt_id)
        return list(reversed(result))
