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

    def store(self, artifact: Artifact) -> str:
        self._artifacts[artifact.artifact_id] = artifact
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

    def get_latest_for_action_pair(
        self, action_pair_id: str, workflow_id: str
    ) -> Artifact | None:
        """
        Get the most recent artifact for an action pair in a workflow.

        Args:
            action_pair_id: The action pair identifier (e.g., 'g_test')
            workflow_id: UUID of the workflow execution instance

        Returns:
            The most recent artifact, or None if not found
        """
        candidates = [
            a
            for a in self._artifacts.values()
            if a.action_pair_id == action_pair_id and a.workflow_id == workflow_id
        ]
        if not candidates:
            return None
        # Sort by created_at descending and return the latest
        candidates.sort(key=lambda a: a.created_at, reverse=True)
        return candidates[0]
