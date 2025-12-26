"""
Filesystem implementation of the Artifact DAG.

Provides persistent, append-only storage for artifacts.
Implements the Versioned Repository R from Definition 4.
"""

import json
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import ArtifactDAGInterface
from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    ContextSnapshot,
    FeedbackEntry,
)


class FilesystemArtifactDAG(ArtifactDAGInterface):
    """
    Persistent, append-only artifact repository.

    Stores artifacts as JSON files with an index for efficient lookups.
    Implements the Versioned Repository R from Definition 4.
    """

    def __init__(self, base_dir: str):
        self._base_dir = Path(base_dir)
        self._objects_dir = self._base_dir / "objects"
        self._index_path = self._base_dir / "index.json"
        self._cache: dict[str, Artifact] = {}
        self._index: dict[str, Any] = self._load_or_create_index()

    def _load_or_create_index(self) -> dict[str, Any]:
        """Load existing index or create new one."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._objects_dir.mkdir(parents=True, exist_ok=True)

        if self._index_path.exists():
            with open(self._index_path) as f:
                result: dict[str, Any] = json.load(f)
                return result

        return {"version": "1.0", "artifacts": {}, "action_pairs": {}, "workflows": {}}

    def _update_index_atomic(self) -> None:
        """Atomically update index.json using write-to-temp + rename."""
        temp_path = self._index_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self._index, f, indent=2)
        temp_path.rename(self._index_path)  # Atomic on POSIX

    def _artifact_to_dict(self, artifact: Artifact) -> dict:
        """Serialize artifact to JSON-compatible dict."""
        return {
            "artifact_id": artifact.artifact_id,
            "workflow_id": artifact.workflow_id,
            "content": artifact.content,
            "previous_attempt_id": artifact.previous_attempt_id,
            "action_pair_id": artifact.action_pair_id,
            "created_at": artifact.created_at,
            "attempt_number": artifact.attempt_number,
            "status": artifact.status.value,
            "guard_result": artifact.guard_result,
            "feedback": artifact.feedback,
            "context": {
                "workflow_id": artifact.context.workflow_id,
                "specification": artifact.context.specification,
                "constraints": artifact.context.constraints,
                "feedback_history": [
                    {"artifact_id": fe.artifact_id, "feedback": fe.feedback}
                    for fe in artifact.context.feedback_history
                ],
                # Serialize tuple → dict for JSON object format (matches schema)
                "dependency_artifacts": dict(artifact.context.dependency_artifacts),
            },
        }

    def _dict_to_artifact(self, data: dict) -> Artifact:
        """Deserialize artifact from JSON dict."""
        context = ContextSnapshot(
            workflow_id=data["context"].get("workflow_id", "unknown"),
            specification=data["context"]["specification"],
            constraints=data["context"]["constraints"],
            feedback_history=tuple(
                FeedbackEntry(artifact_id=fe["artifact_id"], feedback=fe["feedback"])
                for fe in data["context"]["feedback_history"]
            ),
            # Deserialize dict → tuple for immutability
            dependency_artifacts=tuple(
                data["context"].get("dependency_artifacts", {}).items()
            ),
        )
        return Artifact(
            artifact_id=data["artifact_id"],
            workflow_id=data.get("workflow_id", "unknown"),
            content=data["content"],
            previous_attempt_id=data["previous_attempt_id"],
            action_pair_id=data["action_pair_id"],
            created_at=data["created_at"],
            attempt_number=data["attempt_number"],
            status=ArtifactStatus(data["status"]),
            guard_result=data["guard_result"],
            feedback=data["feedback"],
            context=context,
        )

    def _get_object_path(self, artifact_id: str) -> Path:
        """Get filesystem path for artifact (using prefix directories)."""
        prefix = artifact_id[:2]
        return self._objects_dir / prefix / f"{artifact_id}.json"

    def store(self, artifact: Artifact) -> str:
        """
        Append artifact to DAG (immutable, append-only).

        Args:
            artifact: The artifact to store

        Returns:
            The artifact_id
        """
        # 1. Serialize to JSON
        artifact_dict = self._artifact_to_dict(artifact)

        # 2. Write to objects/{prefix}/{artifact_id}.json
        object_path = self._get_object_path(artifact.artifact_id)
        object_path.parent.mkdir(parents=True, exist_ok=True)
        with open(object_path, "w") as f:
            json.dump(artifact_dict, f, indent=2)

        # 3. Update index
        self._index["artifacts"][artifact.artifact_id] = {
            "path": str(object_path.relative_to(self._base_dir)),
            "workflow_id": artifact.workflow_id,
            "action_pair_id": artifact.action_pair_id,
            "status": artifact.status.value,
            "created_at": artifact.created_at,
        }

        # Track by action pair
        if artifact.action_pair_id not in self._index["action_pairs"]:
            self._index["action_pairs"][artifact.action_pair_id] = []
        self._index["action_pairs"][artifact.action_pair_id].append(
            artifact.artifact_id
        )

        # Track by workflow
        if "workflows" not in self._index:
            self._index["workflows"] = {}
        if artifact.workflow_id not in self._index["workflows"]:
            self._index["workflows"][artifact.workflow_id] = []
        self._index["workflows"][artifact.workflow_id].append(artifact.artifact_id)

        # 4. Atomically update index
        self._update_index_atomic()

        # 5. Add to cache
        self._cache[artifact.artifact_id] = artifact

        return artifact.artifact_id

    def get_artifact(self, artifact_id: str) -> Artifact:
        """Retrieve artifact by ID (cache-first)."""
        # Check cache first
        if artifact_id in self._cache:
            return self._cache[artifact_id]

        # Check index
        if artifact_id not in self._index["artifacts"]:
            raise KeyError(f"Artifact not found: {artifact_id}")

        # Load from filesystem
        rel_path = self._index["artifacts"][artifact_id]["path"]
        object_path = self._base_dir / rel_path

        with open(object_path) as f:
            data = json.load(f)

        artifact = self._dict_to_artifact(data)
        self._cache[artifact_id] = artifact
        return artifact

    def get_provenance(self, artifact_id: str) -> list[Artifact]:
        """Trace retry chain via previous_attempt_id."""
        result = []
        current_id: str | None = artifact_id

        while current_id:
            artifact = self.get_artifact(current_id)
            result.append(artifact)
            current_id = artifact.previous_attempt_id

        return list(reversed(result))

    def get_by_action_pair(self, action_pair_id: str) -> list[Artifact]:
        """Get all artifacts for an action pair."""
        if action_pair_id not in self._index["action_pairs"]:
            return []

        artifact_ids = self._index["action_pairs"][action_pair_id]
        return [self.get_artifact(aid) for aid in artifact_ids]

    def get_accepted(self, action_pair_id: str) -> Artifact | None:
        """Get the accepted artifact for an action pair (if any)."""
        artifacts = self.get_by_action_pair(action_pair_id)
        for artifact in artifacts:
            if artifact.status == ArtifactStatus.ACCEPTED:
                return artifact
        return None

    def update_status(self, artifact_id: str, new_status: ArtifactStatus) -> None:
        """
        Update artifact status (e.g., mark as ACCEPTED or SUPERSEDED).

        Note: This creates a new file version, preserving append-only semantics.
        """
        artifact = self.get_artifact(artifact_id)

        # Create updated artifact (immutable, so we create new instance)
        updated = Artifact(
            artifact_id=artifact.artifact_id,
            workflow_id=artifact.workflow_id,
            content=artifact.content,
            previous_attempt_id=artifact.previous_attempt_id,
            action_pair_id=artifact.action_pair_id,
            created_at=artifact.created_at,
            attempt_number=artifact.attempt_number,
            status=new_status,
            guard_result=artifact.guard_result,
            feedback=artifact.feedback,
            context=artifact.context,
        )

        # Update file
        artifact_dict = self._artifact_to_dict(updated)
        object_path = self._get_object_path(artifact_id)
        with open(object_path, "w") as f:
            json.dump(artifact_dict, f, indent=2)

        # Update index
        self._index["artifacts"][artifact_id]["status"] = new_status.value
        self._update_index_atomic()

        # Update cache
        self._cache[artifact_id] = updated

    def get_by_workflow(self, workflow_id: str) -> list[Artifact]:
        """Get all artifacts for a workflow execution."""
        if "workflows" not in self._index:
            return []
        if workflow_id not in self._index["workflows"]:
            return []

        artifact_ids = self._index["workflows"][workflow_id]
        return [self.get_artifact(aid) for aid in artifact_ids]
