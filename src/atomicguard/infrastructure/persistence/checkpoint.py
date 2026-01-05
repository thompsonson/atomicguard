"""
Filesystem implementation of the Checkpoint DAG.

Provides persistent storage for workflow checkpoints and human amendments,
enabling resumable workflows after failure/escalation.
"""

import json
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import CheckpointDAGInterface
from atomicguard.domain.models import (
    AmendmentType,
    FailureType,
    HumanAmendment,
    WorkflowCheckpoint,
)


class FilesystemCheckpointDAG(CheckpointDAGInterface):
    """
    Persistent storage for workflow checkpoints and amendments.

    Directory structure:
    {base_dir}/
        checkpoints/
            {prefix}/{checkpoint_id}.json
        amendments/
            {prefix}/{amendment_id}.json
        checkpoint_index.json  # Maps workflow_id -> checkpoint_ids
    """

    def __init__(self, base_dir: str):
        self._base_dir = Path(base_dir)
        self._checkpoints_dir = self._base_dir / "checkpoints"
        self._amendments_dir = self._base_dir / "amendments"
        self._index_path = self._base_dir / "checkpoint_index.json"
        self._cache_checkpoints: dict[str, WorkflowCheckpoint] = {}
        self._cache_amendments: dict[str, HumanAmendment] = {}
        self._index: dict[str, Any] = self._load_or_create_index()

    def _load_or_create_index(self) -> dict[str, Any]:
        """Load existing index or create new one."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._amendments_dir.mkdir(parents=True, exist_ok=True)

        if self._index_path.exists():
            with open(self._index_path) as f:
                result: dict[str, Any] = json.load(f)
                return result

        return {
            "version": "1.0",
            "checkpoints": {},  # checkpoint_id -> metadata
            "amendments": {},  # amendment_id -> metadata
            "by_workflow": {},  # workflow_id -> [checkpoint_ids]
            "by_checkpoint": {},  # checkpoint_id -> [amendment_ids]
        }

    def _update_index_atomic(self) -> None:
        """Atomically update index.json using write-to-temp + rename."""
        temp_path = self._index_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self._index, f, indent=2)
        temp_path.rename(self._index_path)

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get filesystem path for checkpoint (using prefix directories)."""
        prefix = checkpoint_id[:2]
        return self._checkpoints_dir / prefix / f"{checkpoint_id}.json"

    def _get_amendment_path(self, amendment_id: str) -> Path:
        """Get filesystem path for amendment (using prefix directories)."""
        prefix = amendment_id[:2]
        return self._amendments_dir / prefix / f"{amendment_id}.json"

    def _checkpoint_to_dict(self, checkpoint: WorkflowCheckpoint) -> dict:
        """Serialize checkpoint to JSON-compatible dict."""
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "workflow_id": checkpoint.workflow_id,
            "created_at": checkpoint.created_at,
            "specification": checkpoint.specification,
            "constraints": checkpoint.constraints,
            "rmax": checkpoint.rmax,
            "completed_steps": list(checkpoint.completed_steps),
            "artifact_ids": dict(checkpoint.artifact_ids),
            "failure_type": checkpoint.failure_type.value,
            "failed_step": checkpoint.failed_step,
            "failed_artifact_id": checkpoint.failed_artifact_id,
            "failure_feedback": checkpoint.failure_feedback,
            "provenance_ids": list(checkpoint.provenance_ids),
        }

    def _dict_to_checkpoint(self, data: dict) -> WorkflowCheckpoint:
        """Deserialize checkpoint from JSON dict."""
        return WorkflowCheckpoint(
            checkpoint_id=data["checkpoint_id"],
            workflow_id=data["workflow_id"],
            created_at=data["created_at"],
            specification=data["specification"],
            constraints=data["constraints"],
            rmax=data["rmax"],
            completed_steps=tuple(data["completed_steps"]),
            artifact_ids=tuple(data["artifact_ids"].items()),
            failure_type=FailureType(data["failure_type"]),
            failed_step=data["failed_step"],
            failed_artifact_id=data["failed_artifact_id"],
            failure_feedback=data["failure_feedback"],
            provenance_ids=tuple(data["provenance_ids"]),
        )

    def _amendment_to_dict(self, amendment: HumanAmendment) -> dict:
        """Serialize amendment to JSON-compatible dict."""
        return {
            "amendment_id": amendment.amendment_id,
            "checkpoint_id": amendment.checkpoint_id,
            "amendment_type": amendment.amendment_type.value,
            "created_at": amendment.created_at,
            "created_by": amendment.created_by,
            "content": amendment.content,
            "context": amendment.context,
            "parent_artifact_id": amendment.parent_artifact_id,
            "additional_rmax": amendment.additional_rmax,
        }

    def _dict_to_amendment(self, data: dict) -> HumanAmendment:
        """Deserialize amendment from JSON dict."""
        return HumanAmendment(
            amendment_id=data["amendment_id"],
            checkpoint_id=data["checkpoint_id"],
            amendment_type=AmendmentType(data["amendment_type"]),
            created_at=data["created_at"],
            created_by=data["created_by"],
            content=data["content"],
            context=data.get("context", ""),
            parent_artifact_id=data.get("parent_artifact_id"),
            additional_rmax=data.get("additional_rmax", 0),
        )

    def store_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        """
        Store a checkpoint and return its ID.

        Args:
            checkpoint: The checkpoint to store

        Returns:
            The checkpoint_id
        """
        # 1. Serialize to JSON
        checkpoint_dict = self._checkpoint_to_dict(checkpoint)

        # 2. Write to checkpoints/{prefix}/{checkpoint_id}.json
        object_path = self._get_checkpoint_path(checkpoint.checkpoint_id)
        object_path.parent.mkdir(parents=True, exist_ok=True)
        with open(object_path, "w") as f:
            json.dump(checkpoint_dict, f, indent=2)

        # 3. Update index
        self._index["checkpoints"][checkpoint.checkpoint_id] = {
            "path": str(object_path.relative_to(self._base_dir)),
            "workflow_id": checkpoint.workflow_id,
            "failed_step": checkpoint.failed_step,
            "failure_type": checkpoint.failure_type.value,
            "created_at": checkpoint.created_at,
        }

        # Track by workflow
        if checkpoint.workflow_id not in self._index["by_workflow"]:
            self._index["by_workflow"][checkpoint.workflow_id] = []
        self._index["by_workflow"][checkpoint.workflow_id].append(
            checkpoint.checkpoint_id
        )

        # 4. Atomically update index
        self._update_index_atomic()

        # 5. Add to cache
        self._cache_checkpoints[checkpoint.checkpoint_id] = checkpoint

        return checkpoint.checkpoint_id

    def get_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint:
        """Retrieve checkpoint by ID (cache-first)."""
        # Check cache first
        if checkpoint_id in self._cache_checkpoints:
            return self._cache_checkpoints[checkpoint_id]

        # Check index
        if checkpoint_id not in self._index["checkpoints"]:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")

        # Load from filesystem
        rel_path = self._index["checkpoints"][checkpoint_id]["path"]
        object_path = self._base_dir / rel_path

        with open(object_path) as f:
            data = json.load(f)

        checkpoint = self._dict_to_checkpoint(data)
        self._cache_checkpoints[checkpoint_id] = checkpoint
        return checkpoint

    def store_amendment(self, amendment: HumanAmendment) -> str:
        """
        Store a human amendment and return its ID.

        Args:
            amendment: The amendment to store

        Returns:
            The amendment_id
        """
        # 1. Serialize to JSON
        amendment_dict = self._amendment_to_dict(amendment)

        # 2. Write to amendments/{prefix}/{amendment_id}.json
        object_path = self._get_amendment_path(amendment.amendment_id)
        object_path.parent.mkdir(parents=True, exist_ok=True)
        with open(object_path, "w") as f:
            json.dump(amendment_dict, f, indent=2)

        # 3. Update index
        self._index["amendments"][amendment.amendment_id] = {
            "path": str(object_path.relative_to(self._base_dir)),
            "checkpoint_id": amendment.checkpoint_id,
            "amendment_type": amendment.amendment_type.value,
            "created_at": amendment.created_at,
        }

        # Track by checkpoint
        if amendment.checkpoint_id not in self._index["by_checkpoint"]:
            self._index["by_checkpoint"][amendment.checkpoint_id] = []
        self._index["by_checkpoint"][amendment.checkpoint_id].append(
            amendment.amendment_id
        )

        # 4. Atomically update index
        self._update_index_atomic()

        # 5. Add to cache
        self._cache_amendments[amendment.amendment_id] = amendment

        return amendment.amendment_id

    def get_amendment(self, amendment_id: str) -> HumanAmendment:
        """Retrieve amendment by ID (cache-first)."""
        # Check cache first
        if amendment_id in self._cache_amendments:
            return self._cache_amendments[amendment_id]

        # Check index
        if amendment_id not in self._index["amendments"]:
            raise KeyError(f"Amendment not found: {amendment_id}")

        # Load from filesystem
        rel_path = self._index["amendments"][amendment_id]["path"]
        object_path = self._base_dir / rel_path

        with open(object_path) as f:
            data = json.load(f)

        amendment = self._dict_to_amendment(data)
        self._cache_amendments[amendment_id] = amendment
        return amendment

    def get_amendments_for_checkpoint(self, checkpoint_id: str) -> list[HumanAmendment]:
        """Get all amendments for a checkpoint."""
        if checkpoint_id not in self._index.get("by_checkpoint", {}):
            return []

        amendment_ids = self._index["by_checkpoint"][checkpoint_id]
        return [self.get_amendment(aid) for aid in amendment_ids]

    def list_checkpoints(
        self, workflow_id: str | None = None
    ) -> list[WorkflowCheckpoint]:
        """
        List checkpoints, optionally filtered by workflow_id.

        Args:
            workflow_id: Optional filter by workflow

        Returns:
            List of matching checkpoints, newest first
        """
        if workflow_id is not None:
            # Filter by workflow
            if workflow_id not in self._index.get("by_workflow", {}):
                return []
            checkpoint_ids = self._index["by_workflow"][workflow_id]
        else:
            # All checkpoints
            checkpoint_ids = list(self._index.get("checkpoints", {}).keys())

        # Load checkpoints and sort by created_at descending
        checkpoints = [self.get_checkpoint(cid) for cid in checkpoint_ids]
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        return checkpoints


class InMemoryCheckpointDAG(CheckpointDAGInterface):
    """
    In-memory checkpoint storage for testing.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, WorkflowCheckpoint] = {}
        self._amendments: dict[str, HumanAmendment] = {}
        self._by_workflow: dict[str, list[str]] = {}
        self._by_checkpoint: dict[str, list[str]] = {}

    def store_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint

        if checkpoint.workflow_id not in self._by_workflow:
            self._by_workflow[checkpoint.workflow_id] = []
        self._by_workflow[checkpoint.workflow_id].append(checkpoint.checkpoint_id)

        return checkpoint.checkpoint_id

    def get_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint:
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")
        return self._checkpoints[checkpoint_id]

    def store_amendment(self, amendment: HumanAmendment) -> str:
        self._amendments[amendment.amendment_id] = amendment

        if amendment.checkpoint_id not in self._by_checkpoint:
            self._by_checkpoint[amendment.checkpoint_id] = []
        self._by_checkpoint[amendment.checkpoint_id].append(amendment.amendment_id)

        return amendment.amendment_id

    def get_amendment(self, amendment_id: str) -> HumanAmendment:
        if amendment_id not in self._amendments:
            raise KeyError(f"Amendment not found: {amendment_id}")
        return self._amendments[amendment_id]

    def get_amendments_for_checkpoint(self, checkpoint_id: str) -> list[HumanAmendment]:
        if checkpoint_id not in self._by_checkpoint:
            return []
        return [self._amendments[aid] for aid in self._by_checkpoint[checkpoint_id]]

    def list_checkpoints(
        self, workflow_id: str | None = None
    ) -> list[WorkflowCheckpoint]:
        if workflow_id is not None:
            if workflow_id not in self._by_workflow:
                return []
            checkpoint_ids = self._by_workflow[workflow_id]
        else:
            checkpoint_ids = list(self._checkpoints.keys())

        checkpoints = [self._checkpoints[cid] for cid in checkpoint_ids]
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        return checkpoints
