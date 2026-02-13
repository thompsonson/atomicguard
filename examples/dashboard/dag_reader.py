"""Read-only DAG wrapper for the dashboard."""

from __future__ import annotations

from pathlib import Path

from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG
from atomicguard.visualization.html_exporter import (
    WorkflowVisualizationData,
    _serialize_artifact,
    extract_workflow_data,
)


class DAGReader:
    """Read-only wrapper around FilesystemArtifactDAG."""

    def __init__(self, dag_dir: Path) -> None:
        self._dag = FilesystemArtifactDAG(str(dag_dir))

    def get_visualization_data(self) -> WorkflowVisualizationData:
        """Extract full visualization data (nodes, edges, artifacts, runs)."""
        return extract_workflow_data(self._dag)

    def get_artifact_detail(self, artifact_id: str) -> dict:
        """Serialize a single artifact for the sidebar."""
        return _serialize_artifact(self._dag.get_artifact(artifact_id))
