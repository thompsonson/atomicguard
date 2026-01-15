"""
WorkspaceService: Bidirectional filesystem ↔ DAG synchronization.

Responsibilities:
- Materialize artifacts to filesystem
- Capture filesystem changes to artifact format
- Manage workspace lifecycle (create/cleanup)

Does NOT:
- Validate content (Guard's job)
- Decide what to materialize (Orchestrator's job)
- Store artifacts (DAG's job)
- Make LLM calls
"""

import hashlib
import json
import shutil
from pathlib import Path

from .interfaces import IWorkspaceService, WorkspaceManifest


class WorkspaceService(IWorkspaceService):
    """Manages ephemeral workspaces for agent execution.

    Each phase gets a temporary workspace where:
    1. Upstream artifacts are materialized (JSON → files)
    2. Agent operates on real filesystem (Claude SDK)
    3. Changes are captured back (files → JSON)
    4. Workspace is cleaned up

    The workspace is the ONLY place where filesystem operations occur.
    The DAG remains the source of truth for all artifacts.
    """

    def __init__(self, base_dir: Path, persist: bool = False):
        """Initialize workspace service.

        Args:
            base_dir: Root directory for all workspaces
            persist: If True, keep workspaces after cleanup (for debugging)
        """
        self.base_dir = base_dir
        self.persist = persist
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_workspace(self, phase_id: str) -> Path:
        """Create an ephemeral workspace for a phase.

        Args:
            phase_id: Identifier for the phase (e.g., "g_ddd")

        Returns:
            Path to the workspace directory

        Example:
            >>> service = WorkspaceService(Path("/tmp/atomicguard"))
            >>> workspace = service.create_workspace("g_ddd")
            >>> # workspace = /tmp/atomicguard/g_ddd/
        """
        workspace = self.base_dir / phase_id
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def materialize(self, manifest: WorkspaceManifest, workspace: Path) -> None:
        """Write artifact files to filesystem.

        Args:
            manifest: Files to materialize
            workspace: Target directory

        Raises:
            IOError: If filesystem write fails

        Example:
            >>> manifest = WorkspaceManifest(
            ...     files=[
            ...         {"path": "docs/domain_model.md", "content": "# Domain Model"},
            ...     ],
            ...     metadata={}
            ... )
            >>> service.materialize(manifest, workspace)
            >>> # Now workspace/docs/domain_model.md exists
        """
        for file_entry in manifest.files:
            file_path = workspace / file_entry["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)

            content = file_entry["content"]
            file_path.write_text(content, encoding="utf-8")

    def capture(
        self, workspace: Path, patterns: list[str] | None = None
    ) -> WorkspaceManifest:
        """Capture filesystem changes into artifact format.

        Args:
            workspace: Source directory
            patterns: Glob patterns (default: ["**/*.py", "**/*.md", "**/*.feature"])

        Returns:
            WorkspaceManifest ready for artifact storage

        Example:
            >>> # After agent writes files to workspace
            >>> manifest = service.capture(workspace)
            >>> # manifest.files = [{"path": "src/main.py", "content": "..."}]
        """
        if patterns is None:
            patterns = ["**/*.py", "**/*.md", "**/*.feature", "**/*.txt"]

        files = []
        captured_paths = set()

        for pattern in patterns:
            for file_path in workspace.rglob(pattern):
                if not file_path.is_file():
                    continue

                # Avoid duplicates
                rel_path = file_path.relative_to(workspace)
                if str(rel_path) in captured_paths:
                    continue

                captured_paths.add(str(rel_path))

                content = file_path.read_text(encoding="utf-8")
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

                files.append(
                    {
                        "path": str(rel_path),
                        "content": content,
                        "content_hash": content_hash,
                    }
                )

        # Sort by path for determinism
        files.sort(key=lambda f: f["path"])

        metadata = {
            "total_files": len(files),
            "total_lines": sum(f["content"].count("\n") for f in files),
            "patterns": patterns,
        }

        return WorkspaceManifest(files=files, metadata=metadata)

    def cleanup(self, workspace: Path) -> None:
        """Remove workspace directory.

        Args:
            workspace: Directory to remove

        Note:
            If persist=True was set, workspace is kept for debugging.
        """
        if self.persist:
            # Keep workspace for debugging
            return

        if workspace.exists() and workspace.is_dir():
            shutil.rmtree(workspace)

    def manifest_from_artifact_content(self, content: str) -> WorkspaceManifest:
        """Parse artifact JSON content into WorkspaceManifest.

        Args:
            content: JSON string from artifact.content

        Returns:
            WorkspaceManifest

        Example:
            >>> artifact = artifact_dag.get_accepted("g_ddd")
            >>> manifest = service.manifest_from_artifact_content(artifact.content)
        """
        data = json.loads(content)

        # Handle different artifact formats
        if "files" in data:
            files = data["files"]
        else:
            # Fallback: treat entire content as single file
            files = [{"path": "artifact.json", "content": content, "content_hash": ""}]

        metadata = data.get("metadata", {})

        return WorkspaceManifest(files=files, metadata=metadata)

    def manifest_to_artifact_content(self, manifest: WorkspaceManifest) -> str:
        """Serialize WorkspaceManifest to JSON string for artifact storage.

        Args:
            manifest: Workspace manifest

        Returns:
            JSON string suitable for artifact.content

        Example:
            >>> manifest = service.capture(workspace)
            >>> content = service.manifest_to_artifact_content(manifest)
            >>> artifact = Artifact(content=content, ...)
        """
        data = {"files": manifest.files, "metadata": manifest.metadata}
        return json.dumps(data, indent=2)
