"""Artifact Viewer TUI Application."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.syntax import Syntax
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Pretty,
    Select,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)
from textual.widgets.tree import TreeNode

if TYPE_CHECKING:
    from atomicguard import FilesystemArtifactDAG
    from atomicguard.domain.models import Artifact


def _status_icon(status: str) -> str:
    """Return emoji icon for artifact status."""
    icons = {
        "accepted": "\u2705",  # ✅
        "rejected": "\u274c",  # ❌
        "pending": "\u23f3",   # ⏳
        "superseded": "\u25cb", # ○
    }
    return icons.get(status.lower(), "\u23f3")


def _status_style(status: str) -> str:
    """Return Rich style for artifact status."""
    styles = {
        "accepted": "bold green",
        "rejected": "bold red",
        "pending": "dim",
        "superseded": "dim italic",
    }
    return styles.get(status.lower(), "")


class ArtifactTree(Tree[str]):
    """Tree widget for displaying artifacts grouped by action pair."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._artifact_map: dict[str, str] = {}  # node_id -> artifact_id

    def set_artifact_map(self, mapping: dict[str, str]) -> None:
        """Set the mapping from tree node IDs to artifact IDs."""
        self._artifact_map = mapping

    def get_artifact_id(self, node_id: str) -> str | None:
        """Get artifact ID for a tree node."""
        return self._artifact_map.get(node_id)


class ContentPanel(Static):
    """Panel for displaying artifact content with syntax highlighting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._content = ""
        self._language = "python"

    def set_content(self, content: str, language: str = "python") -> None:
        """Set the content to display."""
        self._content = content
        self._language = language
        self._render_content()

    def _render_content(self) -> None:
        """Render the content with syntax highlighting."""
        if not self._content:
            self.update("Select an artifact to view its content")
            return

        syntax = Syntax(
            self._content,
            self._language,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )
        self.update(syntax)


class MetadataPanel(Static):
    """Panel for displaying artifact metadata."""

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Set the metadata to display."""
        if not metadata:
            self.update("No metadata available")
            return

        lines = []
        for key, value in metadata.items():
            if isinstance(value, dict):
                lines.append(f"[bold cyan]{key}:[/bold cyan]")
                for k, v in value.items():
                    lines.append(f"  [dim]{k}:[/dim] {v}")
            else:
                lines.append(f"[bold cyan]{key}:[/bold cyan] {value}")

        self.update(Text.from_markup("\n".join(lines)))


class ProvenancePanel(Static):
    """Panel for displaying artifact provenance/retry history."""

    def set_provenance(self, provenance: list[dict[str, Any]]) -> None:
        """Set the provenance data to display."""
        if not provenance:
            self.update("No provenance history available")
            return

        lines = ["[bold underline]Attempt History[/bold underline]\n"]
        for entry in provenance:
            attempt = entry.get("attempt", "?")
            status = entry.get("status", "unknown")
            feedback = entry.get("feedback", "")
            icon = _status_icon(status)
            style = _status_style(status)

            lines.append(f"[{style}]{icon} Attempt {attempt} - {status}[/{style}]")
            if feedback:
                # Truncate long feedback
                if len(feedback) > 200:
                    feedback = feedback[:200] + "..."
                lines.append(f"  [dim]{feedback}[/dim]")
            lines.append("")

        self.update(Text.from_markup("\n".join(lines)))


class ContextPanel(Static):
    """Panel for displaying artifact generation context."""

    def set_context(self, context: dict[str, Any]) -> None:
        """Set the context data to display."""
        if not context:
            self.update("No context available")
            return

        lines = []

        if "specification_preview" in context:
            lines.append("[bold cyan]Specification:[/bold cyan]")
            lines.append(f"  {context['specification_preview']}")
            lines.append("")

        if "constraints" in context:
            lines.append("[bold cyan]Constraints:[/bold cyan]")
            lines.append(f"  {context['constraints']}")
            lines.append("")

        if "dependency_artifacts" in context:
            deps = context["dependency_artifacts"]
            if deps:
                lines.append("[bold cyan]Dependencies:[/bold cyan]")
                for dep in deps:
                    if isinstance(dep, (list, tuple)) and len(dep) >= 2:
                        lines.append(f"  - {dep[0]}: {dep[1][:12]}...")
                lines.append("")

        if "feedback_history" in context:
            history = context["feedback_history"]
            if history:
                lines.append(f"[bold cyan]Feedback History ({len(history)} entries):[/bold cyan]")
                for entry in history[:3]:  # Show first 3
                    lines.append(f"  - {entry.get('feedback', '')[:80]}...")
                lines.append("")

        self.update(Text.from_markup("\n".join(lines)) if lines else "No context data")


class ArtifactViewerApp(App[None]):
    """TUI Application for browsing AtomicGuard artifacts."""

    TITLE = "AtomicGuard Artifact Viewer"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("l", "load", "Load Directory"),
        Binding("1", "tab_content", "Content"),
        Binding("2", "tab_metadata", "Metadata"),
        Binding("3", "tab_provenance", "Provenance"),
        Binding("4", "tab_context", "Context"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self, artifact_path: str | None = None) -> None:
        super().__init__()
        self._artifact_path = artifact_path
        self._dag: FilesystemArtifactDAG | None = None
        self._current_workflow: str | None = None
        self._selected_artifact_id: str | None = None
        self._artifact_node_map: dict[str, str] = {}  # node_id -> artifact_id

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

        with Horizontal(id="main-container"):
            # Left sidebar
            with Vertical(id="sidebar"):
                yield Label("Directory:", classes="section-label")
                yield Input(
                    placeholder="Enter artifact directory path...",
                    id="path-input",
                    value=self._artifact_path or "",
                )
                yield Button("Load", id="load-btn", variant="primary")

                yield Label("Workflow Filter:", classes="section-label")
                yield Select(
                    [(Text("All Workflows"), None)],
                    id="workflow-select",
                    allow_blank=False,
                )

                yield Label("Artifacts:", classes="section-label")
                yield ArtifactTree("Artifacts", id="artifact-tree")

            # Right main area
            with Vertical(id="main-area"):
                with TabbedContent(id="detail-tabs"):
                    with TabPane("Content", id="tab-content"):
                        with VerticalScroll():
                            yield ContentPanel(id="content-panel")

                    with TabPane("Metadata", id="tab-metadata"):
                        with VerticalScroll():
                            yield MetadataPanel(id="metadata-panel")

                    with TabPane("Provenance", id="tab-provenance"):
                        with VerticalScroll():
                            yield ProvenancePanel(id="provenance-panel")

                    with TabPane("Context", id="tab-context"):
                        with VerticalScroll():
                            yield ContextPanel(id="context-panel")

        yield Footer()

    def on_mount(self) -> None:
        """Handle application mount."""
        if self._artifact_path:
            self._load_directory(self._artifact_path)

    @on(Button.Pressed, "#load-btn")
    def on_load_button(self) -> None:
        """Handle load button press."""
        path_input = self.query_one("#path-input", Input)
        self._load_directory(path_input.value)

    @on(Input.Submitted, "#path-input")
    def on_path_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key in path input."""
        self._load_directory(event.value)

    @on(Select.Changed, "#workflow-select")
    def on_workflow_changed(self, event: Select.Changed) -> None:
        """Handle workflow filter change."""
        self._current_workflow = event.value
        self._refresh_tree()

    @on(Tree.NodeSelected, "#artifact-tree")
    def on_tree_select(self, event: Tree.NodeSelected[str]) -> None:
        """Handle artifact tree selection."""
        tree = self.query_one("#artifact-tree", ArtifactTree)
        node_id = str(event.node.id) if event.node.id else ""
        artifact_id = tree.get_artifact_id(node_id)

        if artifact_id:
            self._selected_artifact_id = artifact_id
            self._load_artifact_details(artifact_id)

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_refresh(self) -> None:
        """Refresh the artifact tree."""
        if self._dag:
            self._refresh_tree()
            self.notify("Refreshed artifact list")

    def action_load(self) -> None:
        """Focus the path input for loading."""
        self.query_one("#path-input", Input).focus()

    def action_tab_content(self) -> None:
        """Switch to content tab."""
        self.query_one("#detail-tabs", TabbedContent).active = "tab-content"

    def action_tab_metadata(self) -> None:
        """Switch to metadata tab."""
        self.query_one("#detail-tabs", TabbedContent).active = "tab-metadata"

    def action_tab_provenance(self) -> None:
        """Switch to provenance tab."""
        self.query_one("#detail-tabs", TabbedContent).active = "tab-provenance"

    def action_tab_context(self) -> None:
        """Switch to context tab."""
        self.query_one("#detail-tabs", TabbedContent).active = "tab-context"

    def action_cursor_down(self) -> None:
        """Move cursor down in tree."""
        tree = self.query_one("#artifact-tree", ArtifactTree)
        tree.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up in tree."""
        tree = self.query_one("#artifact-tree", ArtifactTree)
        tree.action_cursor_up()

    def _load_directory(self, path: str) -> None:
        """Load artifacts from directory."""
        if not path:
            self.notify("Please enter a directory path", severity="warning")
            return

        path_obj = Path(path).expanduser().resolve()
        if not path_obj.exists():
            self.notify(f"Directory not found: {path}", severity="error")
            return

        try:
            # Import here to avoid circular imports
            from atomicguard import FilesystemArtifactDAG

            self._dag = FilesystemArtifactDAG(str(path_obj))

            # Count artifacts
            all_artifacts = self._dag.get_all()
            count = len(all_artifacts)

            self.notify(f"Loaded {count} artifacts from {path_obj.name}")

            # Update workflow dropdown
            self._update_workflow_choices()

            # Refresh tree
            self._refresh_tree()

        except Exception as e:
            self.notify(f"Error loading: {e}", severity="error")

    def _update_workflow_choices(self) -> None:
        """Update the workflow dropdown with available workflows."""
        if self._dag is None:
            return

        try:
            index = self._dag._index
            workflows_index = index.get("workflows", {})
            artifacts_index = index.get("artifacts", {})

            choices: list[tuple[Text, str | None]] = [(Text("All Workflows"), None)]

            # Build workflow info with timestamps
            workflow_info: list[tuple[str, str, int]] = []
            for wf_id, artifact_ids in workflows_index.items():
                timestamps = [
                    artifacts_index.get(aid, {}).get("created_at", "")
                    for aid in artifact_ids
                ]
                valid_timestamps = [t for t in timestamps if t]
                earliest = min(valid_timestamps) if valid_timestamps else ""
                workflow_info.append((wf_id, earliest, len(artifact_ids)))

            # Sort by timestamp descending
            workflow_info.sort(key=lambda x: x[1], reverse=True)

            for wf_id, timestamp, count in workflow_info:
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        date_str = dt.strftime("%Y-%m-%d %H:%M")
                    except ValueError:
                        date_str = "Unknown"
                else:
                    date_str = "Unknown"

                wf_display = wf_id if len(wf_id) <= 10 else f"{wf_id[:8]}..."
                label = f"{date_str} - {wf_display} ({count})"
                choices.append((Text(label), wf_id))

            select = self.query_one("#workflow-select", Select)
            select.set_options(choices)

        except Exception:
            pass

    def _refresh_tree(self) -> None:
        """Refresh the artifact tree."""
        if self._dag is None:
            return

        tree = self.query_one("#artifact-tree", ArtifactTree)
        tree.clear()
        self._artifact_node_map.clear()

        try:
            index = self._dag._index
            workflows_index = index.get("workflows", {})
            artifacts_index = index.get("artifacts", {})

            # Determine which workflows to show
            if self._current_workflow and self._current_workflow in workflows_index:
                workflows_to_show = {self._current_workflow: workflows_index[self._current_workflow]}
            else:
                workflows_to_show = workflows_index

            node_counter = 0

            for wf_id in sorted(workflows_to_show.keys(), reverse=True):
                artifact_ids = workflows_to_show[wf_id]

                # Group by action_pair_id
                by_action_pair: dict[str, list[str]] = {}
                for artifact_id in artifact_ids:
                    meta = artifacts_index.get(artifact_id, {})
                    action_pair_id = meta.get("action_pair_id", "unknown")
                    if action_pair_id not in by_action_pair:
                        by_action_pair[action_pair_id] = []
                    by_action_pair[action_pair_id].append(artifact_id)

                # Add workflow node if showing multiple workflows
                if len(workflows_to_show) > 1:
                    wf_display = wf_id[:12] + "..." if len(wf_id) > 12 else wf_id
                    wf_node = tree.root.add(f"\U0001f4c1 {wf_display}", expand=True)
                    parent_node = wf_node
                else:
                    parent_node = tree.root

                # Add action pairs
                for action_pair_id in sorted(by_action_pair.keys()):
                    artifacts = by_action_pair[action_pair_id]
                    ap_node = parent_node.add(f"\U0001f4e6 {action_pair_id}", expand=True)

                    # Add artifacts
                    for i, artifact_id in enumerate(artifacts, 1):
                        meta = artifacts_index.get(artifact_id, {})
                        status = meta.get("status", "pending")
                        icon = _status_icon(status)

                        node_id = f"artifact_{node_counter}"
                        node_counter += 1

                        ap_node.add_leaf(f"{icon} Attempt {i}", data=node_id)
                        self._artifact_node_map[node_id] = artifact_id

            tree.set_artifact_map(self._artifact_node_map)
            tree.root.expand_all()

        except Exception as e:
            self.notify(f"Error refreshing tree: {e}", severity="error")

    def _load_artifact_details(self, artifact_id: str) -> None:
        """Load and display artifact details."""
        if self._dag is None:
            return

        try:
            artifact = self._dag.get_artifact(artifact_id)

            # Update content panel
            content_panel = self.query_one("#content-panel", ContentPanel)
            language = self._detect_language(artifact.content)
            content_panel.set_content(artifact.content, language)

            # Update metadata panel
            metadata = self._build_metadata(artifact)
            metadata_panel = self.query_one("#metadata-panel", MetadataPanel)
            metadata_panel.set_metadata(metadata)

            # Update provenance panel
            provenance = self._build_provenance(artifact_id)
            provenance_panel = self.query_one("#provenance-panel", ProvenancePanel)
            provenance_panel.set_provenance(provenance)

            # Update context panel
            context = self._build_context(artifact)
            context_panel = self.query_one("#context-panel", ContextPanel)
            context_panel.set_context(context)

        except Exception as e:
            self.notify(f"Error loading artifact: {e}", severity="error")

    def _detect_language(self, content: str) -> str:
        """Detect programming language from content."""
        content_lower = content.lower()
        if "def " in content or "import " in content or "class " in content_lower:
            return "python"
        if "function " in content or "const " in content or "=>" in content:
            return "javascript"
        if "{" in content and "}" in content:
            if "public " in content or "private " in content:
                return "java"
        return "python"  # Default

    def _build_metadata(self, artifact: Artifact) -> dict[str, Any]:
        """Build metadata dict from artifact."""
        guard_result_data = None
        if artifact.guard_result is not None:
            guard_result_data = {
                "passed": artifact.guard_result.passed,
                "feedback": artifact.guard_result.feedback[:100] + "..."
                if len(artifact.guard_result.feedback) > 100
                else artifact.guard_result.feedback,
                "fatal": artifact.guard_result.fatal,
                "guard_name": artifact.guard_result.guard_name,
            }

        return {
            "artifact_id": artifact.artifact_id,
            "workflow_id": artifact.workflow_id,
            "action_pair_id": artifact.action_pair_id,
            "parent_action_pair_id": artifact.parent_action_pair_id,
            "attempt_number": artifact.attempt_number,
            "status": artifact.status.value if artifact.status else None,
            "created_at": artifact.created_at,
            "previous_attempt_id": artifact.previous_attempt_id,
            "guard_result": guard_result_data,
        }

    def _build_provenance(self, artifact_id: str) -> list[dict[str, Any]]:
        """Build provenance history for artifact."""
        if self._dag is None:
            return []

        try:
            provenance = self._dag.get_provenance(artifact_id)
            result = []
            for art in provenance:
                feedback = ""
                if art.guard_result is not None:
                    if art.guard_result.sub_results:
                        parts = []
                        for sr in art.guard_result.sub_results:
                            status_icon = "\u2713" if sr.passed else "\u2717"
                            if sr.passed:
                                parts.append(f"{sr.guard_name}: {status_icon}")
                            else:
                                short = sr.feedback.split("\n")[0][:60]
                                parts.append(f"{sr.guard_name}: {status_icon} {short}")
                        feedback = " | ".join(parts)
                    else:
                        feedback = art.guard_result.feedback

                result.append({
                    "attempt": art.attempt_number,
                    "status": art.status.value if art.status else "unknown",
                    "feedback": feedback,
                })
            return result
        except Exception:
            return []

    def _build_context(self, artifact: Artifact) -> dict[str, Any]:
        """Build context dict from artifact."""
        if artifact.context is None:
            return {}

        ctx = artifact.context
        result: dict[str, Any] = {
            "workflow_id": ctx.workflow_id,
        }

        if ctx.specification:
            result["specification_preview"] = (
                ctx.specification[:500] + "..."
                if len(ctx.specification) > 500
                else ctx.specification
            )

        if ctx.constraints:
            result["constraints"] = ctx.constraints

        if ctx.dependency_artifacts:
            result["dependency_artifacts"] = list(ctx.dependency_artifacts)

        if ctx.feedback_history:
            result["feedback_history"] = [
                {"artifact_id": h.artifact_id, "feedback": h.feedback}
                for h in ctx.feedback_history[:5]
            ]

        return result


def main() -> None:
    """Main entry point for the TUI."""
    parser = argparse.ArgumentParser(
        description="AtomicGuard Artifact Viewer TUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m examples.tui.artifact_viewer.app ./artifact_dag
  python -m examples.tui.artifact_viewer.app ~/projects/my-atomicguard/artifact_dag

Keyboard Shortcuts:
  q          Quit
  r          Refresh artifact list
  l          Focus load path input
  1-4        Switch tabs (Content, Metadata, Provenance, Context)
  j/k        Navigate tree up/down
  Enter      Select artifact
        """,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to artifact_dag directory",
    )

    args = parser.parse_args()

    app = ArtifactViewerApp(artifact_path=args.path)
    app.run()


if __name__ == "__main__":
    main()
