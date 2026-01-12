"""Artifact browser component with clickable list, diagram, and comparison."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

import gradio as gr

if TYPE_CHECKING:
    from atomicguard import FilesystemArtifactDAG


def _status_icon(status: str) -> str:
    """Return emoji icon for artifact status."""
    icons = {
        "accepted": "✅",
        "rejected": "❌",
        "pending": "⏳",
    }
    return icons.get(status.lower(), "⏳")


def _status_color(status: str) -> str:
    """Return color for artifact status."""
    colors = {
        "accepted": "#34D399",  # green
        "rejected": "#F87171",  # red
        "pending": "#9CA3AF",  # gray
    }
    return colors.get(status.lower(), "#9CA3AF")


def _get_workflow_choices(
    artifact_dag: FilesystemArtifactDAG | None,
) -> list[tuple[str, str | None]]:
    """
    Build workflow dropdown choices with date/time, sorted by most recent.

    Returns:
        List of (display_label, workflow_id) tuples, starting with "All Workflows"
    """
    if artifact_dag is None:
        return [("All Workflows", None)]

    try:
        index = artifact_dag._index
        workflows_index = index.get("workflows", {})
        artifacts_index = index.get("artifacts", {})

        if not workflows_index:
            return [("All Workflows", None)]

        workflow_info: list[tuple[str, str, int]] = []
        for wf_id, artifact_ids in workflows_index.items():
            # Find earliest timestamp for this workflow
            timestamps = [
                artifacts_index.get(aid, {}).get("created_at", "")
                for aid in artifact_ids
            ]
            valid_timestamps = [t for t in timestamps if t]
            earliest = min(valid_timestamps) if valid_timestamps else ""
            workflow_info.append((wf_id, earliest, len(artifact_ids)))

        # Sort by timestamp descending (most recent first)
        workflow_info.sort(key=lambda x: x[1], reverse=True)

        # Build choices: (display_label, workflow_id)
        choices: list[tuple[str, str | None]] = [("All Workflows", None)]
        for wf_id, timestamp, count in workflow_info:
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    date_str = "Unknown"
            else:
                date_str = "Unknown"

            # Format: "2025-12-26 19:56 - abc12345... (N artifacts)"
            wf_display = wf_id if len(wf_id) <= 10 else f"{wf_id[:8]}..."
            artifact_word = "artifact" if count == 1 else "artifacts"
            label = f"{date_str} - {wf_display} ({count} {artifact_word})"
            choices.append((label, wf_id))

        return choices

    except Exception:
        return [("All Workflows", None)]


def _build_artifact_diagram(
    artifact_dag: FilesystemArtifactDAG | None,
    workflow_id: str | None,
    selected_action_pair_id: str | None = None,
) -> str:
    """
    Build Mermaid diagram showing artifact flow.

    Highlights the selected artifact's action_pair node.
    """
    if artifact_dag is None:
        return "```mermaid\ngraph LR\n    empty[No artifacts]\n```"

    try:
        index = artifact_dag._index
        workflows_index = index.get("workflows", {})

        if not workflows_index:
            return "```mermaid\ngraph LR\n    empty[No artifacts]\n```"

        # Use specific workflow or most recent
        wf_id: str
        if workflow_id and workflow_id in workflows_index:
            wf_id = workflow_id
        elif workflows_index:
            wf_id = sorted(workflows_index.keys(), reverse=True)[0]
        else:
            return "```mermaid\ngraph LR\n    empty[No artifacts]\n```"

        artifact_ids = workflows_index.get(wf_id, [])

        # Group by action_pair_id and get status summary
        action_pair_stats: dict[str, dict] = {}
        for artifact_id in artifact_ids:
            meta = index.get("artifacts", {}).get(artifact_id, {})
            action_pair_id = meta.get("action_pair_id", "unknown")
            status = meta.get("status", "pending")

            if action_pair_id not in action_pair_stats:
                action_pair_stats[action_pair_id] = {
                    "attempts": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "pending": 0,
                    "final_status": "pending",
                }

            stats = action_pair_stats[action_pair_id]
            stats["attempts"] += 1
            if status.lower() in stats:
                stats[status.lower()] += 1
            stats["final_status"] = status  # Last status

        if not action_pair_stats:
            return "```mermaid\ngraph LR\n    empty[No artifacts]\n```"

        lines = ["```mermaid", "graph LR"]

        # Add nodes for each action pair
        for action_pair_id, stats in action_pair_stats.items():
            attempts = stats["attempts"]
            final_status = stats["final_status"].lower()
            color = _status_color(final_status)

            # Escape special chars for Mermaid
            safe_id = action_pair_id.replace("-", "_")

            lines.append(
                f'    {safe_id}["{action_pair_id}<br/><small>{attempts} attempt(s)</small>"]'
            )

            # Style with highlighting for selected
            is_selected = action_pair_id == selected_action_pair_id
            stroke_width = "4px" if is_selected else "2px"
            stroke_color = "#3B82F6" if is_selected else "#374151"

            lines.append(
                f"    style {safe_id} fill:{color},stroke:{stroke_color},stroke-width:{stroke_width}"
            )

        # Add edges if we can infer dependencies (from artifact order)
        action_pair_list = list(action_pair_stats.keys())
        for i in range(1, len(action_pair_list)):
            prev_id = action_pair_list[i - 1].replace("-", "_")
            curr_id = action_pair_list[i].replace("-", "_")
            lines.append(f"    {prev_id} --> {curr_id}")

        lines.append("```")
        return "\n".join(lines)

    except Exception as e:
        return f"```mermaid\ngraph LR\n    error[Error: {e}]\n```"


def create_artifact_browser() -> tuple[
    gr.Dropdown,  # workflow_dropdown (filter by workflow)
    gr.Dataframe,  # artifact_tree (clickable table)
    gr.State,  # artifact_id_map (row_idx -> artifact_id)
    gr.Markdown,  # artifact_diagram
    gr.Code,  # content_display
    gr.JSON,  # metadata_display
    gr.Dataframe,  # provenance_display
    gr.JSON,  # context_display
    gr.Dropdown,  # compare_left
    gr.Dropdown,  # compare_right
    gr.Code,  # compare_left_content
    gr.Code,  # compare_right_content
    Callable[..., Any],  # refresh_artifacts_fn
    Callable[..., Any],  # load_artifact_fn
    Callable[..., Any],  # update_diagram_fn
    Callable[..., Any],  # compare_fn
    Callable[..., Any],  # get_workflow_choices_fn
]:
    """
    Create the artifact browser component with clickable list, diagram, and comparison.

    Returns:
        Tuple of Gradio components and update functions
    """
    with gr.Column():
        gr.Markdown("## Artifact Browser")

        # Workflow filter dropdown
        workflow_dropdown = gr.Dropdown(
            label="Filter by Workflow",
            choices=[("All Workflows", None)],
            value=None,
            interactive=True,
        )

        with gr.Row():
            # Left column: Clickable artifact list and diagram
            with gr.Column(scale=1):
                gr.Markdown("### Artifacts")
                gr.Markdown(
                    "*Click a row to view artifact details*", elem_classes=["text-sm"]
                )
                artifact_tree = gr.Dataframe(
                    headers=["Artifact"],
                    datatype=["str"],
                    value=[],
                    interactive=False,
                    column_count=1,  # type: ignore[arg-type]
                    wrap=True,
                )
                # State to map row index -> artifact_id (None for header rows)
                artifact_id_map = gr.State({})

                gr.Markdown("### Artifact Flow")
                artifact_diagram = gr.Markdown(
                    value="```mermaid\ngraph LR\n    empty[No artifacts yet]\n```",
                )

            # Right column: Details and comparison
            with gr.Column(scale=2), gr.Tabs():
                with gr.Tab("Content"):
                    content_display = gr.Code(
                        label="Artifact Content",
                        language="python",
                        value="Select an artifact to view its content",
                        lines=18,
                    )

                with gr.Tab("Metadata"):
                    metadata_display = gr.JSON(
                        label="Artifact Metadata",
                        value={},
                    )

                with gr.Tab("Provenance"):
                    provenance_display = gr.Dataframe(
                        headers=["Attempt", "Status", "Feedback"],
                        label="Attempt History",
                        value=[],
                    )

                with gr.Tab("Context"):
                    context_display = gr.JSON(
                        label="Generation Context",
                        value={},
                    )

                with gr.Tab("Compare"):
                    gr.Markdown("Compare two artifacts side by side")
                    with gr.Row():
                        compare_left = gr.Dropdown(
                            label="Left Artifact",
                            choices=[],
                            value=None,
                            interactive=True,
                            scale=1,
                        )
                        compare_right = gr.Dropdown(
                            label="Right Artifact",
                            choices=[],
                            value=None,
                            interactive=True,
                            scale=1,
                        )
                    with gr.Row():
                        compare_left_content = gr.Code(
                            label="Left Content",
                            language="python",
                            value="",
                            lines=12,
                        )
                        compare_right_content = gr.Code(
                            label="Right Content",
                            language="python",
                            value="",
                            lines=12,
                        )

    def refresh_artifacts(
        artifact_dag: FilesystemArtifactDAG | None,
        workflow_id: str | None = None,
    ) -> tuple[Any, dict[int, str | None], Any, Any]:
        """
        Refresh the artifact list from the DAG.

        Returns updates for: artifact_tree, artifact_id_map, compare_left, compare_right

        Tree format uses indentation to show hierarchy:
        📦 action_pair_id
            ├─ ✅ Attempt 1
            └─ ❌ Attempt 2
        """
        if artifact_dag is None:
            empty_update = gr.update(value=[])
            dropdown_empty = gr.update(choices=[], value=None)
            return empty_update, {}, dropdown_empty, dropdown_empty

        try:
            index = artifact_dag._index
            tree_rows: list[list[str]] = []  # For Dataframe (single column)
            id_map: dict[
                int, str | None
            ] = {}  # row_idx -> artifact_id (None for headers)
            flat_choices: list[tuple[str, str]] = []  # For comparison dropdowns

            workflows_index = index.get("workflows", {})

            def _build_tree_for_workflow(
                wf_id: str,
                artifact_ids: list[str],
                show_workflow_prefix: bool = False,
            ) -> None:
                """Build tree rows for a single workflow with hierarchy support."""
                # Group by action_pair_id
                by_action_pair: dict[str, list[str]] = {}
                # Track parent relationships
                parent_map: dict[
                    str, str | None
                ] = {}  # action_pair_id -> parent_action_pair_id

                for artifact_id in artifact_ids:
                    meta = index.get("artifacts", {}).get(artifact_id, {})
                    action_pair_id = meta.get("action_pair_id", "unknown")
                    parent_action_pair_id = meta.get("parent_action_pair_id")

                    if action_pair_id not in by_action_pair:
                        by_action_pair[action_pair_id] = []
                    by_action_pair[action_pair_id].append(artifact_id)

                    # Track parent (use first seen)
                    if action_pair_id not in parent_map:
                        parent_map[action_pair_id] = parent_action_pair_id

                # Build hierarchy: separate root action pairs from children
                root_action_pairs: list[str] = []
                children_by_parent: dict[str, list[str]] = {}

                for action_pair_id in by_action_pair:
                    parent_id = parent_map.get(action_pair_id)
                    if parent_id is None:
                        root_action_pairs.append(action_pair_id)
                    else:
                        if parent_id not in children_by_parent:
                            children_by_parent[parent_id] = []
                        children_by_parent[parent_id].append(action_pair_id)

                # Handle virtual parents: if children reference a parent that doesn't
                # have its own artifacts, create a virtual entry for it
                for parent_id in list(children_by_parent.keys()):
                    if parent_id not in by_action_pair:
                        root_action_pairs.append(parent_id)
                        by_action_pair[parent_id] = []  # No artifacts, just a container

                def _render_action_pair(
                    action_pair_id: str,
                    indent: str = "",
                    is_child: bool = False,
                ) -> None:
                    """Render an action pair and its artifacts, then recurse into children."""
                    # Add action pair header row (non-selectable)
                    header_prefix = f"{indent}├─ 📦" if is_child else "📦"

                    if show_workflow_prefix and not is_child:
                        header = f"{header_prefix} [{wf_id[:8]}] {action_pair_id}"
                    else:
                        header = f"{header_prefix} {action_pair_id}"
                    tree_rows.append([header])
                    id_map[len(tree_rows) - 1] = None  # Header row, not selectable

                    artifacts = by_action_pair.get(action_pair_id, [])
                    child_indent = indent + "│   " if is_child else "    "

                    # Check if this action pair has children
                    has_children = action_pair_id in children_by_parent

                    for i, artifact_id in enumerate(artifacts, 1):
                        meta = index.get("artifacts", {}).get(artifact_id, {})
                        status = meta.get("status", "unknown")
                        icon = _status_icon(status)

                        is_last_attempt = i == len(artifacts) and not has_children
                        branch = "└─" if is_last_attempt else "├─"

                        # Indented attempt row
                        tree_rows.append([f"{child_indent}{branch} {icon} Attempt {i}"])
                        id_map[len(tree_rows) - 1] = artifact_id

                        # Build flat label for comparison dropdowns
                        flat_label = f"{icon} {action_pair_id} / Attempt {i}"
                        if show_workflow_prefix:
                            flat_label = (
                                f"{icon} [{wf_id[:8]}] {action_pair_id} / Attempt {i}"
                            )
                        flat_choices.append((flat_label, artifact_id))

                    # Render children of this action pair
                    children = children_by_parent.get(action_pair_id, [])
                    for child_action_pair_id in children:
                        _render_action_pair(
                            child_action_pair_id,
                            indent=child_indent,
                            is_child=True,
                        )

                # Render all root action pairs
                for action_pair_id in root_action_pairs:
                    _render_action_pair(action_pair_id)

            if workflow_id and workflow_id in workflows_index:
                artifact_ids = workflows_index[workflow_id]
                _build_tree_for_workflow(
                    workflow_id, artifact_ids, show_workflow_prefix=False
                )
            else:
                # Show all workflows
                for wf_id in sorted(workflows_index.keys(), reverse=True):
                    artifact_ids = workflows_index[wf_id]
                    _build_tree_for_workflow(
                        wf_id, artifact_ids, show_workflow_prefix=True
                    )

            dataframe_update = gr.update(value=tree_rows)
            dropdown_update = gr.update(choices=flat_choices, value=None)
            return dataframe_update, id_map, dropdown_update, dropdown_update

        except Exception:
            empty_update = gr.update(value=[])
            dropdown_empty = gr.update(choices=[], value=None)
            return empty_update, {}, dropdown_empty, dropdown_empty

    def load_artifact(
        artifact_id: str | None,
        artifact_dag: FilesystemArtifactDAG | None,
    ) -> tuple[str, dict[str, Any], list, dict[str, Any], str | None]:
        """
        Load artifact details.

        Returns:
            Tuple of (content, metadata, provenance_rows, context, action_pair_id)
        """
        if not artifact_id or artifact_dag is None:
            return (
                "Select an artifact to view its content",
                {},
                [],
                {},
                None,
            )

        try:
            artifact = artifact_dag.get_artifact(artifact_id)

            content = artifact.content

            # Build guard_result metadata
            guard_result_data = None
            if artifact.guard_result is not None:
                guard_result_data = {
                    "passed": artifact.guard_result.passed,
                    "feedback": artifact.guard_result.feedback,
                    "fatal": artifact.guard_result.fatal,
                    "guard_name": artifact.guard_result.guard_name,
                    "sub_results": [
                        {
                            "guard_name": sr.guard_name,
                            "passed": sr.passed,
                            "feedback": sr.feedback,
                            "execution_time_ms": sr.execution_time_ms,
                        }
                        for sr in artifact.guard_result.sub_results
                    ],
                }

            metadata = {
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

            provenance_rows = []
            try:
                provenance = artifact_dag.get_provenance(artifact_id)
                for art in provenance:
                    # Build feedback showing all sub-guard results for composite guards
                    feedback_parts = []
                    if art.guard_result is not None:
                        if art.guard_result.sub_results:
                            # Show each sub-guard result
                            for sr in art.guard_result.sub_results:
                                status_icon = "✓" if sr.passed else "✗"
                                if sr.passed:
                                    feedback_parts.append(
                                        f"{sr.guard_name}: {status_icon} Success"
                                    )
                                else:
                                    # Truncate failure feedback
                                    short_feedback = sr.feedback.split("\n")[0][:60]
                                    if len(sr.feedback) > 60:
                                        short_feedback += "..."
                                    feedback_parts.append(
                                        f"{sr.guard_name}: {status_icon} {short_feedback}"
                                    )
                        else:
                            # Non-composite guard - show main feedback
                            feedback = art.guard_result.feedback
                            feedback = (
                                feedback[:100] + "..."
                                if len(feedback) > 100
                                else feedback
                            )
                            feedback_parts.append(feedback)

                    feedback_display = (
                        "\n".join(feedback_parts) if feedback_parts else ""
                    )
                    provenance_rows.append(
                        [
                            art.attempt_number,
                            art.status.value if art.status else "unknown",
                            feedback_display,
                        ]
                    )
            except Exception:
                pass

            context: dict[str, Any] = {}
            if artifact.context:
                ctx = artifact.context
                context = {
                    "workflow_id": ctx.workflow_id,
                    "specification_preview": ctx.specification[:500]
                    + ("..." if len(ctx.specification) > 500 else ""),
                    "constraints": ctx.constraints,
                    "feedback_history_count": len(ctx.feedback_history),
                    "dependency_artifacts": list(ctx.dependency_artifacts),
                }

                if ctx.feedback_history:
                    context["feedback_history"] = [
                        {"artifact_id": h.artifact_id, "feedback": h.feedback}
                        for h in ctx.feedback_history[:5]
                    ]

            return content, metadata, provenance_rows, context, artifact.action_pair_id

        except KeyError:
            return (
                f"Artifact not found: {artifact_id}",
                {"error": "Artifact not found"},
                [],
                {},
                None,
            )
        except Exception as e:
            return (
                f"Error loading artifact: {e}",
                {"error": str(e)},
                [],
                {},
                None,
            )

    def update_diagram(
        artifact_dag: FilesystemArtifactDAG | None,
        workflow_id: str | None,
        selected_action_pair_id: str | None = None,
    ) -> str:
        """Update artifact diagram with highlighting."""
        return _build_artifact_diagram(
            artifact_dag, workflow_id, selected_action_pair_id
        )

    def compare_artifacts(
        left_id: str | None,
        right_id: str | None,
        artifact_dag: FilesystemArtifactDAG | None,
    ) -> tuple[str, str]:
        """Load content for comparison."""
        left_content = ""
        right_content = ""

        if artifact_dag is not None:
            if left_id:
                try:
                    left_content = artifact_dag.get_artifact(left_id).content
                except Exception:
                    left_content = f"Error loading {left_id}"

            if right_id:
                try:
                    right_content = artifact_dag.get_artifact(right_id).content
                except Exception:
                    right_content = f"Error loading {right_id}"

        return left_content, right_content

    def get_workflow_choices(
        artifact_dag: FilesystemArtifactDAG | None,
    ) -> list[tuple[str, str | None]]:
        """Get workflow choices for the dropdown."""
        return _get_workflow_choices(artifact_dag)

    return (
        workflow_dropdown,
        artifact_tree,
        artifact_id_map,
        artifact_diagram,
        content_display,
        metadata_display,
        provenance_display,
        context_display,
        compare_left,
        compare_right,
        compare_left_content,
        compare_right_content,
        refresh_artifacts,
        load_artifact,
        update_diagram,
        compare_artifacts,
        get_workflow_choices,
    )
