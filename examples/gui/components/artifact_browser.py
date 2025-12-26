"""Artifact browser component with clickable list, diagram, and comparison."""

from __future__ import annotations

from collections.abc import Callable
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


def create_artifact_browser() -> (
    tuple[
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
    ]
):
    """
    Create the artifact browser component with clickable list, diagram, and comparison.

    Returns:
        Tuple of Gradio components and update functions
    """
    with gr.Column():
        gr.Markdown("## Artifact Browser")

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
                """Build tree rows for a single workflow."""
                # Group by action_pair_id
                by_action_pair: dict[str, list[str]] = {}
                for artifact_id in artifact_ids:
                    meta = index.get("artifacts", {}).get(artifact_id, {})
                    action_pair_id = meta.get("action_pair_id", "unknown")
                    if action_pair_id not in by_action_pair:
                        by_action_pair[action_pair_id] = []
                    by_action_pair[action_pair_id].append(artifact_id)

                # Preserve insertion order (execution order) - don't alphabetize
                action_pairs = list(by_action_pair.keys())
                for action_pair_id in action_pairs:
                    # Add action pair header row (non-selectable)
                    if show_workflow_prefix:
                        header = f"📦 [{wf_id[:8]}] {action_pair_id}"
                    else:
                        header = f"📦 {action_pair_id}"
                    tree_rows.append([header])
                    id_map[len(tree_rows) - 1] = None  # Header row, not selectable

                    artifacts = by_action_pair[action_pair_id]
                    for i, artifact_id in enumerate(artifacts, 1):
                        meta = index.get("artifacts", {}).get(artifact_id, {})
                        status = meta.get("status", "unknown")
                        icon = _status_icon(status)

                        is_last_attempt = i == len(artifacts)
                        branch = "└─" if is_last_attempt else "├─"

                        # Indented attempt row
                        tree_rows.append([f"    {branch} {icon} Attempt {i}"])
                        id_map[len(tree_rows) - 1] = artifact_id

                        # Build flat label for comparison dropdowns
                        flat_label = f"{icon} {action_pair_id} / Attempt {i}"
                        if show_workflow_prefix:
                            flat_label = (
                                f"{icon} [{wf_id[:8]}] {action_pair_id} / Attempt {i}"
                            )
                        flat_choices.append((flat_label, artifact_id))

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

            metadata = {
                "artifact_id": artifact.artifact_id,
                "workflow_id": artifact.workflow_id,
                "action_pair_id": artifact.action_pair_id,
                "attempt_number": artifact.attempt_number,
                "status": artifact.status.value if artifact.status else None,
                "created_at": artifact.created_at,
                "previous_attempt_id": artifact.previous_attempt_id,
            }

            provenance_rows = []
            try:
                provenance = artifact_dag.get_provenance(artifact_id)
                for art in provenance:
                    feedback = (
                        art.feedback[:100] + "..."
                        if len(art.feedback) > 100
                        else art.feedback
                    )
                    provenance_rows.append(
                        [
                            art.attempt_number,
                            art.status.value if art.status else "unknown",
                            feedback,
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

    return (
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
    )
