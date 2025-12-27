#!/usr/bin/env python3
"""
Standalone Artifact Viewer - Browse workflow artifacts without execution.

A simplified Gradio app for viewing artifacts from AtomicGuard workflow executions.
Unlike the full GUI (run.py), this app focuses only on artifact browsing - no
workflow execution capability.

Usage:
    python -m examples.gui.artifact_viewer
    python -m examples.gui.artifact_viewer --artifact-dir ./output/artifacts
    python -m examples.gui.artifact_viewer --port 7861

Features:
    - Browse artifacts by workflow and action pair
    - View artifact content, metadata, and provenance
    - Compare artifacts side-by-side
    - Mermaid diagram of artifact flow
"""

from __future__ import annotations

from pathlib import Path

import click
import gradio as gr

from atomicguard import FilesystemArtifactDAG

from .components.artifact_browser import create_artifact_browser


def create_viewer_app(default_artifact_dir: str | None = None) -> gr.Blocks:
    """
    Create the standalone artifact viewer app.

    Args:
        default_artifact_dir: Pre-populate the artifact directory input

    Returns:
        Configured Gradio Blocks app
    """
    with gr.Blocks(
        title="AtomicGuard Artifact Viewer",
    ) as app:
        gr.Markdown("# Artifact Viewer")
        gr.Markdown("Browse artifacts from AtomicGuard workflow executions")

        # Directory input section
        with gr.Row():
            artifact_dir_input = gr.Textbox(
                label="Artifact Directory",
                placeholder="./output/artifacts or absolute path",
                value=default_artifact_dir or "",
                scale=4,
            )
            load_btn = gr.Button("Load Artifacts", variant="primary", scale=1)

        # Status message
        status_msg = gr.Markdown("")

        # Hidden state for artifact_dag instance
        artifact_dag_state = gr.State(None)

        # Create the artifact browser component (reuses existing implementation)
        (
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
            refresh_artifacts_fn,
            load_artifact_fn,
            update_diagram_fn,
            compare_fn,
            get_workflow_choices_fn,
        ) = create_artifact_browser()

        # Event handlers
        def load_directory(dir_path: str) -> tuple:
            """Load artifacts from the specified directory."""
            if not dir_path.strip():
                return (
                    None,
                    "⚠️ Please enter an artifact directory path",
                    gr.update(),
                    gr.update(),
                    {},
                    gr.update(),
                    gr.update(),
                )

            path = Path(dir_path).expanduser().resolve()

            if not path.exists():
                return (
                    None,
                    f"❌ Directory not found: {path}",
                    gr.update(),
                    gr.update(),
                    {},
                    gr.update(),
                    gr.update(),
                )

            if not path.is_dir():
                return (
                    None,
                    f"❌ Not a directory: {path}",
                    gr.update(),
                    gr.update(),
                    {},
                    gr.update(),
                    gr.update(),
                )

            try:
                dag = FilesystemArtifactDAG(str(path))

                # Check if there are any artifacts
                index = dag._index
                artifact_count = len(index.get("artifacts", {}))
                workflow_count = len(index.get("workflows", {}))

                if artifact_count == 0:
                    return (
                        dag,
                        f"⚠️ No artifacts found in: {path}",
                        gr.update(choices=[("All Workflows", None)], value=None),
                        gr.update(value=[]),
                        {},
                        gr.update(),
                        gr.update(choices=[], value=None),
                    )

                # Get workflow choices for dropdown
                workflow_choices = get_workflow_choices_fn(dag)

                # Default to most recent workflow (first non-None choice after "All Workflows")
                default_workflow_id = None
                if len(workflow_choices) > 1:
                    # workflow_choices[0] is "All Workflows", [1] is most recent
                    default_workflow_id = workflow_choices[1][1]

                # Refresh the artifact browser with the most recent workflow
                tree, id_map, left_dropdown, right_dropdown = refresh_artifacts_fn(
                    dag, default_workflow_id
                )
                diagram = update_diagram_fn(dag, default_workflow_id, None)

                return (
                    dag,
                    f"✅ Loaded {artifact_count} artifacts from {workflow_count} workflow(s): {path}",
                    gr.update(choices=workflow_choices, value=default_workflow_id),
                    tree,
                    id_map,
                    diagram,
                    left_dropdown,
                )

            except Exception as e:
                return (
                    None,
                    f"❌ Error loading artifacts: {e}",
                    gr.update(),
                    gr.update(),
                    {},
                    gr.update(),
                    gr.update(),
                )

        # Wire up load button
        load_btn.click(
            fn=load_directory,
            inputs=[artifact_dir_input],
            outputs=[
                artifact_dag_state,
                status_msg,
                workflow_dropdown,
                artifact_tree,
                artifact_id_map,
                artifact_diagram,
                compare_left,
            ],
        )

        # Also load on Enter key in the textbox
        artifact_dir_input.submit(
            fn=load_directory,
            inputs=[artifact_dir_input],
            outputs=[
                artifact_dag_state,
                status_msg,
                workflow_dropdown,
                artifact_tree,
                artifact_id_map,
                artifact_diagram,
                compare_left,
            ],
        )

        # Handle workflow filter dropdown change
        def on_workflow_filter_change(
            workflow_id: str | None,
            dag: FilesystemArtifactDAG | None,
        ) -> tuple:
            """Filter artifacts by workflow when dropdown changes."""
            if dag is None:
                return gr.update(), {}, gr.update(), gr.update()

            # Refresh artifacts with the selected workflow filter
            tree, id_map, left_dropdown, right_dropdown = refresh_artifacts_fn(
                dag, workflow_id
            )
            diagram = update_diagram_fn(dag, workflow_id, None)

            return tree, id_map, diagram, left_dropdown

        workflow_dropdown.change(
            fn=on_workflow_filter_change,
            inputs=[workflow_dropdown, artifact_dag_state],
            outputs=[artifact_tree, artifact_id_map, artifact_diagram, compare_left],
        )

        # Handle artifact selection from the tree
        def on_artifact_select(
            evt: gr.SelectData,
            id_map: dict,
            dag: FilesystemArtifactDAG | None,
            current_workflow_id: str | None,
        ) -> tuple:
            """Handle clicking on an artifact in the tree."""
            if dag is None:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            row_idx = evt.index[0]
            artifact_id = id_map.get(row_idx)

            # Header rows have None as artifact_id
            if artifact_id is None:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            # Load artifact details
            content, metadata, provenance, context, action_pair_id = load_artifact_fn(
                artifact_id, dag
            )

            # Update diagram with highlight, preserving the current workflow filter
            diagram = update_diagram_fn(dag, current_workflow_id, action_pair_id)

            return content, metadata, provenance, context, diagram

        artifact_tree.select(
            fn=on_artifact_select,
            inputs=[artifact_id_map, artifact_dag_state, workflow_dropdown],
            outputs=[
                content_display,
                metadata_display,
                provenance_display,
                context_display,
                artifact_diagram,
            ],
        )

        # Handle comparison dropdown changes
        def on_compare_change(
            left_id: str | None,
            right_id: str | None,
            dag: FilesystemArtifactDAG | None,
        ) -> tuple[str, str]:
            """Update comparison content when dropdowns change."""
            return compare_fn(left_id, right_id, dag)

        compare_left.change(
            fn=on_compare_change,
            inputs=[compare_left, compare_right, artifact_dag_state],
            outputs=[compare_left_content, compare_right_content],
        )

        compare_right.change(
            fn=on_compare_change,
            inputs=[compare_left, compare_right, artifact_dag_state],
            outputs=[compare_left_content, compare_right_content],
        )

        # Auto-load if default directory provided and exists
        if default_artifact_dir:
            app.load(
                fn=load_directory,
                inputs=[artifact_dir_input],
                outputs=[
                    artifact_dag_state,
                    status_msg,
                    workflow_dropdown,
                    artifact_tree,
                    artifact_id_map,
                    artifact_diagram,
                    compare_left,
                ],
            )

    return app


@click.command()
@click.option(
    "--port",
    default=7861,
    type=int,
    help="Port for the Gradio server (default: 7861)",
)
@click.option(
    "--artifact-dir",
    default=None,
    type=click.Path(),
    help="Default artifact directory to load on startup",
)
@click.option(
    "--share",
    is_flag=True,
    help="Create a public sharing link via Gradio",
)
def main(port: int, artifact_dir: str | None, share: bool) -> None:
    """Launch the AtomicGuard Artifact Viewer."""
    click.echo("=" * 50)
    click.echo("AtomicGuard Artifact Viewer")
    click.echo("=" * 50)
    click.echo(f"Port: {port}")
    if artifact_dir:
        click.echo(f"Artifact Directory: {artifact_dir}")
    click.echo("=" * 50)
    click.echo()

    app = create_viewer_app(default_artifact_dir=artifact_dir)
    app.launch(
        server_port=port,
        share=share,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
