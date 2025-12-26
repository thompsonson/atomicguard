"""Main Gradio application for AtomicGuard workflow monitoring."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import gradio as gr

from atomicguard import FilesystemArtifactDAG
from examples.base import load_prompts, load_workflow_config

from .adapters.log_handler import cleanup_gui_logging, setup_gui_logging
from .adapters.observable_workflow import ObservableWorkflow
from .components.artifact_browser import create_artifact_browser
from .components.config_panel import create_config_panel
from .components.execution_panel import create_execution_panel
from .components.log_viewer import create_log_viewer
from .components.workflow_viz import create_workflow_viz
from .state.manager import StateManager


def create_app(
    default_host: str = "http://localhost:11434",
    default_model: str = "qwen2.5-coder:14b",
    default_workflow: str | None = None,
    default_prompts: str | None = None,
) -> gr.Blocks:
    """
    Create the Gradio application for workflow monitoring.

    Args:
        default_host: Default Ollama host URL
        default_model: Default model name
        default_workflow: Default path to workflow.json
        default_prompts: Default path to prompts.json

    Returns:
        Configured Gradio Blocks application
    """
    # Shared state
    state_manager = StateManager()
    artifact_dag: FilesystemArtifactDAG | None = None
    execution_thread: threading.Thread | None = None
    observable_workflow: ObservableWorkflow | None = None
    current_workflow_id: str | None = None

    with gr.Blocks(
        title="AtomicGuard Workflow Monitor",
    ) as app:
        gr.Markdown(
            """
            # AtomicGuard Workflow Monitor

            Real-time monitoring of AtomicGuard workflow execution.
            """
        )

        with gr.Tabs():
            # Tab 1: Configuration & Execution
            with gr.Tab("Execution"):
                with gr.Row():
                    # Left column: Configuration
                    with gr.Column(scale=1):
                        (
                            workflow_path,
                            prompts_path,
                            host_input,
                            model_input,
                            rmax_slider,
                            start_btn,
                            stop_btn,
                        ) = create_config_panel(
                            default_host=default_host,
                            default_model=default_model,
                            default_workflow=default_workflow,
                            default_prompts=default_prompts,
                        )

                    # Right column: Workflow visualization
                    with gr.Column(scale=2):
                        diagram, update_diagram = create_workflow_viz()

                # Steps and Logs side by side
                with gr.Row():
                    # Left column: Execution status panel
                    with gr.Column(scale=1):
                        (
                            step_cards,
                            status_text,
                            duration_text,
                            update_execution_panel,
                        ) = create_execution_panel()

                    # Right column: Logs
                    with gr.Column(scale=1):
                        level_filter, log_output, update_logs_fn = create_log_viewer()

            # Tab 2: Artifacts
            with gr.Tab("Artifacts"):
                (
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
                ) = create_artifact_browser()

                refresh_btn = gr.Button("Refresh Artifacts", size="sm")

        # ===== Event Handlers =====

        def start_workflow(
            workflow_path_val: str,
            prompts_path_val: str,
            host_val: str,
            model_val: str,
            rmax_val: int,
        ) -> tuple[str, str, str, str]:
            """Start workflow execution in background thread."""
            nonlocal \
                artifact_dag, \
                execution_thread, \
                observable_workflow, \
                current_workflow_id

            # Reset state
            state_manager.reset()
            current_workflow_id = (
                None  # Will be populated when first artifact is created
            )

            try:
                # Load configuration
                workflow_config = load_workflow_config(
                    Path(workflow_path_val),
                    required_fields=("name", "specification", "action_pairs"),
                )
                prompts_data = load_prompts(Path(prompts_path_val))

                # Override rmax from slider
                workflow_config["rmax"] = int(rmax_val)

                # Initialize steps in state manager
                state_manager.set_steps_from_config(
                    workflow_config.get("action_pairs", {}),
                    rmax=int(rmax_val),
                )

                # Create artifact directory (preserve historical artifacts)
                script_dir = Path(__file__).parent
                artifact_dir = script_dir / "output" / "artifacts"
                artifact_dir.mkdir(parents=True, exist_ok=True)

                # Create artifact DAG
                artifact_dag = FilesystemArtifactDAG(str(artifact_dir))

                # Setup logging with event emission
                logger, log_handler = setup_gui_logging(state_manager.update)

                # Create observable workflow
                observable_workflow = ObservableWorkflow(
                    workflow_config=workflow_config,
                    prompts=prompts_data,
                    artifact_dag=artifact_dag,
                    emit_callback=state_manager.update,
                    host=host_val,
                    model_override=model_val,
                    logger=logger,
                )

                # Execute in background thread
                def run_workflow() -> None:
                    try:
                        result, duration = observable_workflow.execute()
                        state_manager.set_result(result)
                    except Exception as e:
                        state_manager.set_error(str(e))
                    finally:
                        cleanup_gui_logging(logger, log_handler)

                execution_thread = threading.Thread(target=run_workflow, daemon=True)
                execution_thread.start()

                # Return initial state update
                initial_state = state_manager.get_snapshot()
                return (
                    update_diagram(initial_state),
                    *update_execution_panel(initial_state),
                )

            except Exception as e:
                state_manager.set_error(str(e))
                error_state = state_manager.get_snapshot()
                return (
                    update_diagram(error_state),
                    *update_execution_panel(error_state),
                )

        def stop_workflow() -> tuple[str, str, str, str]:
            """Request workflow stop."""
            nonlocal observable_workflow

            if observable_workflow:
                observable_workflow.request_stop()

            state = state_manager.get_snapshot()
            return (
                update_diagram(state),
                *update_execution_panel(state),
            )

        def poll_state() -> tuple[str, str, str, str, str, Any, Any, Any, Any, Any]:
            """Poll for state updates (called by timer)."""
            nonlocal current_workflow_id

            state = state_manager.get_snapshot()

            # Apply current log filter
            filter_level = state_manager.get_log_filter()
            level = filter_level if filter_level != "All" else None
            logs = "\n".join(state_manager.get_logs(level)[-100:]) or "No logs yet..."

            # Auto-refresh artifacts when workflow completes
            artifact_tree_update: Any = gr.update()  # No change by default
            artifact_id_map_update: Any = gr.update()  # No change by default
            compare_left_update: Any = gr.update()
            compare_right_update: Any = gr.update()
            diagram_update: Any = gr.update()

            if state_manager.should_refresh_artifacts() and artifact_dag is not None:
                # Get workflow_id from most recent artifact
                if current_workflow_id is None:
                    workflows = artifact_dag._index.get("workflows", {})
                    if workflows:
                        # Get the most recent workflow (by timestamp in created_at)
                        for wf_id, artifact_ids in workflows.items():
                            if artifact_ids:
                                current_workflow_id = wf_id
                                break

                (
                    artifact_tree_update,
                    artifact_id_map_update,
                    compare_left_update,
                    compare_right_update,
                ) = refresh_artifacts_fn(artifact_dag, current_workflow_id)
                diagram_update = update_diagram_fn(
                    artifact_dag, current_workflow_id, None
                )

            return (
                update_diagram(state),
                *update_execution_panel(state),
                logs,
                artifact_tree_update,
                artifact_id_map_update,
                compare_left_update,
                compare_right_update,
                diagram_update,
            )

        def refresh_artifacts_handler() -> tuple[Any, Any, Any, Any, str]:
            """Refresh artifact list (shows current workflow only)."""
            tree_update, id_map_update, left_update, right_update = (
                refresh_artifacts_fn(artifact_dag, current_workflow_id)
            )
            diagram_md = update_diagram_fn(artifact_dag, current_workflow_id, None)
            return tree_update, id_map_update, left_update, right_update, diagram_md

        def on_tree_select(
            evt: gr.SelectData,
            id_map: dict[int, str | None],
        ) -> tuple[str, dict, list, dict, str]:
            """Handle artifact tree row selection."""
            row_idx = evt.index[0] if isinstance(evt.index, list | tuple) else evt.index
            artifact_id = id_map.get(row_idx)

            # If header row (None) or invalid, return empty state
            if artifact_id is None:
                return (
                    "Select an artifact to view its content",
                    {},
                    [],
                    {},
                    update_diagram_fn(artifact_dag, current_workflow_id, None),
                )

            content, metadata, provenance, context, action_pair_id = load_artifact_fn(
                artifact_id, artifact_dag
            )
            # Update diagram with action_pair highlighting
            diagram_md = update_diagram_fn(
                artifact_dag, current_workflow_id, action_pair_id
            )
            return content, metadata, provenance, context, diagram_md

        def compare_handler(
            left_id: str | None, right_id: str | None
        ) -> tuple[str, str]:
            """Load artifacts for comparison."""
            left, right = compare_fn(left_id, right_id, artifact_dag)
            return left, right

        def filter_logs(level: str) -> str:
            """Filter logs by level and save preference."""
            state_manager.set_log_filter(level)
            logs = state_manager.get_logs(level if level != "All" else None)
            return "\n".join(logs[-100:]) if logs else "No logs yet..."

        # ===== Wire up events =====

        # Start/Stop buttons
        start_btn.click(
            fn=start_workflow,
            inputs=[workflow_path, prompts_path, host_input, model_input, rmax_slider],
            outputs=[diagram, step_cards, status_text, duration_text],
        )

        stop_btn.click(
            fn=stop_workflow,
            inputs=[],
            outputs=[diagram, step_cards, status_text, duration_text],
        )

        # Polling timer for live updates
        timer = gr.Timer(0.5)
        timer.tick(
            fn=poll_state,
            inputs=[],
            outputs=[
                diagram,
                step_cards,
                status_text,
                duration_text,
                log_output,
                artifact_tree,
                artifact_id_map,
                compare_left,
                compare_right,
                artifact_diagram,
            ],
        )

        # Log level filter
        level_filter.change(
            fn=filter_logs,
            inputs=[level_filter],
            outputs=[log_output],
        )

        # Artifact browser
        refresh_btn.click(
            fn=refresh_artifacts_handler,
            inputs=[],
            outputs=[
                artifact_tree,
                artifact_id_map,
                compare_left,
                compare_right,
                artifact_diagram,
            ],
        )

        artifact_tree.select(
            fn=on_tree_select,
            inputs=[artifact_id_map],
            outputs=[
                content_display,
                metadata_display,
                provenance_display,
                context_display,
                artifact_diagram,
            ],
        )

        # Comparison handlers
        compare_left.change(
            fn=compare_handler,
            inputs=[compare_left, compare_right],
            outputs=[compare_left_content, compare_right_content],
        )

        compare_right.change(
            fn=compare_handler,
            inputs=[compare_left, compare_right],
            outputs=[compare_left_content, compare_right_content],
        )

    return app  # type: ignore[no-any-return]
