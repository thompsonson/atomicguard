"""Configuration panel component for workflow setup."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    pass


def create_config_panel(
    default_host: str = "http://localhost:11434",
    default_model: str = "qwen2.5-coder:14b",
    default_workflow: str | None = None,
    default_prompts: str | None = None,
) -> tuple[
    gr.Textbox,  # workflow_path
    gr.Textbox,  # prompts_path
    gr.Textbox,  # host
    gr.Textbox,  # model
    gr.Slider,  # rmax
    gr.Button,  # start_btn
    gr.Button,  # stop_btn
]:
    """
    Create the configuration panel for workflow setup.

    Args:
        default_host: Default Ollama host URL
        default_model: Default model name
        default_workflow: Default path to workflow.json
        default_prompts: Default path to prompts.json

    Returns:
        Tuple of Gradio components for configuration
    """
    # Find default paths if not provided
    script_dir = Path(__file__).parent.parent
    if default_workflow is None:
        default_workflow = str(script_dir / "workflow.json")
    if default_prompts is None:
        default_prompts = str(script_dir / "prompts.json")

    with gr.Column():
        gr.Markdown("## Configuration")

        workflow_path = gr.Textbox(
            label="Workflow Config",
            value=default_workflow,
            placeholder="Path to workflow.json",
            info="Path to the workflow configuration file",
        )

        prompts_path = gr.Textbox(
            label="Prompts Config",
            value=default_prompts,
            placeholder="Path to prompts.json",
            info="Path to the prompts configuration file",
        )

        with gr.Row():
            host = gr.Textbox(
                label="Ollama Host",
                value=default_host,
                placeholder="http://localhost:11434",
                scale=2,
            )
            model = gr.Textbox(
                label="Model",
                value=default_model,
                placeholder="qwen2.5-coder:14b",
                scale=2,
            )
            rmax = gr.Slider(
                label="Max Retries",
                minimum=1,
                maximum=10,
                step=1,
                value=3,
                scale=1,
            )

        with gr.Row():
            start_btn = gr.Button(
                "Start Workflow",
                variant="primary",
                scale=2,
            )
            stop_btn = gr.Button(
                "Stop",
                variant="stop",
                scale=1,
            )

    return workflow_path, prompts_path, host, model, rmax, start_btn, stop_btn
