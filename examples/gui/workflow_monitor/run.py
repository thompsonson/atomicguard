#!/usr/bin/env python3
"""
AtomicGuard GUI - Live Workflow Monitoring.

A web-based GUI for monitoring AtomicGuard workflow execution in real-time.

Usage:
    python -m examples.gui.run
    python -m examples.gui.run --port 8080
    python -m examples.gui.run --share  # Create public sharing link
    python -m examples.gui.run --workflow examples/tdd_import_guard/workflow.json

Features:
    - Live workflow visualization (DAG)
    - Step-by-step execution monitoring
    - Real-time log streaming
    - Artifact browsing with provenance
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option(
    "--port",
    default=7860,
    type=int,
    help="Port to run the Gradio server on (default: 7860)",
)
@click.option(
    "--share",
    is_flag=True,
    help="Create a public sharing link via Gradio",
)
@click.option(
    "--host",
    default="http://localhost:11434",
    help="Default Ollama API URL (default: http://localhost:11434)",
)
@click.option(
    "--model",
    default="qwen2.5-coder:14b",
    help="Default model to use (default: qwen2.5-coder:14b)",
)
@click.option(
    "--workflow",
    default=None,
    type=click.Path(exists=True),
    help="Default path to workflow.json",
)
@click.option(
    "--prompts",
    default=None,
    type=click.Path(exists=True),
    help="Default path to prompts.json",
)
def main(
    port: int,
    share: bool,
    host: str,
    model: str,
    workflow: str | None,
    prompts: str | None,
) -> None:
    """Launch the AtomicGuard Workflow Monitor GUI."""
    from .app import create_app

    # Resolve default paths if not provided
    script_dir = Path(__file__).parent

    if workflow is None:
        default_workflow = script_dir / "workflow.json"
        if not default_workflow.exists():
            # Fall back to tdd_import_guard example
            fallback = script_dir.parent / "tdd_import_guard" / "workflow.json"
            workflow = str(fallback) if fallback.exists() else str(default_workflow)
        else:
            workflow = str(default_workflow)

    if prompts is None:
        default_prompts = script_dir / "prompts.json"
        if not default_prompts.exists():
            # Fall back to tdd_import_guard example
            fallback = script_dir.parent / "tdd_import_guard" / "prompts.json"
            prompts = str(fallback) if fallback.exists() else str(default_prompts)
        else:
            prompts = str(default_prompts)

    click.echo("=" * 60)
    click.echo("AtomicGuard Workflow Monitor")
    click.echo("=" * 60)
    click.echo(f"Port: {port}")
    click.echo(f"Share: {share}")
    click.echo(f"Default Host: {host}")
    click.echo(f"Default Model: {model}")
    click.echo(f"Default Workflow: {workflow}")
    click.echo(f"Default Prompts: {prompts}")
    click.echo("=" * 60)
    click.echo()

    # Create and launch the app
    app = create_app(
        default_host=host,
        default_model=model,
        default_workflow=workflow,
        default_prompts=prompts,
    )

    import gradio as gr

    app.launch(
        server_port=port,
        share=share,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
