"""FastAPI app factory and CLI entry point."""

from __future__ import annotations

from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

from .config_loader import ConfigLoader
from .discovery import ExperimentDiscovery
from .experiment_locator import ExperimentLocator
from .routes import register_routes

_PACKAGE_DIR = Path(__file__).parent


def create_app(
    artifact_dir: Path | None = None,
    workflows_dir: Path | None = None,
    prompts_path: Path | None = None,
    output_dir: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="AtomicGuard Dashboard", docs_url=None, redoc_url=None)

    # Build experiment registry
    experiments: dict[str, object] = {}  # slug -> ExperimentEntry
    discoveries: dict[str, ExperimentDiscovery] = {}  # slug -> discovery

    if output_dir and output_dir.is_dir():
        locator = ExperimentLocator(output_dir)
        for entry in locator.discover():
            experiments[entry.slug] = entry
            discoveries[entry.slug] = ExperimentDiscovery(entry.artifact_dags_path)
    elif artifact_dir:
        # Legacy single-experiment fallback
        from .experiment_locator import ExperimentEntry

        entry = ExperimentEntry(
            slug="default",
            display_name="Default Experiment",
            artifact_dags_path=artifact_dir,
        )
        # Derive counts from filesystem
        locator = ExperimentLocator(Path("."))
        locator._count_from_filesystem(entry)
        experiments["default"] = entry
        discoveries["default"] = ExperimentDiscovery(artifact_dir)

    app.state.experiments = experiments
    app.state.discoveries = discoveries
    app.state.output_dir = output_dir
    app.state.config_loader = ConfigLoader(workflows_dir, prompts_path)

    # Jinja2 templates
    template_dir = _PACKAGE_DIR / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    app.state.templates = _TemplateRenderer(env)

    # Static files
    static_dir = _PACKAGE_DIR / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routes
    register_routes(app)

    return app


class _TemplateRenderer:
    """Thin wrapper to match Starlette's TemplateResponse interface."""

    def __init__(self, env: Environment) -> None:
        self._env = env

    def TemplateResponse(  # noqa: N802 — matches Starlette convention
        self,
        name: str,
        context: dict,
        status_code: int = 200,
    ) -> HTMLResponse:
        template = self._env.get_template(name)
        html = template.render(**context)
        return HTMLResponse(content=html, status_code=status_code)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Root output directory containing experiment runs.",
)
@click.option(
    "--artifact-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="(Legacy) Single artifact_dags directory.",
)
@click.option(
    "--workflows-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory with workflow *.json configs.",
)
@click.option(
    "--prompts",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to prompts.json.",
)
@click.option("--host", default="0.0.0.0", help="Bind address.")
@click.option("--port", default=8000, type=int, help="Port number.")
def main(
    output_dir: Path | None,
    artifact_dir: Path | None,
    workflows_dir: Path | None,
    prompts: Path | None,
    host: str,
    port: int,
) -> None:
    """Start the AtomicGuard dashboard server."""
    # Auto-detect ./output if neither flag given
    if output_dir is None and artifact_dir is None:
        default_output = Path("./output")
        if default_output.is_dir():
            output_dir = default_output

    if output_dir is None and artifact_dir is None:
        raise click.UsageError("Provide --output-dir or --artifact-dir.")

    app = create_app(artifact_dir, workflows_dir, prompts, output_dir)
    uvicorn.run(app, host=host, port=port)
