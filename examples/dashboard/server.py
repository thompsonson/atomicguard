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
from .routes import register_routes

_PACKAGE_DIR = Path(__file__).parent


def create_app(
    artifact_dir: Path,
    workflows_dir: Path | None = None,
    prompts_path: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="AtomicGuard Dashboard", docs_url=None, redoc_url=None)

    # Data layer
    app.state.discovery = ExperimentDiscovery(artifact_dir)
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
    "--artifact-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Root directory containing instance/arm DAGs.",
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
    artifact_dir: Path,
    workflows_dir: Path | None,
    prompts: Path | None,
    host: str,
    port: int,
) -> None:
    """Start the AtomicGuard dashboard server."""
    app = create_app(artifact_dir, workflows_dir, prompts)
    uvicorn.run(app, host=host, port=port)
