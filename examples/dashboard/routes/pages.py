"""SSR page routes for the dashboard."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from ..discovery import ExperimentDiscovery
from ..experiment_locator import ExperimentLocator

router = APIRouter()


def _ctx(request: Request, slug: str | None = None) -> dict[str, Any]:
    """Base template context with shared state."""
    app = request.app
    ctx: dict[str, Any] = {
        "request": request,
        "config_loader": app.state.config_loader,
    }
    if slug:
        ctx["slug"] = slug
        entry = app.state.experiments.get(slug)
        if entry:
            ctx["experiment_name"] = entry.display_name
    return ctx


def _rescan_experiments(request: Request) -> None:
    """Merge newly discovered experiments into app state."""
    output_dir = getattr(request.app.state, "output_dir", None)
    if not output_dir or not output_dir.is_dir():
        return
    locator = ExperimentLocator(output_dir)
    for entry in locator.discover():
        if entry.slug not in request.app.state.experiments:
            request.app.state.experiments[entry.slug] = entry
            request.app.state.discoveries[entry.slug] = ExperimentDiscovery(
                entry.artifact_dags_path
            )


def _get_discovery(request: Request, slug: str):
    """Return the ExperimentDiscovery for *slug* or raise 404."""
    discovery = request.app.state.discoveries.get(slug)
    if discovery is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{slug}' not found")
    return discovery


def _get_experiment(request: Request, slug: str):
    """Return the ExperimentEntry for *slug* or raise 404."""
    entry = request.app.state.experiments.get(slug)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{slug}' not found")
    return entry


# -- Experiment list (homepage) ------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def experiment_list(request: Request) -> HTMLResponse:
    """Home page: list all discovered experiments."""
    _rescan_experiments(request)
    experiments = sorted(
        request.app.state.experiments.values(),
        key=lambda e: (e.modified_at is not None, e.modified_at),
        reverse=True,
    )
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "experiment_list.html",
        {**_ctx(request), "experiments": experiments},
    )


# -- Experiment detail (merged stats + grid) -----------------------------------


@router.get("/experiments/{slug}/", response_class=HTMLResponse)
async def experiment_detail(request: Request, slug: str) -> HTMLResponse:
    """Stats cards + instance x arm grid for a single experiment."""
    entry = _get_experiment(request, slug)
    discovery = _get_discovery(request, slug)
    instances = discovery.get_instance_infos()
    summary = discovery.get_summary()
    arm_names = summary.arm_names
    templates = request.app.state.templates

    # Build status grid
    grid_data = []
    for inst in instances:
        row: dict[str, Any] = {
            "instance_id": inst.instance_id,
            "short_name": inst.short_name,
            "arms": {},
        }
        for arm in arm_names:
            dag_path = discovery.get_dag_path(inst.instance_id, arm)
            index_path = dag_path / "index.json"
            if arm in inst.arms and index_path.exists():
                try:
                    idx = json.loads(index_path.read_text())
                    artifact_count = len(idx.get("artifacts", {}))
                    statuses = [
                        a.get("status") for a in idx.get("artifacts", {}).values()
                    ]
                    has_accepted = any(s == "accepted" for s in statuses)
                    has_rejected = any(s == "rejected" for s in statuses)
                    if has_accepted:
                        status = "mixed" if has_rejected else "success"
                    elif has_rejected:
                        status = "failed"
                    else:
                        status = "pending"
                    row["arms"][arm] = {
                        "status": status,
                        "artifact_count": artifact_count,
                    }
                except (json.JSONDecodeError, OSError):
                    row["arms"][arm] = {"status": "error", "artifact_count": 0}
            else:
                row["arms"][arm] = None
        grid_data.append(row)

    # Render NOTES.md if present
    notes_html = None
    if entry.notes_path and entry.notes_path.exists():
        import markdown

        notes_html = markdown.markdown(
            entry.notes_path.read_text(),
            extensions=["fenced_code", "tables"],
        )

    return templates.TemplateResponse(
        "experiment.html",
        {
            **_ctx(request, slug),
            "summary": summary,
            "grid_data": grid_data,
            "arm_names": arm_names,
            "notes_html": notes_html,
        },
    )


# -- DAG viewer ----------------------------------------------------------------


@router.get("/experiments/{slug}/{arm}/{instance}/", response_class=HTMLResponse)
async def dag_viewer(
    request: Request, slug: str, arm: str, instance: str
) -> HTMLResponse:
    """DAG viewer page for a specific instance/arm within an experiment."""
    from examples.dashboard.dag_reader import DAGReader
    from examples.dashboard.discovery import parse_short_name

    discovery = _get_discovery(request, slug)
    dag_path = discovery.get_dag_path(instance, arm)
    if not (dag_path / "index.json").exists():
        raise HTTPException(status_code=404, detail="DAG not found")

    reader = DAGReader(dag_path)
    data = reader.get_visualization_data()
    templates = request.app.state.templates

    return templates.TemplateResponse(
        "dag_viewer.html",
        {
            **_ctx(request, slug),
            "instance": instance,
            "arm": arm,
            "short_name": parse_short_name(instance),
            "data": data,
            "nodes_json": json.dumps(data.nodes),
            "edges_json": json.dumps(data.edges),
            "artifacts_json": json.dumps(data.artifacts),
            "runs_json": json.dumps(data.runs),
        },
    )


# -- Config routes (unchanged) -------------------------------------------------


@router.get("/config/", response_class=HTMLResponse)
async def config_index(request: Request) -> HTMLResponse:
    """List of workflow config variants."""
    ctx = _ctx(request)
    config_loader = ctx["config_loader"]
    variants = config_loader.list_variants()
    templates = request.app.state.templates

    variant_infos = []
    for v in variants:
        cfg = config_loader.get_config(v)
        variant_infos.append(
            {
                "name": v,
                "display_name": cfg.get("name", v) if cfg else v,
                "description": cfg.get("description", "") if cfg else "",
                "step_count": len(cfg.get("action_pairs", {})) if cfg else 0,
            }
        )

    return templates.TemplateResponse(
        "config_index.html",
        {**ctx, "variants": variant_infos},
    )


@router.get("/config/{variant}", response_class=HTMLResponse)
async def config_detail(request: Request, variant: str) -> HTMLResponse:
    """Config topology viewer for a workflow variant."""
    config_loader = request.app.state.config_loader
    result = config_loader.get_config_graph(variant)
    if result is None:
        raise HTTPException(status_code=404, detail="Config not found")

    nodes, edges, prompt_data = result
    config = config_loader.get_config(variant)
    templates = request.app.state.templates

    return templates.TemplateResponse(
        "config_detail.html",
        {
            **_ctx(request),
            "variant": variant,
            "config": config,
            "nodes_json": json.dumps(nodes),
            "edges_json": json.dumps(edges),
            "prompts_json": json.dumps(prompt_data),
        },
    )
