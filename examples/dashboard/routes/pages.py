"""SSR page routes for the dashboard."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


def _ctx(request: Request) -> dict[str, Any]:
    """Base template context with shared state."""
    app = request.app
    return {
        "request": request,
        "discovery": app.state.discovery,
        "config_loader": app.state.config_loader,
    }


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Home page: experiment overview stats + config link."""
    ctx = _ctx(request)
    summary = ctx["discovery"].get_summary()
    config_loader = ctx["config_loader"]
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "dashboard.html",
        {
            **ctx,
            "summary": summary,
            "has_configs": bool(config_loader.list_variants()),
        },
    )


@router.get("/experiment/", response_class=HTMLResponse)
async def experiment(request: Request) -> HTMLResponse:
    """Instance x arm grid table with status badges."""
    ctx = _ctx(request)
    discovery = ctx["discovery"]
    instances = discovery.get_instance_infos()
    arm_names = discovery.get_summary().arm_names
    templates = request.app.state.templates

    # Build status grid: for each (instance, arm) check index
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
                    if has_accepted and not has_rejected:
                        status = "success"
                    elif has_rejected:
                        status = "mixed"
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

    return templates.TemplateResponse(
        "experiment.html",
        {**ctx, "grid_data": grid_data, "arm_names": arm_names},
    )


@router.get("/dag/{instance}/{arm}", response_class=HTMLResponse)
async def dag_viewer(request: Request, instance: str, arm: str) -> HTMLResponse:
    """DAG viewer page for a specific instance/arm."""
    from examples.dashboard.dag_reader import DAGReader

    discovery = request.app.state.discovery
    dag_path = discovery.get_dag_path(instance, arm)
    if not (dag_path / "index.json").exists():
        raise HTTPException(status_code=404, detail="DAG not found")

    reader = DAGReader(dag_path)
    data = reader.get_visualization_data()
    templates = request.app.state.templates

    from examples.dashboard.discovery import parse_short_name

    return templates.TemplateResponse(
        "dag_viewer.html",
        {
            **_ctx(request),
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
