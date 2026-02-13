# Architecture: AtomicGuard Live Dashboard

## Overview

A read-only, real-time development dashboard for browsing workflow
configurations, prompt templates, and live/completed execution runs
across multiple experiment arms and instances.

**Design principles:**
- Read-only — never mutates artifacts or configs
- Zero config — point at a directory, it discovers everything
- Real-time — watches filesystem for new artifacts during execution
- Self-contained — single `uv run` command, no build step


## 1. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Browser                                │
│                                                               │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────────────┐│
│  │Site Map  │  │ Config Viz   │  │   Live DAG Viewer        ││
│  │(nav tree)│  │ (Cytoscape)  │  │ (Cytoscape + WebSocket)  ││
│  └─────────┘  └──────────────┘  └──────────────────────────┘│
│                                                               │
│  All pages: server-rendered HTML + Cytoscape.js + vanilla JS  │
│  No build step. No React/Vue/npm.                             │
└───────────────────────┬──────────────────────────────────────┘
                        │  HTTP + WebSocket
┌───────────────────────┴──────────────────────────────────────┐
│                     FastAPI (uvicorn)                          │
│                                                               │
│  ┌──────────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │   Page Routes    │  │  REST API    │  │  WebSocket Hub │ │
│  │   (Jinja2 SSR)   │  │  /api/*      │  │  /ws/dag/...   │ │
│  └────────┬─────────┘  └──────┬───────┘  └───────┬────────┘ │
│           │                    │                   │          │
│  ┌────────┴────────────────────┴───────────────────┴────────┐│
│  │                    Data Layer (read-only)                 ││
│  │                                                           ││
│  │  ┌─────────────────┐  ┌────────────┐  ┌───────────────┐ ││
│  │  │ DAG Reader      │  │Config      │  │ File Watcher  │ ││
│  │  │ (Filesystem     │  │Loader      │  │ (watchfiles)  │ ││
│  │  │  ArtifactDAG)   │  │            │  │               │ ││
│  │  └────────┬────────┘  └─────┬──────┘  └───────┬───────┘ ││
│  └───────────┼─────────────────┼──────────────────┼─────────┘│
└──────────────┼─────────────────┼──────────────────┼──────────┘
               │                 │                  │
       ┌───────┴─────────────────┴──────────────────┴──────┐
       │                    Filesystem                      │
       │                                                    │
       │  <experiment_output>/                              │
       │  ├── artifact_dags/                                │
       │  │   ├── <instance_id>/                            │
       │  │   │   ├── <arm>/                                │
       │  │   │   │   ├── index.json  ◄── watched          │
       │  │   │   │   └── objects/                          │
       │  │   │   │       └── <prefix>/<artifact>.json      │
       │  │   │   └── <arm>/                                │
       │  │   └── <instance_id>/                            │
       │  │                                                 │
       │  <workflows_dir>/                                  │
       │  ├── 07_s1_decomposed.json                         │
       │  └── ...                                           │
       │                                                    │
       │  <prompts_dir>/                                    │
       │  └── prompts.json                                  │
       └───────────────────────────────────────────────────┘
```


## 2. Site Map

```
/
├── /                                    Dashboard: experiment overview
│   Lists all discovered experiment dirs with stats
│   (instance count, arm count, completion %)
│
├── /experiments/{exp}/                  Experiment detail
│   Lists all instances × arms as a grid/table
│   Columns: instance | arm | status | steps | artifacts
│
├── /experiments/{exp}/{arm}/{instance}/ Live DAG viewer
│   Reuses existing Cytoscape visualization
│   WebSocket push for real-time artifact updates
│   Run selector, sidebar with artifact details
│
├── /config/                             Workflow configs index
│   Lists all discovered workflow JSON files
│
├── /config/{variant}                    Workflow config viz
│   Static DAG of action pairs with requires edges
│   Sidebar: prompt details (role, task, constraints)
│   Escalation routes shown as dashed edges
│
└── /api/                                REST API (JSON)
    ├── GET  /api/experiments
    ├── GET  /api/experiments/{exp}/summary
    ├── GET  /api/experiments/{exp}/{arm}/{instance}/dag
    ├── GET  /api/config/{variant}
    └── WS   /ws/dag/{exp}/{arm}/{instance}
```


## 3. Component Detail

### 3.1 File Watcher

Uses `watchfiles` (already a transitive dep of uvicorn) to monitor
`index.json` files across all experiment artifact DAGs.

```python
# Watches all index.json files under the experiment root.
# On change: re-reads the index, diffs against cached version,
# identifies new artifact IDs, loads them, broadcasts via WebSocket.

class DAGWatcher:
    def __init__(self, root: Path):
        self.root = root
        self._subscribers: dict[str, set[WebSocket]] = {}  # path -> connections
        self._cache: dict[str, dict] = {}                   # path -> last index

    async def watch(self):
        """Background task: watch for index.json changes."""
        async for changes in awatch(self.root, watch_filter=index_json_filter):
            for change_type, path in changes:
                await self._handle_change(Path(path))

    async def _handle_change(self, index_path: Path):
        """Diff index, load new artifacts, broadcast."""
        dag_key = str(index_path.parent)
        new_index = json.loads(index_path.read_text())
        old_index = self._cache.get(dag_key, {})

        new_ids = set(new_index.get("artifacts", {})) - set(old_index.get("artifacts", {}))
        # ... load new artifacts, broadcast to subscribers
```

### 3.2 WebSocket Hub

Each DAG viewer page opens a WebSocket to `/ws/dag/{exp}/{arm}/{instance}`.
The server subscribes that connection to the corresponding `index.json` watcher.
New artifacts arrive as JSON messages:

```json
{
  "type": "artifact_added",
  "artifact": { ... serialized artifact ... },
  "node": { ... cytoscape node data ... },
  "edges": [ ... new edges ... ]
}
```

The client JS appends the node/edges to the live Cytoscape instance.

### 3.3 DAG Reader

Wraps `FilesystemArtifactDAG` in read-only mode. No `store()` calls.
Provides methods tailored to the dashboard:

```python
class DAGReader:
    def __init__(self, dag_dir: Path):
        self._dag = FilesystemArtifactDAG(dag_dir)

    def get_visualization_data(self) -> WorkflowVisualizationData:
        """Reuse extract_workflow_data() from html_exporter."""
        return extract_workflow_data(self._dag)

    def get_artifact_detail(self, artifact_id: str) -> dict:
        """Single artifact for sidebar."""
        return _serialize_artifact(self._dag.get_artifact(artifact_id))
```

### 3.4 Config Loader

Reads workflow JSON + prompts JSON and merges them:

```python
class ConfigLoader:
    def __init__(self, workflows_dir: Path, prompts_path: Path | None):
        self.workflows_dir = workflows_dir
        self.prompts = load_prompts(prompts_path) if prompts_path else {}

    def get_config(self, variant: str) -> dict:
        """Workflow config + merged prompt data per action pair."""
        config = json.loads((self.workflows_dir / f"{variant}.json").read_text())
        for ap_id, ap in config["action_pairs"].items():
            if ap_id in self.prompts:
                ap["prompt"] = self.prompts[ap_id]  # merge in
        return config
```

### 3.5 Experiment Discovery

Scans the artifact_dags directory to discover the hierarchy:

```
artifact_dags/
  <instance_id>/       ← instance (SWE-bench problem)
    <arm>/             ← arm (workflow variant, e.g. "02_singleshot")
      index.json       ← one DAG per (instance, arm) pair
      objects/
```

```python
class ExperimentDiscovery:
    def __init__(self, root: Path):
        self.root = root

    def list_instances(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    def list_arms(self, instance: str) -> list[str]:
        return sorted(d.name for d in (self.root / instance).iterdir() if d.is_dir())

    def get_dag_path(self, instance: str, arm: str) -> Path:
        return self.root / instance / arm
```


## 4. Where the Code Lives

```
examples/dashboard/                     ← NEW package
├── __init__.py
├── __main__.py                         ← `python -m examples.dashboard`
├── server.py                           ← FastAPI app factory
├── discovery.py                        ← ExperimentDiscovery
├── dag_reader.py                       ← Read-only DAG wrapper
├── config_loader.py                    ← Workflow + prompts merger
├── watcher.py                          ← DAGWatcher (watchfiles)
├── routes/
│   ├── __init__.py
│   ├── pages.py                        ← HTML page routes (Jinja2 SSR)
│   ├── api.py                          ← REST API routes (JSON)
│   └── websocket.py                    ← WebSocket endpoint
├── templates/                          ← Jinja2 HTML templates
│   ├── base.html                       ← Layout (header, nav, footer)
│   ├── dashboard.html                  ← / — experiment overview
│   ├── experiment.html                 ← /experiments/{exp}
│   ├── dag_viewer.html                 ← /experiments/{exp}/{arm}/{instance}
│   ├── config_index.html               ← /config/
│   └── config_detail.html              ← /config/{variant}
└── static/                             ← CSS, JS (no build step)
    ├── style.css                       ← Shared styles (reuse existing palette)
    ├── cytoscape-setup.js              ← Shared Cytoscape init + styles
    ├── dag-viewer.js                   ← WebSocket + live update logic
    └── config-viewer.js                ← Config DAG + prompt sidebar
```

**Why `examples/dashboard/` not `src/atomicguard/dashboard/`:**
- It's a **development tool**, not core library functionality
- Depends on FastAPI/uvicorn — shouldn't be a core dependency
- Follows the pattern of existing `examples/gui/` tools
- Can graduate to core if it proves essential


## 5. How to Run

```bash
# Minimal — point at an experiment output directory
uv run python -m examples.dashboard \
    --artifact-dir ./examples/swe_bench_pro/example_output/artifact_dags

# Full — with workflow configs and prompts for config viz
uv run python -m examples.dashboard \
    --artifact-dir ./examples/swe_bench_pro/example_output/artifact_dags \
    --workflows-dir ./examples/swe_bench_common/workflows \
    --prompts ./examples/swe_bench_ablation/prompts.json \
    --port 8000

# Opens at http://localhost:8000
```

**CLI options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--artifact-dir` | (required) | Root of `artifact_dags/` directory |
| `--workflows-dir` | None | Directory containing `*.json` workflow configs |
| `--prompts` | None | Path to `prompts.json` |
| `--port` | 8000 | Server port |
| `--host` | 127.0.0.1 | Bind address |
| `--no-watch` | False | Disable filesystem watcher (static mode) |


## 6. Dependencies

New optional dependency group in `pyproject.toml`:

```toml
[dependency-groups]
dashboard = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",   # includes watchfiles
    "jinja2>=3.0.0",
]
```

`watchfiles` is a transitive dependency of `uvicorn[standard]` — no
separate install needed.


## 7. Data Flow: Real-Time Update

```
Workflow Engine                    Dashboard Server                Browser
     │                                  │                            │
     │  store(artifact)                 │                            │
     │──► index.json updated            │                            │
     │                                  │                            │
     │                     watchfiles detects change                 │
     │                                  │                            │
     │                     read new artifact from objects/           │
     │                                  │                            │
     │                     build Cytoscape node + edges              │
     │                                  │                            │
     │                     broadcast via WebSocket ──────────────►   │
     │                                  │               cy.add(...)  │
     │                                  │               re-layout    │
     │                                  │                            │
```


## 8. Reuse from Existing Codebase

| What | From | How |
|------|------|-----|
| Artifact serialization | `html_exporter._serialize_artifact()` | Import directly |
| DAG → Cytoscape data | `html_exporter.extract_workflow_data()` | Import directly |
| Cytoscape styles | `templates/workflow.html` | Extract to shared JS |
| Run computation | `html_exporter` (run selector logic) | Import directly |
| Filesystem DAG | `FilesystemArtifactDAG` | Read-only wrapper |
| Config loading | `examples/base/load_workflow_config` | Import directly |
| Prompt loading | `examples/base/load_prompts` | Import directly |
| Visual style | Existing CSS palette | Copy to `static/style.css` |


## 9. Relationship to Existing GUI Tools

| Tool | Purpose | Tech | Status |
|------|---------|------|--------|
| `examples/gui/artifact_viewer/` | Post-hoc artifact browsing | Gradio | Exists |
| `examples/gui/workflow_monitor/` | Live execution + control | Gradio | Exists |
| **`examples/dashboard/`** | **Read-only multi-experiment browser** | **FastAPI** | **NEW** |

The dashboard is complementary: it's read-only (no execution control),
multi-experiment (browse across arms/instances), and lightweight (no
Gradio dependency, pure HTML+JS).


## 10. Implementation Order

1. **Scaffold** — FastAPI app, CLI entry point, base template
2. **Experiment discovery** — scan dirs, list instances/arms
3. **DAG viewer page** — reuse `extract_workflow_data()`, serve as HTML
4. **Config viewer page** — reuse/build `export_workflow_config_html()` logic
5. **WebSocket watcher** — live updates for running workflows
6. **Polish** — responsive layout, error handling, loading states
