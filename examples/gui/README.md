# GUI Examples

Gradio-based visualization tools for AtomicGuard workflows.

## Tools

| Directory | Purpose | Use When |
|-----------|---------|----------|
| [artifact_viewer/](artifact_viewer/) | Browse existing artifacts | Inspecting completed workflow outputs |
| [workflow_monitor/](workflow_monitor/) | Live workflow execution | Running and observing workflows in real-time |

## Artifact Viewer

A standalone browser for exploring artifacts from completed workflow executions.

**Features:**

- Browse artifacts by workflow and action pair
- View content, metadata, and provenance
- Compare artifacts side-by-side
- Mermaid diagrams of artifact flow

```bash
python -m examples.gui.artifact_viewer.app --artifact-dir ./output/artifacts
```

## Workflow Monitor

A full-featured UI for configuring, running, and monitoring workflows.

**Features:**

- Real-time workflow DAG visualization
- Live execution status updates
- Log streaming with level filtering
- Configuration panel for workflow setup
- Integrated artifact browser

```bash
python -m examples.gui.workflow_monitor.run --port 7860
```

## Prerequisites

Both tools require Gradio:

```bash
pip install gradio>=4.0.0
```

Or install from the artifact_viewer requirements:

```bash
pip install -r examples/gui/artifact_viewer/requirements.txt
```
