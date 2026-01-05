# Artifact Viewer

A standalone Gradio application for browsing artifacts from AtomicGuard workflow executions.

## Features

- **Browse artifacts** by workflow and action pair
- **View content** with syntax highlighting
- **Inspect metadata** including workflow ID, action pair, attempt number, status
- **Trace provenance** - see the full attempt history with guard feedback
- **Compare artifacts** side-by-side
- **Visualize flow** with Mermaid diagrams showing artifact relationships

## Installation

This example requires Gradio:

```bash
pip install gradio>=4.0.0
```

Or install from the requirements file:

```bash
pip install -r examples/artifact_viewer/requirements.txt
```

## Usage

### Run from command line

```bash
# Basic usage - opens UI to select artifact directory
python -m examples.artifact_viewer.app

# Pre-load a specific artifact directory
python -m examples.artifact_viewer.app --artifact-dir ./output/artifacts

# Use a different port
python -m examples.artifact_viewer.app --port 7862

# Create a public sharing link
python -m examples.artifact_viewer.app --share
```

### Use programmatically

```python
from examples.artifact_viewer import create_viewer_app

# Create the app with a default directory
app = create_viewer_app(default_artifact_dir="./my_artifacts")

# Launch with custom settings
app.launch(server_port=8080, share=False)
```

## Artifact Directory Structure

The viewer expects artifacts stored by `FilesystemArtifactDAG`:

```
artifacts/
├── index.json           # Artifact index with metadata
└── objects/
    ├── ab/
    │   └── ab12cd34...json  # Individual artifact files
    └── cd/
        └── cd56ef78...json
```

This structure is automatically created when running AtomicGuard workflows with `FilesystemArtifactDAG`.

## Example Workflow

1. Run any AtomicGuard example that produces artifacts:

   ```bash
   python -m examples.checkpoint_tdd.demo run
   ```

2. Launch the artifact viewer:

   ```bash
   python -m examples.artifact_viewer.app --artifact-dir ./examples/checkpoint_tdd/output/artifact_dag
   ```

3. Browse artifacts in the web UI at <http://localhost:7861>

## UI Components

### Artifact Browser

- **Workflow Filter**: Select a specific workflow or view all
- **Artifact Tree**: Hierarchical view grouped by action pair
- **Artifact Flow**: Mermaid diagram showing dependencies

### Detail Tabs

- **Content**: The generated code/content with syntax highlighting
- **Metadata**: Artifact ID, workflow ID, status, timestamps
- **Provenance**: History of all attempts with guard feedback
- **Context**: Generation context including specification and constraints
- **Compare**: Side-by-side comparison of two artifacts
