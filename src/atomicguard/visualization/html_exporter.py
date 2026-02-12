"""
HTML Exporter for Workflow DAG Visualization.

Generates self-contained HTML files with embedded Cytoscape.js
for interactive visualization of workflow artifacts and their relationships.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import ArtifactDAGInterface
    from atomicguard.domain.models import Artifact


@dataclass
class WorkflowVisualizationData:
    """Data structure for workflow visualization."""

    workflow_id: str
    status: str
    total_steps: int
    total_artifacts: int
    escalation_count: int
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    artifacts: dict[str, dict[str, Any]]  # artifact_id -> full artifact data


def extract_workflow_data(
    dag: ArtifactDAGInterface,
    workflow_id: str | None = None,
) -> WorkflowVisualizationData:
    """Extract visualization data from artifact DAG.

    Args:
        dag: The artifact DAG containing workflow artifacts.
        workflow_id: Specific workflow ID to extract. If None or "auto",
            uses the workflow_id from the first artifact.

    Returns:
        WorkflowVisualizationData ready for rendering.
    """
    all_artifacts = dag.get_all()

    if not all_artifacts:
        return WorkflowVisualizationData(
            workflow_id=workflow_id or "empty",
            status="EMPTY",
            total_steps=0,
            total_artifacts=0,
            escalation_count=0,
            nodes=[],
            edges=[],
            artifacts={},
        )

    # Auto-detect workflow_id from first artifact if needed
    if workflow_id is None or workflow_id == "auto":
        workflow_id = all_artifacts[0].workflow_id

    # Filter artifacts for this workflow
    artifacts = [a for a in all_artifacts if a.workflow_id == workflow_id]

    # Group artifacts by action_pair_id (step)
    steps: dict[str, list[Artifact]] = {}
    for artifact in artifacts:
        if artifact.action_pair_id not in steps:
            steps[artifact.action_pair_id] = []
        steps[artifact.action_pair_id].append(artifact)

    # Sort artifacts within each step by attempt number
    for step_id in steps:
        steps[step_id].sort(key=lambda a: a.attempt_number)

    # Build nodes for visualization — flat DAG, no compound grouping.
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    artifact_data: dict[str, dict[str, Any]] = {}
    artifact_node_lookup: dict[str, str] = {}  # full artifact_id -> node_id
    artifact_step_lookup: dict[str, str] = {}  # full artifact_id -> step_id

    # Track escalation events
    escalation_count = 0

    # Track final artifact per step (for dependency edge targets)
    step_final_artifact: dict[str, str] = {}  # step_id -> artifact_node_id

    for step_id, step_artifacts in steps.items():
        # Detect escalation
        if any(
            len(a.context.escalation_feedback) > 0 for a in step_artifacts
        ):
            escalation_count += 1

        # Create flat artifact nodes
        prev_artifact_node_id = None
        prev_esc_len = 0
        for artifact in step_artifacts:
            artifact_node_id = f"artifact_{artifact.artifact_id[:8]}"
            artifact_node_lookup[artifact.artifact_id] = artifact_node_id
            artifact_step_lookup[artifact.artifact_id] = step_id

            # Serialize artifact data for detail panel
            artifact_data[artifact.artifact_id] = _serialize_artifact(
                artifact
            )

            nodes.append(
                {
                    "data": {
                        "id": artifact_node_id,
                        "label": f"{step_id} #{artifact.attempt_number}",
                        "type": "artifact",
                        "status": artifact.status.value,
                        "artifact_id": artifact.artifact_id,
                        "attempt_number": artifact.attempt_number,
                        "step_id": step_id,
                        "has_escalation_feedback": len(
                            artifact.context.escalation_feedback
                        )
                        > 0,
                        "feedback_count": len(
                            artifact.context.feedback_history
                        ),
                    }
                }
            )

            # Detect escalation boundary for edge type
            curr_esc_len = len(artifact.context.escalation_feedback)
            is_escalation_retry = curr_esc_len > prev_esc_len

            # Retry / escalation-retry edges
            if artifact.previous_attempt_id:
                source_id = (
                    f"artifact_{artifact.previous_attempt_id[:8]}"
                )
                edges.append(
                    {
                        "data": {
                            "id": f"retry_{artifact.artifact_id[:8]}",
                            "source": source_id,
                            "target": artifact_node_id,
                            "type": "escalation_retry"
                            if is_escalation_retry
                            else "retry",
                        }
                    }
                )
            elif prev_artifact_node_id is not None:
                edges.append(
                    {
                        "data": {
                            "id": f"retry_{artifact.artifact_id[:8]}",
                            "source": prev_artifact_node_id,
                            "target": artifact_node_id,
                            "type": "escalation_retry"
                            if is_escalation_retry
                            else "retry",
                        }
                    }
                )

            prev_artifact_node_id = artifact_node_id
            prev_esc_len = curr_esc_len

        # Track final artifact for this step (accepted or latest)
        step_final_artifact[step_id] = f"artifact_{step_artifacts[-1].artifact_id[:8]}"
        for a in reversed(step_artifacts):
            if a.status.value == "accepted":
                step_final_artifact[step_id] = (
                    f"artifact_{a.artifact_id[:8]}"
                )
                break

    # Create dependency edges — one per (source_step, target_step) pair,
    # connecting the final artifact of the source step to the first
    # artifact in the target step that referenced it.
    seen_step_pairs: set[tuple[str, str]] = set()
    for step_id, step_artifacts in steps.items():
        for artifact in step_artifacts:
            for (
                _dep_action_pair_id,
                dep_artifact_id,
            ) in artifact.context.dependency_artifacts:
                source_step = artifact_step_lookup.get(dep_artifact_id)
                if source_step and source_step != step_id:
                    pair = (source_step, step_id)
                    if pair not in seen_step_pairs:
                        seen_step_pairs.add(pair)
                        source_node = step_final_artifact.get(source_step)
                        target_node = artifact_node_lookup.get(
                            step_artifacts[0].artifact_id
                        )
                        if source_node and target_node:
                            edges.append(
                                {
                                    "data": {
                                        "id": f"dep_{source_step}_{step_id}",
                                        "source": source_node,
                                        "target": target_node,
                                        "type": "dependency",
                                    }
                                }
                            )

    # Determine overall workflow status
    all_accepted = all(
        any(a.status.value == "accepted" for a in step_arts)
        for step_arts in steps.values()
    )
    workflow_status = "SUCCESS" if all_accepted else "IN_PROGRESS"

    return WorkflowVisualizationData(
        workflow_id=workflow_id,
        status=workflow_status,
        total_steps=len(steps),
        total_artifacts=len(artifacts),
        escalation_count=escalation_count,
        nodes=nodes,
        edges=edges,
        artifacts=artifact_data,
    )


def _serialize_artifact(artifact: Artifact) -> dict[str, Any]:
    """Serialize artifact for JSON embedding in HTML.

    Args:
        artifact: The artifact to serialize.

    Returns:
        Dict with all artifact fields serialized for JSON.
    """
    return {
        "artifact_id": artifact.artifact_id,
        "action_pair_id": artifact.action_pair_id,
        "workflow_id": artifact.workflow_id,
        "content": artifact.content,
        "status": artifact.status.value,
        "attempt_number": artifact.attempt_number,
        "created_at": artifact.created_at,
        "previous_attempt_id": artifact.previous_attempt_id,
        "guard_result": {
            "passed": artifact.guard_result.passed,
            "feedback": artifact.guard_result.feedback,
            "fatal": artifact.guard_result.fatal,
            "guard_name": artifact.guard_result.guard_name,
            "sub_results": [
                {
                    "guard_name": sr.guard_name,
                    "passed": sr.passed,
                    "feedback": sr.feedback,
                }
                for sr in artifact.guard_result.sub_results
            ],
        }
        if artifact.guard_result
        else None,
        "context": {
            "specification": artifact.context.specification,
            "constraints": artifact.context.constraints,
            "feedback_history": [
                {"artifact_id": fe.artifact_id, "feedback": fe.feedback}
                for fe in artifact.context.feedback_history
            ],
            "dependency_artifacts": [
                {"action_pair_id": dep[0], "artifact_id": dep[1]}
                for dep in artifact.context.dependency_artifacts
            ],
            "escalation_feedback": list(artifact.context.escalation_feedback),
        },
    }


def export_workflow_html(
    dag: ArtifactDAGInterface,
    workflow_id: str | None = None,
    output_path: str | Path = "workflow_visualization.html",
) -> Path:
    """Generate an interactive HTML visualization of a workflow DAG.

    Creates a self-contained HTML file with embedded Cytoscape.js that
    displays the workflow structure, artifact history, and escalation events.

    Args:
        dag: The artifact DAG containing workflow artifacts.
        workflow_id: Specific workflow ID to visualize. If None or "auto",
            uses the workflow_id from the first artifact.
        output_path: Path where the HTML file will be written.

    Returns:
        Path to the generated HTML file.

    Example:
        >>> from atomicguard.visualization import export_workflow_html
        >>> from atomicguard import InMemoryArtifactDAG
        >>>
        >>> dag = InMemoryArtifactDAG()
        >>> # ... run workflow ...
        >>> path = export_workflow_html(dag, workflow_id="auto", output_path="report.html")
        >>> print(f"Visualization saved to: {path}")
    """
    # Extract workflow data
    data = extract_workflow_data(dag, workflow_id)

    # Render template
    html_content = _render_template(data)

    # Write to file
    output = Path(output_path)
    output.write_text(html_content, encoding="utf-8")

    return output


def _render_template(data: WorkflowVisualizationData) -> str:
    """Render the HTML template with embedded data.

    Uses a simple string template to avoid jinja2 dependency for basic usage.
    Falls back to jinja2 if available for more complex templates.

    Args:
        data: The workflow visualization data.

    Returns:
        Complete HTML string ready to write to file.
    """
    # Try to use jinja2 if available
    jinja2_result = _try_render_with_jinja2(data)
    if jinja2_result is not None:
        return jinja2_result

    # Fallback to embedded template
    return _generate_embedded_html(data)


def _try_render_with_jinja2(data: WorkflowVisualizationData) -> str | None:
    """Attempt to render using jinja2 template if available.

    Args:
        data: The workflow visualization data.

    Returns:
        Rendered HTML string if jinja2 is available, None otherwise.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        return None

    template_dir = Path(__file__).parent / "templates"
    if not (template_dir / "workflow.html").exists():
        return None

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("workflow.html")
    return template.render(
        workflow_id=data.workflow_id,
        status=data.status,
        total_steps=data.total_steps,
        total_artifacts=data.total_artifacts,
        escalation_count=data.escalation_count,
        nodes_json=json.dumps(data.nodes, indent=2),
        edges_json=json.dumps(data.edges, indent=2),
        artifacts_json=json.dumps(data.artifacts, indent=2),
        generated_at=datetime.now().isoformat(),
    )


def _generate_embedded_html(data: WorkflowVisualizationData) -> str:
    """Generate HTML with embedded template (no jinja2 dependency).

    Args:
        data: The workflow visualization data.

    Returns:
        Complete HTML string.
    """
    nodes_json = json.dumps(data.nodes, indent=2)
    edges_json = json.dumps(data.edges, indent=2)
    artifacts_json = json.dumps(data.artifacts, indent=2)
    generated_at = datetime.now().isoformat()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workflow Visualization: {data.workflow_id[:8]}...</title>
    <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        header h1 {{
            margin: 0;
            font-size: 1.5rem;
            font-weight: 600;
        }}
        .header-info {{
            display: flex;
            gap: 2rem;
            margin-top: 0.5rem;
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        .header-info span {{
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}
        .status-badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .status-SUCCESS {{ background: #10b981; color: white; }}
        .status-IN_PROGRESS {{ background: #f59e0b; color: white; }}
        .status-FAILED {{ background: #ef4444; color: white; }}
        .status-EMPTY {{ background: #6b7280; color: white; }}
        main {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        #cy {{
            flex: 1;
            background: white;
            border-right: 1px solid #e5e7eb;
        }}
        .sidebar {{
            width: 450px;
            background: white;
            overflow-y: auto;
            padding: 1rem;
        }}
        .sidebar h2 {{
            margin: 0 0 1rem 0;
            font-size: 1.1rem;
            color: #374151;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 0.5rem;
        }}
        .no-selection {{
            color: #9ca3af;
            font-style: italic;
            text-align: center;
            padding: 2rem;
        }}
        .artifact-detail {{
            display: none;
        }}
        .artifact-detail.active {{
            display: block;
        }}
        .detail-section {{
            margin-bottom: 1.5rem;
        }}
        .detail-section h3 {{
            font-size: 0.9rem;
            color: #6b7280;
            margin: 0 0 0.5rem 0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .collapsible {{
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
        }}
        .collapsible-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: #f9fafb;
            cursor: pointer;
            user-select: none;
        }}
        .collapsible-header:hover {{
            background: #f3f4f6;
        }}
        .collapsible-header h4 {{
            margin: 0;
            font-size: 0.9rem;
            font-weight: 600;
            color: #374151;
        }}
        .collapsible-header .badge {{
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: #e5e7eb;
            color: #4b5563;
        }}
        .collapsible-header .toggle {{
            font-size: 1rem;
            color: #9ca3af;
            transition: transform 0.2s;
        }}
        .collapsible.expanded .toggle {{
            transform: rotate(180deg);
        }}
        .collapsible-content {{
            display: none;
            padding: 1rem;
            border-top: 1px solid #e5e7eb;
            background: white;
        }}
        .collapsible.expanded .collapsible-content {{
            display: block;
        }}
        .code-block {{
            background: #1f2937;
            color: #f9fafb;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            max-height: 300px;
            overflow-y: auto;
        }}
        .code-block code {{
            background: transparent;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }}
        .meta-item {{
            background: #f9fafb;
            padding: 0.75rem;
            border-radius: 6px;
        }}
        .meta-item label {{
            display: block;
            font-size: 0.75rem;
            color: #6b7280;
            margin-bottom: 0.25rem;
        }}
        .meta-item span {{
            font-size: 0.9rem;
            color: #111827;
            font-weight: 500;
        }}
        .guard-result {{
            padding: 0.75rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
        }}
        .guard-result.passed {{
            background: #d1fae5;
            border: 1px solid #10b981;
        }}
        .guard-result.failed {{
            background: #fee2e2;
            border: 1px solid #ef4444;
        }}
        .guard-result .guard-name {{
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        .guard-result .guard-feedback {{
            font-size: 0.85rem;
            color: #4b5563;
            white-space: pre-wrap;
        }}
        .feedback-entry {{
            background: #fef3c7;
            border: 1px solid #f59e0b;
            padding: 0.75rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
        }}
        .feedback-entry .entry-number {{
            font-size: 0.75rem;
            color: #92400e;
            margin-bottom: 0.25rem;
        }}
        .feedback-entry .entry-content {{
            font-size: 0.85rem;
            white-space: pre-wrap;
        }}
        .escalation-feedback {{
            background: #fce7f3;
            border: 1px solid #ec4899;
            padding: 0.75rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
        }}
        .escalation-feedback .label {{
            font-size: 0.75rem;
            color: #9d174d;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        footer {{
            background: #1f2937;
            color: #9ca3af;
            padding: 0.5rem 2rem;
            font-size: 0.8rem;
            text-align: center;
        }}
        .legend {{
            display: flex;
            gap: 1.5rem;
            padding: 1rem;
            background: #f9fafb;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}
        .legend-color.accepted {{ background: #10b981; }}
        .legend-color.rejected {{ background: #ef4444; }}
        .legend-color.pending {{ background: #f59e0b; }}
        .legend-color.superseded {{ background: #6b7280; }}
        .legend-color.escalation {{ background: #ec4899; border: 2px dashed #ec4899; }}
        .legend-line {{
            width: 24px;
            height: 2px;
            position: relative;
        }}
        .legend-line.dependency {{ background: #6366f1; }}
        .legend-line.retry {{ background: #9ca3af; border-top: 2px dashed #9ca3af; height: 0; }}
        .legend-line.escalation-retry {{ background: #ec4899; border-top: 2px dashed #ec4899; height: 0; }}
        @media (max-width: 768px) {{
            main {{
                flex-direction: column;
            }}
            #cy {{
                min-height: 300px;
                border-right: none;
                border-bottom: 1px solid #e5e7eb;
            }}
            .sidebar {{
                width: 100%;
            }}
            .header-info {{
                flex-wrap: wrap;
                gap: 0.5rem;
            }}
            .legend {{
                flex-wrap: wrap;
                gap: 0.75rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Workflow Visualization</h1>
            <div class="header-info">
                <span><strong>ID:</strong> {data.workflow_id[:8]}...</span>
                <span class="status-badge status-{data.status}">{data.status}</span>
                <span><strong>Steps:</strong> {data.total_steps}</span>
                <span><strong>Artifacts:</strong> {data.total_artifacts}</span>
                <span><strong>Escalations:</strong> {data.escalation_count}</span>
            </div>
        </header>
        <main>
            <div id="cy"></div>
            <div class="sidebar">
                <h2>Artifact Details</h2>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color accepted"></div>
                        <span>Accepted</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color rejected"></div>
                        <span>Rejected</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color pending"></div>
                        <span>Pending</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color escalation"></div>
                        <span>Escalation</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-line dependency"></div>
                        <span>Dependency</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-line retry"></div>
                        <span>Retry</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-line escalation-retry"></div>
                        <span>Escalation Retry</span>
                    </div>
                </div>
                <div id="detail-panel">
                    <p class="no-selection">Click on an artifact node to view details</p>
                </div>
            </div>
        </main>
        <footer>
            Generated by AtomicGuard Visualization | {generated_at}
        </footer>
    </div>

    <script>
        // Embedded data
        const nodes = {nodes_json};
        const edges = {edges_json};
        const artifacts = {artifacts_json};

        // Initialize Cytoscape
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: {{
                nodes: nodes,
                edges: edges
            }},
            style: [
                // Artifact nodes (flat — no compound grouping)
                {{
                    selector: 'node[type="artifact"]',
                    style: {{
                        'shape': 'round-rectangle',
                        'width': 'label',
                        'height': 30,
                        'padding': '12px',
                        'background-color': '#9ca3af',
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '11px',
                        'font-weight': 'bold',
                        'color': 'white'
                    }}
                }},
                {{
                    selector: 'node[type="artifact"][status="accepted"]',
                    style: {{
                        'background-color': '#10b981'
                    }}
                }},
                {{
                    selector: 'node[type="artifact"][status="rejected"]',
                    style: {{
                        'background-color': '#ef4444'
                    }}
                }},
                {{
                    selector: 'node[type="artifact"][status="pending"]',
                    style: {{
                        'background-color': '#f59e0b'
                    }}
                }},
                {{
                    selector: 'node[type="artifact"][status="superseded"]',
                    style: {{
                        'background-color': '#6b7280'
                    }}
                }},
                {{
                    selector: 'node[type="artifact"][has_escalation_feedback]',
                    style: {{
                        'border-width': 3,
                        'border-color': '#ec4899',
                        'border-style': 'dashed'
                    }}
                }},
                // Dependency edges
                {{
                    selector: 'edge[type="dependency"]',
                    style: {{
                        'width': 3,
                        'line-color': '#6366f1',
                        'target-arrow-color': '#6366f1',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }},
                // Retry edges
                {{
                    selector: 'edge[type="retry"]',
                    style: {{
                        'width': 2,
                        'line-color': '#9ca3af',
                        'line-style': 'dashed',
                        'target-arrow-color': '#9ca3af',
                        'target-arrow-shape': 'chevron',
                        'curve-style': 'bezier'
                    }}
                }},
                // Escalation retry edges
                {{
                    selector: 'edge[type="escalation_retry"]',
                    style: {{
                        'width': 2,
                        'line-color': '#ec4899',
                        'line-style': 'dashed',
                        'target-arrow-color': '#ec4899',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }},
                // Selection
                {{
                    selector: 'node:selected',
                    style: {{
                        'border-width': 4,
                        'border-color': '#3b82f6'
                    }}
                }}
            ],
            layout: {{
                name: 'dagre',
                rankDir: 'LR',
                nodeSep: 50,
                rankSep: 100,
                padding: 30
            }}
        }});

        // Handle node clicks
        cy.on('tap', 'node[type="artifact"]', function(evt) {{
            const node = evt.target;
            const artifactId = node.data('artifact_id');
            showArtifactDetail(artifactId);
        }});

        function showArtifactDetail(artifactId) {{
            const artifact = artifacts[artifactId];
            if (!artifact) return;

            const panel = document.getElementById('detail-panel');

            // Build feedback history HTML
            let feedbackHtml = '';
            if (artifact.context.feedback_history && artifact.context.feedback_history.length > 0) {{
                feedbackHtml = artifact.context.feedback_history.map((entry, i) => `
                    <div class="feedback-entry">
                        <div class="entry-number">Attempt ${{i + 1}}</div>
                        <div class="entry-content">${{escapeHtml(entry.feedback)}}</div>
                    </div>
                `).join('');
            }} else {{
                feedbackHtml = '<p style="color: #9ca3af; font-style: italic;">No previous feedback</p>';
            }}

            // Build escalation feedback HTML
            let escalationHtml = '';
            if (artifact.context.escalation_feedback && artifact.context.escalation_feedback.length > 0) {{
                escalationHtml = artifact.context.escalation_feedback.map((fb, i) => `
                    <div class="escalation-feedback">
                        <div class="label">Escalation ${{i + 1}}</div>
                        <div>${{escapeHtml(fb)}}</div>
                    </div>
                `).join('');
            }} else {{
                escalationHtml = '<p style="color: #9ca3af; font-style: italic;">No escalation feedback</p>';
            }}

            // Build guard result HTML
            let guardHtml = '';
            if (artifact.guard_result) {{
                const gr = artifact.guard_result;
                guardHtml = `
                    <div class="guard-result ${{gr.passed ? 'passed' : 'failed'}}">
                        <div class="guard-name">${{gr.guard_name || 'Guard'}}: ${{gr.passed ? 'PASSED' : 'FAILED'}}</div>
                        ${{gr.feedback ? `<div class="guard-feedback">${{escapeHtml(gr.feedback)}}</div>` : ''}}
                    </div>
                `;
                if (gr.sub_results && gr.sub_results.length > 0) {{
                    guardHtml += gr.sub_results.map(sr => `
                        <div class="guard-result ${{sr.passed ? 'passed' : 'failed'}}" style="margin-left: 1rem;">
                            <div class="guard-name">${{sr.guard_name}}: ${{sr.passed ? 'PASSED' : 'FAILED'}}</div>
                            ${{sr.feedback ? `<div class="guard-feedback">${{escapeHtml(sr.feedback)}}</div>` : ''}}
                        </div>
                    `).join('');
                }}
            }} else {{
                guardHtml = '<p style="color: #9ca3af; font-style: italic;">No guard result (pending)</p>';
            }}

            panel.innerHTML = `
                <div class="artifact-detail active">
                    <div class="detail-section">
                        <div class="meta-grid">
                            <div class="meta-item">
                                <label>Step</label>
                                <span>${{artifact.action_pair_id}}</span>
                            </div>
                            <div class="meta-item">
                                <label>Attempt</label>
                                <span>#${{artifact.attempt_number}}</span>
                            </div>
                            <div class="meta-item">
                                <label>Status</label>
                                <span style="text-transform: uppercase;">${{artifact.status}}</span>
                            </div>
                            <div class="meta-item">
                                <label>Created</label>
                                <span>${{new Date(artifact.created_at).toLocaleString()}}</span>
                            </div>
                        </div>
                    </div>

                    <div class="detail-section">
                        <div class="collapsible expanded">
                            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                                <h4>Content</h4>
                                <span class="toggle">&#9660;</span>
                            </div>
                            <div class="collapsible-content">
                                <pre class="code-block"><code class="language-python">${{escapeHtml(artifact.content)}}</code></pre>
                            </div>
                        </div>
                    </div>

                    <div class="detail-section">
                        <div class="collapsible expanded">
                            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                                <h4>Guard Result</h4>
                                <span class="toggle">&#9660;</span>
                            </div>
                            <div class="collapsible-content">
                                ${{guardHtml}}
                            </div>
                        </div>
                    </div>

                    <div class="detail-section">
                        <div class="collapsible">
                            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                                <h4>Feedback History</h4>
                                <span class="badge">${{artifact.context.feedback_history ? artifact.context.feedback_history.length : 0}} entries</span>
                                <span class="toggle">&#9660;</span>
                            </div>
                            <div class="collapsible-content">
                                ${{feedbackHtml}}
                            </div>
                        </div>
                    </div>

                    <div class="detail-section">
                        <div class="collapsible">
                            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                                <h4>Escalation Feedback</h4>
                                <span class="badge">${{artifact.context.escalation_feedback ? artifact.context.escalation_feedback.length : 0}} entries</span>
                                <span class="toggle">&#9660;</span>
                            </div>
                            <div class="collapsible-content">
                                ${{escalationHtml}}
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Highlight code
            document.querySelectorAll('pre code').forEach((block) => {{
                hljs.highlightElement(block);
            }});
        }}

        function toggleCollapsible(header) {{
            const collapsible = header.parentElement;
            collapsible.classList.toggle('expanded');
        }}

        function escapeHtml(text) {{
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
    </script>
</body>
</html>"""
