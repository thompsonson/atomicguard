"""
Static Workflow Config Visualizer.

Generates self-contained HTML files with embedded Cytoscape.js
for interactive visualization of workflow topology — action pairs,
dependencies, escalation routes, and prompt templates.

Unlike export_workflow_html (which visualizes runtime execution),
this visualizes the static *design* of a workflow from its JSON config.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def export_workflow_config_html(
    workflow_path: str | Path,
    prompts_path: str | Path | None = None,
    output_path: str | Path = "workflow_config.html",
) -> Path:
    """Generate an interactive HTML visualization of a workflow config.

    Reads the workflow JSON and optional prompts JSON, then produces a
    self-contained HTML file showing the workflow topology with
    dependency edges, escalation routes, and prompt details.

    Args:
        workflow_path: Path to the workflow JSON file.
        prompts_path: Optional path to the prompts JSON file.
        output_path: Path where the HTML file will be written.

    Returns:
        Path to the generated HTML file.

    Example:
        >>> from atomicguard.visualization import export_workflow_config_html
        >>> path = export_workflow_config_html(
        ...     "examples/swe_bench_common/workflows/07_s1_decomposed.json",
        ...     "examples/swe_bench_pro/prompts.json",
        ...     "workflow_config.html",
        ... )
    """
    workflow_path = Path(workflow_path)
    output_path = Path(output_path)

    with open(workflow_path, encoding="utf-8") as f:
        workflow = json.load(f)

    prompts: dict[str, Any] = {}
    if prompts_path is not None:
        prompts_path = Path(prompts_path)
        with open(prompts_path, encoding="utf-8") as f:
            prompts = json.load(f)

    nodes, edges = _extract_graph(workflow)
    prompt_data = _extract_prompts(workflow, prompts)

    html = _render_html(workflow, nodes, edges, prompt_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path


def _extract_graph(
    workflow: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build Cytoscape nodes and edges from workflow action_pairs."""
    action_pairs = workflow.get("action_pairs", {})
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for ap_id, ap in action_pairs.items():
        guard_value = ap.get("guard", "")
        is_composite = guard_value == "composite"
        guards_list = ap.get("guards", [])
        has_escalation = bool(
            ap.get("escalate_feedback_to") or ap.get("escalate_feedback_by_guard")
        )

        nodes.append(
            {
                "data": {
                    "id": ap_id,
                    "label": ap_id,
                    "generator": ap.get("generator", ""),
                    "guard": guard_value,
                    "is_composite": is_composite,
                    "guards": guards_list,
                    "description": ap.get("description", ""),
                    "r_patience": ap.get("r_patience", ""),
                    "e_max": ap.get("e_max", ""),
                    "has_escalation": has_escalation,
                    "guard_config": json.dumps(ap.get("guard_config", {}))
                    if ap.get("guard_config")
                    else "",
                }
            }
        )

        # requires edges (dependency → dependent)
        for dep in ap.get("requires", []):
            edges.append(
                {
                    "data": {
                        "id": f"req_{dep}__{ap_id}",
                        "source": dep,
                        "target": ap_id,
                        "type": "requires",
                    }
                }
            )

        # escalation edges (failing step → escalation target)
        for target in ap.get("escalate_feedback_to", []):
            edges.append(
                {
                    "data": {
                        "id": f"esc_{ap_id}__{target}",
                        "source": ap_id,
                        "target": target,
                        "type": "escalation",
                    }
                }
            )

        # guard-specific escalation edges
        for guard_name, targets in ap.get("escalate_feedback_by_guard", {}).items():
            for target in targets:
                edge_id = f"gesc_{ap_id}__{guard_name}__{target}"
                edges.append(
                    {
                        "data": {
                            "id": edge_id,
                            "source": ap_id,
                            "target": target,
                            "type": "guard_escalation",
                            "guard_label": guard_name,
                        }
                    }
                )

    return nodes, edges


def _extract_prompts(
    workflow: dict[str, Any],
    prompts: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Merge prompt data keyed by action pair ID."""
    result: dict[str, dict[str, str]] = {}
    action_pairs = workflow.get("action_pairs", {})

    for ap_id in action_pairs:
        if ap_id in prompts:
            p = prompts[ap_id]
            result[ap_id] = {
                "role": p.get("role", ""),
                "task": p.get("task", ""),
                "constraints": p.get("constraints", ""),
                "feedback_wrapper": p.get("feedback_wrapper", ""),
                "escalation_feedback_wrapper": p.get("escalation_feedback_wrapper", ""),
            }

    return result


def _render_html(
    workflow: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    prompt_data: dict[str, dict[str, str]],
) -> str:
    """Render the complete self-contained HTML visualization."""
    name = workflow.get("name", "Unnamed Workflow")
    description = workflow.get("description", "")
    rmax = workflow.get("rmax", "")
    step_count = len(workflow.get("action_pairs", {}))

    nodes_json = json.dumps(nodes, indent=2)
    edges_json = json.dumps(edges, indent=2)
    prompts_json = json.dumps(prompt_data, indent=2)
    generated_at = datetime.now().isoformat()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workflow Config: {_escape(name)}</title>
    <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
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
        .header-desc {{
            margin-top: 0.4rem;
            font-size: 0.85rem;
            opacity: 0.8;
            max-width: 80ch;
            line-height: 1.5;
        }}
        .header-desc p {{
            margin: 0.25rem 0;
        }}
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
        .detail-section {{
            margin-bottom: 1.5rem;
        }}
        .collapsible {{
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 0.75rem;
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
            background: #f6f8fa;
            color: #24292e;
            padding: 1rem;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            overflow-x: auto;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.82rem;
            line-height: 1.5;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
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
        .placeholder-token {{
            background: #dbeafe;
            color: #1d4ed8;
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
            font-weight: 600;
        }}
        .guards-list {{
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin-top: 0.25rem;
        }}
        .guard-tag {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            background: #ede9fe;
            color: #6d28d9;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 500;
        }}
        .legend {{
            display: flex;
            gap: 1.5rem;
            padding: 1rem;
            background: #f9fafb;
            border-radius: 8px;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
        }}
        .legend-line {{
            width: 24px;
            height: 3px;
            position: relative;
        }}
        .legend-line.requires {{
            background: #6366f1;
        }}
        .legend-line.escalation {{
            background: #ec4899;
            border-top: 2px dashed #ec4899;
            height: 0;
        }}
        .legend-line.guard-esc {{
            background: transparent;
            border-top: 2px dotted #ec4899;
            height: 0;
        }}
        footer {{
            background: #1f2937;
            color: #9ca3af;
            padding: 0.5rem 2rem;
            font-size: 0.8rem;
            text-align: center;
        }}
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
            <h1>Workflow Config: {_escape(name)}</h1>
            <div class="header-info">
                <span><strong>Steps:</strong> {step_count}</span>
                <span><strong>rmax:</strong> {rmax}</span>
            </div>
            <div class="header-desc">{_format_description(description)}</div>
        </header>
        <main>
            <div id="cy"></div>
            <div class="sidebar">
                <h2>Action Pair Details</h2>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-line requires"></div>
                        <span>Requires (dependency)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-line escalation"></div>
                        <span>Escalation feedback</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-line guard-esc"></div>
                        <span>Guard escalation</span>
                    </div>
                </div>
                <div id="detail-panel">
                    <p class="no-selection">Click on an action pair node to view details</p>
                </div>
            </div>
        </main>
        <footer>
            Generated by AtomicGuard Workflow Config Visualizer | {generated_at}
        </footer>
    </div>

    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        const prompts = {prompts_json};

        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: {{
                nodes: nodes,
                edges: edges
            }},
            style: [
                // Default action pair nodes
                {{
                    selector: 'node',
                    style: {{
                        'shape': 'round-rectangle',
                        'width': 'label',
                        'height': 40,
                        'padding': '14px',
                        'background-color': '#6366f1',
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '11px',
                        'font-weight': 'bold',
                        'color': 'white'
                    }}
                }},
                // Composite guard nodes — slightly different colour
                {{
                    selector: 'node[is_composite]',
                    style: {{
                        'background-color': '#7c3aed'
                    }}
                }},
                // Nodes with escalation config — pink dashed border
                {{
                    selector: 'node[has_escalation]',
                    style: {{
                        'border-width': 3,
                        'border-color': '#ec4899',
                        'border-style': 'dashed'
                    }}
                }},
                // Requires edges — solid blue
                {{
                    selector: 'edge[type="requires"]',
                    style: {{
                        'width': 3,
                        'line-color': '#6366f1',
                        'target-arrow-color': '#6366f1',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }},
                // Escalation edges — dashed pink
                {{
                    selector: 'edge[type="escalation"]',
                    style: {{
                        'width': 2,
                        'line-color': '#ec4899',
                        'line-style': 'dashed',
                        'target-arrow-color': '#ec4899',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }},
                // Guard escalation edges — dotted pink with label
                {{
                    selector: 'edge[type="guard_escalation"]',
                    style: {{
                        'width': 2,
                        'line-color': '#ec4899',
                        'line-style': 'dotted',
                        'target-arrow-color': '#ec4899',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(guard_label)',
                        'font-size': '9px',
                        'color': '#9d174d',
                        'text-background-color': 'white',
                        'text-background-opacity': 0.9,
                        'text-background-padding': '2px',
                        'text-rotation': 'autorotate'
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
                nodeSep: 60,
                rankSep: 120,
                padding: 30
            }}
        }});

        // Handle node clicks
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            showNodeDetail(node.data());
        }});

        function showNodeDetail(d) {{
            const panel = document.getElementById('detail-panel');
            const p = prompts[d.id] || {{}};

            // Meta grid
            let guardsHtml = '';
            if (d.is_composite && d.guards && d.guards.length > 0) {{
                guardsHtml = `
                    <div class="meta-item" style="grid-column: 1 / -1;">
                        <label>Guards (composite)</label>
                        <div class="guards-list">
                            ${{d.guards.map(g => `<span class="guard-tag">${{escapeHtml(g)}}</span>`).join('')}}
                        </div>
                    </div>`;
            }}

            let guardConfigHtml = '';
            if (d.guard_config) {{
                guardConfigHtml = `
                    <div class="collapsible">
                        <div class="collapsible-header" onclick="toggleCollapsible(this)">
                            <h4>Guard Config</h4>
                            <span class="toggle">&#9660;</span>
                        </div>
                        <div class="collapsible-content">
                            <pre class="code-block">${{escapeHtml(formatJson(d.guard_config))}}</pre>
                        </div>
                    </div>`;
            }}

            // Prompt sections
            let promptSections = '';
            const promptFields = [
                ['role', 'Prompt: Role', false],
                ['task', 'Prompt: Task', true],
                ['constraints', 'Prompt: Constraints', false],
                ['feedback_wrapper', 'Prompt: Feedback Wrapper', false],
                ['escalation_feedback_wrapper', 'Prompt: Escalation Feedback Wrapper', false],
            ];
            for (const [key, title, expandByDefault] of promptFields) {{
                const val = p[key];
                if (!val) continue;
                const expanded = expandByDefault ? ' expanded' : '';
                promptSections += `
                    <div class="collapsible${{expanded}}">
                        <div class="collapsible-header" onclick="toggleCollapsible(this)">
                            <h4>${{title}}</h4>
                            <span class="toggle">&#9660;</span>
                        </div>
                        <div class="collapsible-content">
                            <pre class="code-block">${{highlightPlaceholders(escapeHtml(val))}}</pre>
                        </div>
                    </div>`;
            }}

            if (!promptSections && !Object.keys(p).length) {{
                promptSections = '<p style="color:#9ca3af;font-style:italic;">No prompt data available</p>';
            }}

            panel.innerHTML = `
                <div class="detail-section">
                    <div class="meta-grid">
                        <div class="meta-item">
                            <label>Generator</label>
                            <span>${{escapeHtml(d.generator)}}</span>
                        </div>
                        <div class="meta-item">
                            <label>Guard</label>
                            <span>${{escapeHtml(d.guard)}}</span>
                        </div>
                        <div class="meta-item" style="grid-column: 1 / -1;">
                            <label>Description</label>
                            <span>${{escapeHtml(d.description)}}</span>
                        </div>
                        <div class="meta-item">
                            <label>r_patience</label>
                            <span>${{d.r_patience || 'default'}}</span>
                        </div>
                        <div class="meta-item">
                            <label>e_max</label>
                            <span>${{d.e_max || 'default'}}</span>
                        </div>
                        ${{guardsHtml}}
                    </div>
                </div>
                ${{guardConfigHtml}}
                ${{promptSections}}
            `;
        }}

        function toggleCollapsible(header) {{
            const collapsible = header.parentElement;
            collapsible.classList.toggle('expanded');
        }}

        function escapeHtml(text) {{
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = String(text);
            return div.innerHTML;
        }}

        function highlightPlaceholders(escapedHtml) {{
            // Highlight {{placeholder}} tokens in prompt text
            return escapedHtml.replace(
                /\\{{([^}}]+)\\}}/g,
                '<span class="placeholder-token">{{$1}}</span>'
            );
        }}

        function formatJson(str) {{
            try {{
                return JSON.stringify(JSON.parse(str), null, 2);
            }} catch {{
                return str;
            }}
        }}
    </script>
</body>
</html>"""


def _escape(text: str) -> str:
    """HTML-escape text for safe embedding in the template."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_description(text: str) -> str:
    """Split a description into paragraphs at sentence boundaries."""
    import re

    if not text:
        return ""
    sentences = re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip())
    return "\n".join(f"<p>{_escape(s)}</p>" for s in sentences)
