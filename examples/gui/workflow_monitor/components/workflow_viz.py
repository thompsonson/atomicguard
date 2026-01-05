"""Workflow DAG visualization component."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from ..state.manager import ExecutionState


def generate_mermaid_diagram(
    action_pairs: dict[str, dict],
    step_statuses: dict[str, str] | None = None,
) -> str:
    """
    Generate a Mermaid diagram for the workflow.

    Args:
        action_pairs: Workflow action pairs configuration
        step_statuses: Optional dict of step_id -> status (pending/running/success/failed)

    Returns:
        Mermaid diagram as markdown string
    """
    if not action_pairs:
        return "```mermaid\ngraph LR\n    empty[No steps configured]\n```"

    step_statuses = step_statuses or {}

    # Color mapping for step statuses
    colors = {
        "pending": "#9CA3AF",  # gray
        "running": "#FCD34D",  # yellow
        "success": "#34D399",  # green
        "failed": "#F87171",  # red
    }

    lines = ["```mermaid", "graph LR"]

    # Add nodes
    for step_id, config in action_pairs.items():
        guard_type = config.get("guard", "?")
        if guard_type == "composite":
            guards = config.get("guards", [])
            guard_type = "+".join(guards)

        # Truncate guard type for display
        display_guard = guard_type[:15] + "..." if len(guard_type) > 15 else guard_type

        status = step_statuses.get(step_id, "pending")
        color = colors.get(status, colors["pending"])

        # Node with guard type label
        lines.append(f'    {step_id}["{step_id}<br/><small>{display_guard}</small>"]')
        lines.append(
            f"    style {step_id} fill:{color},stroke:#374151,stroke-width:2px"
        )

    # Add edges for dependencies
    for step_id, config in action_pairs.items():
        requires = config.get("requires", [])
        for req in requires:
            lines.append(f"    {req} --> {step_id}")

    lines.append("```")

    return "\n".join(lines)


def create_workflow_viz() -> tuple[gr.Markdown, Callable[..., str]]:
    """
    Create the workflow visualization component.

    Returns:
        Tuple of (markdown_component, update_function)
    """
    with gr.Column():
        gr.Markdown("## Workflow Visualization")

        diagram = gr.Markdown(
            value="```mermaid\ngraph LR\n    loading[Loading...]\n```",
            label="Workflow DAG",
        )

    def update_diagram(state: ExecutionState | None) -> str:
        """Update the diagram based on current state."""
        if state is None or not state.steps:
            return "```mermaid\ngraph LR\n    empty[No workflow loaded]\n```"

        # Build action_pairs dict from state
        action_pairs = {}
        statuses = {}

        for step_id, step_status in state.steps.items():
            action_pairs[step_id] = {
                "guard": step_status.guard_type,
                "requires": list(step_status.requires),
            }
            statuses[step_id] = step_status.status

        return generate_mermaid_diagram(action_pairs, statuses)

    return diagram, update_diagram
