"""Execution monitoring panel component."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from ..state.manager import ExecutionState


def format_step_card(
    step_id: str,
    guard_type: str,
    status: str,
    attempt: int,
    max_attempts: int,
    feedback: str,
    requires: tuple[str, ...],
) -> str:
    """
    Format a step status card as HTML.

    Args:
        step_id: Step identifier
        guard_type: Guard type string
        status: Current status (pending/running/success/failed)
        attempt: Current attempt number
        max_attempts: Maximum attempts allowed
        feedback: Last feedback message
        requires: Dependencies

    Returns:
        HTML string for the step card
    """
    # Status colors and icons
    status_styles = {
        "pending": ("background: #F3F4F6; border-color: #9CA3AF;", "⏳"),
        "running": ("background: #FEF3C7; border-color: #F59E0B;", "🔄"),
        "success": ("background: #D1FAE5; border-color: #10B981;", "✅"),
        "failed": ("background: #FEE2E2; border-color: #EF4444;", "❌"),
    }

    style, icon = status_styles.get(status, status_styles["pending"])

    deps_str = ", ".join(requires) if requires else "None"
    feedback_html = (
        f'<div style="color: #DC2626; font-size: 0.85em; margin-top: 4px;">'
        f"<strong>Feedback:</strong> {feedback[:200]}{'...' if len(feedback) > 200 else ''}"
        f"</div>"
        if feedback
        else ""
    )

    return f"""
    <div style="border: 2px solid; border-radius: 8px; padding: 12px; margin: 8px 0; {style}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: bold; font-size: 1.1em;">{icon} {step_id}</span>
            <span style="font-size: 0.9em;">Attempt {attempt}/{max_attempts}</span>
        </div>
        <div style="font-size: 0.9em; color: #4B5563; margin-top: 4px;">
            <strong>Guard:</strong> {guard_type}
        </div>
        <div style="font-size: 0.85em; color: #6B7280; margin-top: 2px;">
            <strong>Requires:</strong> {deps_str}
        </div>
        {feedback_html}
    </div>
    """


def create_execution_panel() -> tuple[
    gr.HTML, gr.Textbox, gr.Textbox, Callable[..., tuple[str, str, str]]
]:
    """
    Create the execution monitoring panel.

    Returns:
        Tuple of (step_cards_html, status_text, duration_text, update_function)
    """
    with gr.Column():
        gr.Markdown("## Execution Status")

        with gr.Row():
            status_text = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False,
                scale=2,
            )
            duration_text = gr.Textbox(
                label="Duration",
                value="-",
                interactive=False,
                scale=1,
            )

        gr.Markdown("### Steps")
        step_cards = gr.HTML(
            value="<p style='color: #6B7280;'>No workflow loaded. Configure and start a workflow to see step status.</p>"
        )

    def update_execution_panel(
        state: ExecutionState | None,
    ) -> tuple[str, str, str]:
        """
        Update the execution panel based on current state.

        Returns:
            Tuple of (step_cards_html, status_text, duration_text)
        """
        if state is None:
            return (
                "<p>No state available</p>",
                "Unknown",
                "-",
            )

        # Build step cards HTML
        cards_html = ""
        if state.steps:
            for step_id, step_status in state.steps.items():
                cards_html += format_step_card(
                    step_id=step_id,
                    guard_type=step_status.guard_type,
                    status=step_status.status,
                    attempt=step_status.current_attempt,
                    max_attempts=step_status.max_attempts,
                    feedback=step_status.last_feedback,
                    requires=step_status.requires,
                )
        else:
            cards_html = "<p style='color: #6B7280;'>No steps configured</p>"

        # Status text
        if state.error:
            status = f"Error: {state.error}"
        elif state.is_running:
            current = state.current_step or "preparing"
            status = f"Running ({current})"
        elif state.result:
            status = f"Completed: {state.result.status.value}"
        else:
            status = "Ready"

        # Duration text
        if state.duration > 0:
            duration = f"{state.duration:.2f}s"
        elif state.is_running:
            duration = "Running..."
        else:
            duration = "-"

        return cards_html, status, duration

    return step_cards, status_text, duration_text, update_execution_panel
