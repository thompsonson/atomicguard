"""Log viewer component for real-time log streaming."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from ..state.manager import StateManager


def create_log_viewer() -> tuple[gr.Dropdown, gr.Textbox, Callable[..., str]]:
    """
    Create the log viewer component.

    Returns:
        Tuple of (level_filter, log_output, update_function)
    """
    with gr.Column():
        gr.Markdown("## Execution Logs")

        level_filter = gr.Dropdown(
            label="Filter by Level",
            choices=["All", "DEBUG", "INFO", "WARNING", "ERROR"],
            value="All",
            scale=1,
        )

        log_output = gr.Textbox(
            label="Log Output",
            value="Logs will appear here when workflow execution starts...",
            lines=20,
            max_lines=30,
            interactive=False,
            autoscroll=True,
        )

    def update_logs(state_manager: StateManager, level: str) -> str:
        """
        Update log output based on filter.

        Args:
            state_manager: The state manager instance
            level: Log level filter

        Returns:
            Filtered log text
        """
        logs = state_manager.get_logs(level if level != "All" else None)
        if not logs:
            return "No logs yet..."
        return "\n".join(logs[-100:])  # Last 100 lines

    return level_filter, log_output, update_logs
