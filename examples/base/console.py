"""Rich console utilities for AtomicGuard examples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from atomicguard import Artifact

# Shared console instances
console = Console()
error_console = Console(stderr=True)


def print_header(title: str, subtitle: str | None = None) -> None:
    """Print a styled header panel."""
    content = Text(title, style="bold blue")
    if subtitle:
        content.append(f"\n{subtitle}", style="dim")
    console.print(Panel(content, expand=False))


def print_error(message: str, hint: str | None = None) -> None:
    """Print formatted error message to stderr."""
    content = Text(f"ERROR: {message}", style="bold red")
    if hint:
        content.append(f"\n\nHint: {hint}", style="yellow")
    error_console.print(Panel(content, title="Error", border_style="red"))


def print_success(message: str) -> None:
    """Print success message."""
    console.print(Panel(message, title="Success", border_style="green"))


def print_failure(message: str, details: str | None = None) -> None:
    """Print failure message."""
    content = Text(message, style="bold red")
    if details:
        content.append(f"\n{details}", style="dim")
    console.print(Panel(content, title="Failed", border_style="red"))


def print_workflow_info(
    workflow_name: str,
    model: str,
    host: str,
    rmax: int,
    output_path: str,
    artifact_dir: str,
    log_file: str,
    extra_info: dict[str, Any] | None = None,
) -> None:
    """Print workflow configuration info table."""
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    table.add_row("Workflow", workflow_name)
    table.add_row("Model", model)
    table.add_row("Host", host)
    table.add_row("Max attempts", str(rmax))
    table.add_row("Output", output_path)
    table.add_row("Artifacts", artifact_dir)
    table.add_row("Log file", log_file)

    if extra_info:
        for key, value in extra_info.items():
            table.add_row(key, str(value))

    console.print(table)


def print_steps(action_pairs: dict[str, Any]) -> None:
    """Print workflow steps."""
    console.print("\n[bold]Steps:[/bold]")
    for step_id, ap_config in action_pairs.items():
        guard_desc = ap_config["guard"]
        if guard_desc == "composite":
            guard_desc = f"composite({', '.join(ap_config.get('guards', []))})"
        requires = ap_config.get("requires", [])
        req_str = f" (requires: {requires})" if requires else ""
        console.print(f"  {step_id}: {guard_desc}{req_str}")


def print_provenance(provenance: Sequence[tuple[Artifact, str]]) -> None:
    """Print attempt history/provenance with action pair and guard attribution."""
    console.print("\n[bold]Attempt history:[/bold]")

    # Build a rich table: Retry | Action Pair | Guard | Error summary
    table = Table(show_header=True, box=None)
    table.add_column("Retry", style="cyan", width=6)
    table.add_column("Action Pair", style="magenta", width=18)
    table.add_column("Guard", style="yellow", width=20)
    table.add_column("Error (summary)", style="red")

    for i, (artifact, feedback) in enumerate(provenance, 1):
        ap_id = getattr(artifact, "action_pair_id", "unknown")
        guard_name = ""
        if hasattr(artifact, "guard_result") and artifact.guard_result:
            guard_name = artifact.guard_result.guard_name or ""
        # First line of feedback as summary
        summary = feedback.split("\n")[0][:80]
        table.add_row(str(i), ap_id, guard_name, summary)

    console.print(table)

    # Still print full feedback for debugging
    for i, (_, feedback) in enumerate(provenance, 1):
        console.print(f"\n[cyan]--- Attempt {i} detail ---[/cyan]")
        console.print(f"Feedback: {feedback}")
