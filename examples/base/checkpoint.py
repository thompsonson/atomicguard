"""Checkpoint CLI utilities for resumable workflows."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from atomicguard.domain.models import (
    AmendmentType,
    HumanAmendment,
    WorkflowCheckpoint,
)
from atomicguard.infrastructure.persistence import FilesystemCheckpointDAG

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import CheckpointDAGInterface

console = Console()


def create_checkpoint_commands(checkpoint_dir: str | Path) -> click.Group:
    """
    Create a click group with checkpoint subcommands.

    Args:
        checkpoint_dir: Directory for checkpoint storage

    Returns:
        Click group with list, show, and delete commands
    """
    checkpoint_dir = Path(checkpoint_dir)

    @click.group()
    def checkpoints() -> None:
        """Manage workflow checkpoints."""
        pass

    @checkpoints.command("list")
    @click.option(
        "--workflow-id",
        default=None,
        help="Filter by workflow ID",
    )
    def list_checkpoints(workflow_id: str | None) -> None:
        """List all checkpoints."""
        try:
            dag = FilesystemCheckpointDAG(str(checkpoint_dir))
            checkpoints_list = dag.list_checkpoints(workflow_id)

            if not checkpoints_list:
                console.print("[dim]No checkpoints found.[/dim]")
                return

            table = Table(title="Workflow Checkpoints")
            table.add_column("Checkpoint ID", style="cyan")
            table.add_column("Failed Step", style="yellow")
            table.add_column("Failure Type", style="red")
            table.add_column("Created", style="dim")

            for cp in checkpoints_list:
                table.add_row(
                    cp.checkpoint_id[:12] + "...",
                    cp.failed_step,
                    cp.failure_type.value,
                    cp.created_at[:19],
                )

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error listing checkpoints: {e}[/red]")

    @checkpoints.command("show")
    @click.argument("checkpoint_id")
    def show_checkpoint(checkpoint_id: str) -> None:
        """Show details of a checkpoint."""
        try:
            dag = FilesystemCheckpointDAG(str(checkpoint_dir))

            # Try to find checkpoint by prefix match
            all_checkpoints = dag.list_checkpoints()
            matching = [
                c for c in all_checkpoints if c.checkpoint_id.startswith(checkpoint_id)
            ]

            if not matching:
                console.print(f"[red]Checkpoint not found: {checkpoint_id}[/red]")
                return

            if len(matching) > 1:
                console.print("[yellow]Multiple matches, showing first:[/yellow]")

            cp = matching[0]

            # Display checkpoint details
            console.print(
                Panel(
                    f"[bold]Checkpoint:[/bold] {cp.checkpoint_id}\n"
                    f"[bold]Workflow:[/bold] {cp.workflow_id}\n"
                    f"[bold]Created:[/bold] {cp.created_at}",
                    title="Identity",
                )
            )

            console.print(
                Panel(
                    f"[bold]Failed Step:[/bold] {cp.failed_step}\n"
                    f"[bold]Failure Type:[/bold] {cp.failure_type.value}\n"
                    f"[bold]Feedback:[/bold]\n{cp.failure_feedback}",
                    title="Failure Details",
                )
            )

            if cp.completed_steps:
                console.print(
                    Panel(
                        "\n".join(f"✓ {step}" for step in cp.completed_steps),
                        title="Completed Steps",
                    )
                )

            # Show resume hint
            console.print("\n[bold]Resume with:[/bold]")
            console.print(
                f"  [cyan]python -m examples.sdlc.run resume {cp.checkpoint_id[:12]} --artifact fixed.py[/cyan]"
            )
            console.print(
                f'  [cyan]python -m examples.sdlc.run resume {cp.checkpoint_id[:12]} --feedback "Additional guidance"[/cyan]'
            )

        except Exception as e:
            console.print(f"[red]Error showing checkpoint: {e}[/red]")

    return checkpoints


def create_amendment(
    checkpoint_id: str,
    amendment_type: AmendmentType,
    content: str,
    context: str = "",
    parent_artifact_id: str | None = None,
    additional_rmax: int = 0,
    created_by: str = "cli",
) -> HumanAmendment:
    """
    Create a HumanAmendment for workflow resume.

    Args:
        checkpoint_id: ID of checkpoint to amend
        amendment_type: Type of amendment (ARTIFACT or FEEDBACK)
        content: Human-provided content
        context: Additional context
        parent_artifact_id: Link to failed artifact
        additional_rmax: Extra retry budget
        created_by: Who created the amendment

    Returns:
        HumanAmendment instance
    """
    return HumanAmendment(
        amendment_id=str(uuid.uuid4()),
        checkpoint_id=checkpoint_id,
        amendment_type=amendment_type,
        created_at=datetime.now(UTC).isoformat(),
        created_by=created_by,
        content=content,
        context=context,
        parent_artifact_id=parent_artifact_id,
        additional_rmax=additional_rmax,
    )


def display_checkpoint_result(
    checkpoint: WorkflowCheckpoint,
    _checkpoint_dir: str | Path | None = None,
) -> None:
    """
    Display checkpoint creation result with resume instructions.

    Args:
        checkpoint: The created checkpoint
        _checkpoint_dir: Directory where checkpoint was stored (reserved for future use)
    """
    console.print("\n")
    console.print(
        Panel(
            f"[bold red]Checkpoint created:[/bold red] {checkpoint.checkpoint_id[:12]}...\n\n"
            f"[bold]Failed at:[/bold] {checkpoint.failed_step}\n"
            f"[bold]Type:[/bold] {checkpoint.failure_type.value}\n"
            f"[bold]Feedback:[/bold]\n{checkpoint.failure_feedback[:200]}{'...' if len(checkpoint.failure_feedback) > 200 else ''}",
            title="Workflow Paused",
            border_style="yellow",
        )
    )

    console.print("\n[bold]Resume options:[/bold]")
    console.print("  1. Provide fixed artifact:")
    console.print(
        f"     [cyan]python -m examples.sdlc.run resume {checkpoint.checkpoint_id[:12]} --artifact path/to/fixed.py[/cyan]"
    )
    console.print("\n  2. Provide feedback for retry:")
    console.print(
        f'     [cyan]python -m examples.sdlc.run resume {checkpoint.checkpoint_id[:12]} --feedback "Your guidance" --rmax 3[/cyan]'
    )
    console.print("\n  3. View checkpoint details:")
    console.print(
        f"     [cyan]python -m examples.sdlc.run checkpoints show {checkpoint.checkpoint_id[:12]}[/cyan]"
    )


def find_checkpoint_by_prefix(
    checkpoint_dag: CheckpointDAGInterface,
    prefix: str,
) -> WorkflowCheckpoint | None:
    """
    Find a checkpoint by ID prefix.

    Args:
        checkpoint_dag: Checkpoint DAG interface
        prefix: Checkpoint ID prefix

    Returns:
        Matching checkpoint or None
    """
    all_checkpoints = checkpoint_dag.list_checkpoints()
    matching = [c for c in all_checkpoints if c.checkpoint_id.startswith(prefix)]

    if not matching:
        return None

    if len(matching) > 1:
        console.print(
            f"[yellow]Warning: Multiple checkpoints match prefix '{prefix}', using most recent[/yellow]"
        )

    return matching[0]
