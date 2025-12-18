"""
Human-in-the-loop review guard.

Blocks workflow until human approval via CLI prompts.
"""

from typing import Any

from rich.console import Console
from rich.prompt import Prompt
from rich.syntax import Syntax

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult


class HumanReviewGuard(GuardInterface):
    """
    Blocks workflow until human approval.

    Per paper Phase 8 (Human Oversight):
    - Pauses workflow to poll external oracle (human)
    - Returns approval or rejection with feedback
    - Feedback flows back to generator for retry

    This implementation uses synchronous CLI prompts.
    For async/distributed use, extend with file-based or webhook polling.
    """

    def __init__(self, prompt_title: str = "HUMAN REVIEW REQUIRED"):
        """
        Args:
            prompt_title: Title displayed in the review prompt
        """
        self.prompt_title = prompt_title
        self.console = Console()

    def validate(self, artifact: Artifact, **deps: Any) -> GuardResult:
        """
        Display artifact and prompt for human approval.

        Args:
            artifact: The artifact to review
            **deps: Dependencies shown for context

        Returns:
            GuardResult based on human decision
        """
        self.console.print(f"\n[bold yellow]═══ {self.prompt_title} ═══[/bold yellow]")
        self.console.print(f"[dim]Artifact ID: {artifact.artifact_id}[/dim]")
        self.console.print(f"[dim]Action Pair: {artifact.action_pair_id}[/dim]\n")

        # Display the artifact content with syntax highlighting
        self.console.print(
            Syntax(artifact.content, "python", theme="monokai", line_numbers=True)
        )

        # Show dependencies if present
        if deps:
            self.console.print("\n[dim]Dependencies:[/dim]")
            for key, dep_artifact in deps.items():
                self.console.print(f"  [dim]{key}: {dep_artifact.artifact_id}[/dim]")

        # Prompt for decision
        decision = Prompt.ask(
            "\n[bold]Approve this artifact?[/bold]", choices=["y", "n", "v"]
        )

        if decision == "v":
            # View more context
            self.console.print("\n[dim]Context:[/dim]")
            self.console.print(
                f"  Specification: {artifact.context.specification[:200]}..."
            )
            if artifact.context.feedback_history:
                self.console.print(
                    f"  Previous failures: {len(artifact.context.feedback_history)}"
                )
            decision = Prompt.ask("\n[bold]Approve?[/bold]", choices=["y", "n"])

        if decision == "y":
            return GuardResult(passed=True, feedback="Human approved")
        else:
            feedback = Prompt.ask("[bold]Rejection reason[/bold]")
            return GuardResult(passed=False, feedback=f"Human rejected: {feedback}")
