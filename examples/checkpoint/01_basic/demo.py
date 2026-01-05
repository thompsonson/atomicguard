#!/usr/bin/env python3
"""
Interactive demo showing checkpoint/resume functionality.

Demonstrates human-in-the-loop workflow resumption:
1. `run` command: Execute workflow that fails and creates a checkpoint
2. Human edits the artifact file
3. `resume` command: Resume workflow with the edited artifact

Usage:
    uv run python examples/checkpoint/demo.py run
    # Edit the artifact file, then:
    uv run python examples/checkpoint/demo.py resume <checkpoint_id>
"""

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import click

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.workflow import ResumableWorkflow
from atomicguard.domain.interfaces import GeneratorInterface, GuardInterface
from atomicguard.domain.models import (
    AmendmentType,
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    GuardResult,
    HumanAmendment,
    WorkflowStatus,
)
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.persistence.checkpoint import FilesystemCheckpointDAG
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

# Import from base - handle both direct execution and module execution
try:
    from examples.base.checkpoint import (
        find_checkpoint_by_prefix,
        write_checkpoint_output,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from examples.base.checkpoint import (
        find_checkpoint_by_prefix,
        write_checkpoint_output,
    )

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
ARTIFACT_DAG_DIR = OUTPUT_DIR / "artifact_dag"
CHECKPOINT_DAG_DIR = OUTPUT_DIR / "checkpoints"
HUMAN_ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
INSTRUCTIONS_PATH = OUTPUT_DIR / "instructions.md"
CONTEXT_PATH = OUTPUT_DIR / "context.md"


# =============================================================================
# Mock Components
# =============================================================================


class AlwaysFailGenerator(GeneratorInterface):
    """Generator that always produces failing code."""

    def __init__(self) -> None:
        self._call_count = 0

    def generate(
        self,
        context: Context,
        _template: PromptTemplate | None = None,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
    ) -> Artifact:
        self._call_count += 1
        content = f"def add(a, b):\n    return a - b  # BUG: attempt {self._call_count}"

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            content=content,
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now(UTC).isoformat(),
            attempt_number=self._call_count,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=ContextSnapshot(
                workflow_id=workflow_id,
                specification=context.specification,
                constraints=context.ambient.constraints if context.ambient else "",
                feedback_history=context.feedback_history,
                dependency_artifacts=(),
            ),
        )


class CorrectAddGuard(GuardInterface):
    """Guard that checks if add function returns correct results."""

    @property
    def guard_id(self) -> str:
        return "g_correct_add"

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        if "return a + b" in artifact.content:
            return GuardResult(passed=True, feedback="Correct implementation!")
        return GuardResult(
            passed=False,
            feedback=f"FAILED: Expected 'return a + b' but got:\n{artifact.content}",
        )


# =============================================================================
# Helper Functions
# =============================================================================


def create_workflow(
    artifact_dag: FilesystemArtifactDAG,
    checkpoint_dag: FilesystemCheckpointDAG,
) -> ResumableWorkflow:
    """Create a ResumableWorkflow with the demo configuration."""
    generator = AlwaysFailGenerator()
    guard = CorrectAddGuard()
    action_pair = ActionPair(generator=generator, guard=guard)

    workflow = ResumableWorkflow(
        artifact_dag=artifact_dag,
        checkpoint_dag=checkpoint_dag,
        rmax=2,
        constraints="Must use 'return a + b'",
        auto_checkpoint=True,
    )

    workflow.add_step(
        guard_id="g_correct_add",
        action_pair=action_pair,
        requires=(),
    )

    return workflow


def get_dags() -> tuple[FilesystemArtifactDAG, FilesystemCheckpointDAG]:
    """Initialize and return the DAGs."""
    ARTIFACT_DAG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DAG_DIR.mkdir(parents=True, exist_ok=True)

    artifact_dag = FilesystemArtifactDAG(str(ARTIFACT_DAG_DIR))
    checkpoint_dag = FilesystemCheckpointDAG(str(CHECKPOINT_DAG_DIR))

    return artifact_dag, checkpoint_dag


def check_output_exists() -> bool:
    """Check if output directory exists and prompt user."""
    if not OUTPUT_DIR.exists():
        return True

    click.echo(f"\nOutput directory already exists: {OUTPUT_DIR}")
    click.echo("\nOptions:")
    click.echo("  [c] Clean and continue (removes existing data)")
    click.echo("  [o] Overwrite (keeps checkpoints, overwrites files)")
    click.echo("  [a] Abort")

    choice = click.prompt("\nChoice", type=click.Choice(["c", "o", "a"]), default="c")

    if choice == "a":
        click.echo("Aborted.")
        return False
    elif choice == "c":
        shutil.rmtree(OUTPUT_DIR)
        click.echo(f"Cleaned: {OUTPUT_DIR}\n")

    return True


# =============================================================================
# CLI Commands
# =============================================================================


@click.group()
def cli() -> None:
    """Interactive checkpoint/resume demo.

    Demonstrates human-in-the-loop workflow resumption.
    """
    pass


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Show full context inline")
def run(verbose: bool) -> None:
    """Execute workflow - fails and writes output files for human edit."""
    click.echo("=" * 60)
    click.echo("CHECKPOINT/RESUME DEMO - Run Phase")
    click.echo("=" * 60)

    # Check if output exists
    if not check_output_exists():
        raise SystemExit(0)

    # Initialize DAGs
    artifact_dag, checkpoint_dag = get_dags()

    # Create and execute workflow
    workflow = create_workflow(artifact_dag, checkpoint_dag)

    click.echo("\nExecuting workflow (will fail after 2 attempts)...")
    result = workflow.execute("Write a function that adds two numbers")

    click.echo(f"\nWorkflow Status: {result.status.value}")

    if result.status == WorkflowStatus.CHECKPOINT:
        checkpoint = result.checkpoint
        assert checkpoint is not None

        # Get the failed artifact content
        failed_artifact_content = ""
        if checkpoint.failed_artifact_id:
            try:
                failed_artifact = artifact_dag.get_artifact(
                    checkpoint.failed_artifact_id
                )
                failed_artifact_content = failed_artifact.content
            except KeyError:
                failed_artifact_content = "# Failed artifact not found"

        # Write output files for human editing
        artifact_path = write_checkpoint_output(
            checkpoint=checkpoint,
            failed_artifact_content=failed_artifact_content,
            output_dir=OUTPUT_DIR,
            resume_command="uv run python examples/checkpoint/demo.py",
        )

        short_id = checkpoint.checkpoint_id[:12]

        click.echo("\n" + "=" * 60)
        click.echo(click.style("[CHECKPOINT CREATED]", fg="yellow", bold=True))
        click.echo("=" * 60)
        click.echo(f"\nCheckpoint ID: {short_id}")
        click.echo(f"Failed Step: {checkpoint.failed_step}")
        click.echo(f"Failure Type: {checkpoint.failure_type.value}")
        click.echo(f"\nFeedback:\n  {checkpoint.failure_feedback[:100]}...")

        click.echo("\n" + "-" * 60)
        click.echo(click.style("NEXT STEPS:", fg="green", bold=True))
        click.echo("-" * 60)

        click.echo("\n0. Review the full instructions:")
        click.echo(f"   {click.style(str(INSTRUCTIONS_PATH), fg='cyan')}")

        click.echo("\n1. Edit the artifact file:")
        click.echo(f"   {click.style(str(artifact_path), fg='cyan')}")
        click.echo("\n   Change 'return a - b' to 'return a + b'")

        click.echo("\n2. Resume the workflow:")
        click.echo(
            f"   {click.style(f'uv run python examples/checkpoint/demo.py resume {short_id}', fg='cyan')}"
        )

        if verbose:
            click.echo("\n" + "-" * 60)
            click.echo(click.style("CONTEXT:", fg="blue", bold=True))
            click.echo("-" * 60)
            click.echo(f"\nSpecification: {checkpoint.specification}")
            click.echo(f"Constraints: {checkpoint.constraints}")
            if checkpoint.provenance_ids:
                click.echo(f"Prior attempts: {len(checkpoint.provenance_ids)}")

        click.echo("\n" + "=" * 60)

    elif result.status == WorkflowStatus.SUCCESS:
        click.echo(click.style("\n[SUCCESS]", fg="green", bold=True))
        click.echo("Workflow completed successfully!")
        click.echo(f"Artifacts: {list(result.artifacts.keys())}")

    else:
        click.echo(click.style("\n[FAILED]", fg="red", bold=True))
        click.echo(f"Unexpected status: {result.status.value}")


@cli.command()
@click.argument("checkpoint_id")
@click.option(
    "--artifact",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to artifact file (default: reads from output/artifacts/{step}.py)",
)
def resume(checkpoint_id: str, artifact: Path | None) -> None:
    """Resume workflow from checkpoint using edited artifact."""
    click.echo("=" * 60)
    click.echo("CHECKPOINT/RESUME DEMO - Resume Phase")
    click.echo("=" * 60)

    # Initialize DAGs
    artifact_dag, checkpoint_dag = get_dags()

    # Find checkpoint by prefix
    checkpoint = find_checkpoint_by_prefix(checkpoint_dag, checkpoint_id)

    if checkpoint is None:
        click.echo(
            click.style(f"\n[ERROR] Checkpoint not found: {checkpoint_id}", fg="red")
        )
        click.echo("\nAvailable checkpoints:")
        for cp in checkpoint_dag.list_checkpoints():
            click.echo(f"  - {cp.checkpoint_id[:12]} ({cp.failed_step})")
        raise SystemExit(1)

    click.echo(f"\nResuming from checkpoint: {checkpoint.checkpoint_id[:12]}")
    click.echo(f"Failed step: {checkpoint.failed_step}")

    # Determine artifact path
    if artifact is None:
        artifact = HUMAN_ARTIFACTS_DIR / f"{checkpoint.failed_step}.py"

    if not artifact.exists():
        click.echo(
            click.style(f"\n[ERROR] Artifact file not found: {artifact}", fg="red")
        )
        click.echo("\nMake sure you have:")
        click.echo("  1. Run the 'run' command first")
        click.echo("  2. Edited the artifact file")
        raise SystemExit(1)

    # Read edited artifact content
    content = artifact.read_text()
    click.echo(f"\nLoaded artifact from: {artifact}")
    click.echo(f"Content preview:\n  {content[:80]}...")

    # Create HumanAmendment
    amendment = HumanAmendment(
        amendment_id=str(uuid.uuid4()),
        checkpoint_id=checkpoint.checkpoint_id,
        amendment_type=AmendmentType.ARTIFACT,
        created_at=datetime.now(UTC).isoformat(),
        created_by="cli",
        content=content,
        context="Human edited the artifact to fix the bug",
        parent_artifact_id=checkpoint.failed_artifact_id,
        additional_rmax=0,
    )

    # Create workflow and resume
    workflow = create_workflow(artifact_dag, checkpoint_dag)
    resume_result = workflow.resume(
        checkpoint_id=checkpoint.checkpoint_id,
        amendment=amendment,
    )

    click.echo(f"\nResume Result: {resume_result.status.value}")

    if resume_result.status == WorkflowStatus.SUCCESS:
        click.echo("\n" + "=" * 60)
        click.echo(click.style("[SUCCESS]", fg="green", bold=True))
        click.echo("=" * 60)
        click.echo("\nWorkflow completed successfully!")
        click.echo(f"Artifacts: {list(resume_result.artifacts.keys())}")

        final_artifact = resume_result.artifacts.get("g_correct_add")
        if final_artifact:
            click.echo("\nFinal artifact:")
            click.echo(f"  Source: {final_artifact.source.value}")
            click.echo(f"  Status: {final_artifact.status.value}")
            click.echo("\nContent:")
            click.echo(final_artifact.content)

    elif resume_result.status == WorkflowStatus.CHECKPOINT:
        new_checkpoint = resume_result.checkpoint
        assert new_checkpoint is not None
        click.echo("\n" + "=" * 60)
        click.echo(click.style("[STILL FAILING]", fg="yellow", bold=True))
        click.echo("=" * 60)
        click.echo(f"\nNew checkpoint created: {new_checkpoint.checkpoint_id[:12]}")
        click.echo(f"Feedback: {new_checkpoint.failure_feedback[:100]}...")
        click.echo("\nEdit the artifact again and retry.")

    else:
        click.echo(click.style("\n[FAILED]", fg="red", bold=True))
        click.echo(f"Status: {resume_result.status.value}")


@cli.command()
def clean() -> None:
    """Remove the output directory and start fresh."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        click.echo(f"Removed: {OUTPUT_DIR}")
    else:
        click.echo("Output directory does not exist.")


@cli.command("list")
def list_checkpoints() -> None:
    """List all checkpoints."""
    if not CHECKPOINT_DAG_DIR.exists():
        click.echo("No checkpoints found. Run the 'run' command first.")
        return

    checkpoint_dag = FilesystemCheckpointDAG(str(CHECKPOINT_DAG_DIR))
    checkpoints = checkpoint_dag.list_checkpoints()

    if not checkpoints:
        click.echo("No checkpoints found.")
        return

    click.echo(f"\nFound {len(checkpoints)} checkpoint(s):\n")
    for cp in checkpoints:
        click.echo(f"  {click.style(cp.checkpoint_id[:12], fg='cyan')}")
        click.echo(f"    Failed step: {cp.failed_step}")
        click.echo(f"    Type: {cp.failure_type.value}")
        click.echo(f"    Created: {cp.created_at[:19]}")
        click.echo()


if __name__ == "__main__":
    cli()
