#!/usr/bin/env python3
"""
Level 2 Checkpoint Demo: TDD workflow with config files.

Demonstrates checkpoint/resume using workflow.json and prompts.json configuration,
bridging the gap between the basic mock demo and real LLM-powered examples.

Key differences from Level 1 (examples/checkpoint/):
- Uses workflow.json to define steps
- Uses prompts.json for prompt templates
- Two-step workflow (g_test → g_impl) showing partial success
- Generators declared in config and registered in code

Usage:
    uv run python examples/checkpoint_tdd/demo.py run
    # Edit the artifact file, then:
    uv run python examples/checkpoint_tdd/demo.py resume <checkpoint_id>
"""

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import click

from atomicguard.application.workflow import ResumableWorkflow
from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import (
    AmendmentType,
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    HumanAmendment,
    WorkflowStatus,
)
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure import GeneratorRegistry
from atomicguard.infrastructure.persistence.checkpoint import FilesystemCheckpointDAG
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

# Import from base - handle both direct execution and module execution
try:
    from examples.base import (
        build_guard,
        find_checkpoint_by_prefix,
        load_prompts,
        load_workflow_config,
        write_checkpoint_output,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from examples.base import (
        build_guard,
        find_checkpoint_by_prefix,
        load_prompts,
        load_workflow_config,
        write_checkpoint_output,
    )

from atomicguard.application.action_pair import ActionPair

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
ARTIFACT_DAG_DIR = OUTPUT_DIR / "artifact_dag"
CHECKPOINT_DAG_DIR = OUTPUT_DIR / "checkpoints"
HUMAN_ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
WORKFLOW_PATH = SCRIPT_DIR / "workflow.json"
PROMPTS_PATH = SCRIPT_DIR / "prompts.json"

# =============================================================================
# Mock Generators (Registered so workflow.json can reference them)
# =============================================================================

# Valid test code that will pass SyntaxGuard
VALID_TEST_CODE = '''import pytest
from implementation import Stack


class TestStack:
    """Test suite for Stack implementation."""

    def test_push_and_pop(self):
        """Test basic push/pop operations."""
        s = Stack()
        s.push(1)
        s.push(2)
        assert s.pop() == 2
        assert s.pop() == 1

    def test_is_empty(self):
        """Test is_empty method."""
        s = Stack()
        assert s.is_empty() is True
        s.push(1)
        assert s.is_empty() is False

    def test_peek(self):
        """Test peek returns top without removing."""
        s = Stack()
        s.push(42)
        assert s.peek() == 42
        assert s.peek() == 42  # Still there
        assert s.pop() == 42

    def test_pop_empty_raises(self):
        """Test underflow on pop."""
        s = Stack()
        with pytest.raises(IndexError):
            s.pop()

    def test_peek_empty_raises(self):
        """Test underflow on peek."""
        s = Stack()
        with pytest.raises(IndexError):
            s.peek()
'''

# Buggy implementation with obvious fix hints
BUGGY_IMPL_CODE = '''class Stack:
    """Stack implementation with intentional bugs for demo."""

    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        # BUG: Change self._items.pop(0) to self._items.pop()
        return self._items.pop(0)

    def peek(self):
        if not self._items:
            raise IndexError("peek from empty stack")
        # BUG: Change self._items[0] to self._items[-1]
        return self._items[0]

    def is_empty(self):
        return len(self._items) == 0
'''


class MockTestGenerator(GeneratorInterface):
    """
    Mock generator that produces valid test code.

    In workflow.json: "generator": "MockTestGenerator"
    In real workflows, use: "generator": "OllamaGenerator"
    """

    def __init__(self, **_kwargs) -> None:
        """Accept any kwargs for compatibility with registry."""
        self._call_count = 0

    def generate(
        self,
        context: Context,
        _template: PromptTemplate | None = None,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
    ) -> Artifact:
        self._call_count += 1

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            content=VALID_TEST_CODE,
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


class MockImplGenerator(GeneratorInterface):
    """
    Mock generator that produces buggy implementation.

    The bugs have inline comments showing exactly what to fix.
    This simulates an LLM that got it wrong, requiring human intervention.

    In workflow.json: "generator": "MockImplGenerator"
    In real workflows, use: "generator": "OllamaGenerator"
    """

    def __init__(self, **_kwargs) -> None:
        """Accept any kwargs for compatibility with registry."""
        self._call_count = 0

    def generate(
        self,
        context: Context,
        _template: PromptTemplate | None = None,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
    ) -> Artifact:
        self._call_count += 1

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            content=BUGGY_IMPL_CODE,
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


# Register mock generators so workflow.json can reference them by name
# This is the same pattern used for real generators like OllamaGenerator
GeneratorRegistry.register("MockTestGenerator", MockTestGenerator)
GeneratorRegistry.register("MockImplGenerator", MockImplGenerator)


# =============================================================================
# Helper Functions
# =============================================================================


def get_dags() -> tuple[FilesystemArtifactDAG, FilesystemCheckpointDAG]:
    """Initialize and return the DAGs."""
    ARTIFACT_DAG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DAG_DIR.mkdir(parents=True, exist_ok=True)

    artifact_dag = FilesystemArtifactDAG(str(ARTIFACT_DAG_DIR))
    checkpoint_dag = FilesystemCheckpointDAG(str(CHECKPOINT_DAG_DIR))

    return artifact_dag, checkpoint_dag


def create_workflow(
    artifact_dag: FilesystemArtifactDAG,
    checkpoint_dag: FilesystemCheckpointDAG,
) -> ResumableWorkflow:
    """
    Create ResumableWorkflow from workflow.json and prompts.json.

    This demonstrates the config-driven pattern used in real workflows.
    """
    # Load configuration files
    workflow_config = load_workflow_config(
        WORKFLOW_PATH,
        required_fields=("name", "specification", "action_pairs"),
    )
    prompts = load_prompts(PROMPTS_PATH)

    # Create workflow
    workflow = ResumableWorkflow(
        artifact_dag=artifact_dag,
        checkpoint_dag=checkpoint_dag,
        rmax=workflow_config.get("rmax", 3),
        constraints=workflow_config.get("constraints", ""),
        auto_checkpoint=True,
    )

    # Build steps from action_pairs configuration
    action_pairs_config = workflow_config["action_pairs"]

    for step_id, ap_config in action_pairs_config.items():
        # Get generator from registry (declared in workflow.json)
        generator_name = ap_config.get("generator")
        if generator_name:
            generator = GeneratorRegistry.create(generator_name)
        else:
            raise ValueError(f"No generator specified for step '{step_id}'")

        # Build guard from config
        guard = build_guard(ap_config)

        # Get prompt template
        prompt_template = prompts.get(step_id)

        # Create action pair
        action_pair = ActionPair(
            generator=generator,
            guard=guard,
            prompt_template=prompt_template,
        )

        # Add step with dependencies
        requires = tuple(ap_config.get("requires", []))
        workflow.add_step(
            guard_id=step_id,
            action_pair=action_pair,
            requires=requires,
            deps=requires,
        )

    return workflow


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
    """Level 2 Checkpoint Demo: TDD workflow with config files.

    Demonstrates checkpoint/resume using workflow.json and prompts.json.
    """
    pass


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Show full context inline")
def run(verbose: bool) -> None:
    """Execute workflow - g_test passes, g_impl fails, creates checkpoint."""
    click.echo("=" * 60)
    click.echo("CHECKPOINT TDD DEMO - Run Phase")
    click.echo("=" * 60)
    click.echo("\nThis demo uses workflow.json and prompts.json configuration.")
    click.echo("Step 1 (g_test) will pass, Step 2 (g_impl) will fail.\n")

    # Check if output exists
    if not check_output_exists():
        raise SystemExit(0)

    # Initialize DAGs
    artifact_dag, checkpoint_dag = get_dags()

    # Load workflow config for specification
    workflow_config = load_workflow_config(
        WORKFLOW_PATH,
        required_fields=("name", "specification", "action_pairs"),
    )

    # Create and execute workflow
    workflow = create_workflow(artifact_dag, checkpoint_dag)

    click.echo("Executing workflow...")
    click.echo("  [g_test] Generating tests...")
    result = workflow.execute(workflow_config["specification"])

    # Display step results
    if "g_test" in result.artifacts:
        click.echo(click.style("  [g_test] PASSED", fg="green"))
    if result.failed_step == "g_impl":
        click.echo(click.style("  [g_impl] FAILED", fg="red"))

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
            resume_command="uv run python examples/checkpoint_tdd/demo.py",
        )

        short_id = checkpoint.checkpoint_id[:12]

        click.echo("\n" + "=" * 60)
        click.echo(click.style("[CHECKPOINT CREATED]", fg="yellow", bold=True))
        click.echo("=" * 60)
        click.echo(f"\nCheckpoint ID: {short_id}")
        click.echo(f"Failed Step: {checkpoint.failed_step}")
        click.echo(f"Completed Steps: {list(checkpoint.completed_steps)}")
        click.echo("\nFeedback (test failure output):")
        # Show first few lines of test output
        feedback_lines = checkpoint.failure_feedback.split("\n")[:5]
        for line in feedback_lines:
            click.echo(f"  {line}")
        if len(checkpoint.failure_feedback.split("\n")) > 5:
            click.echo("  ...")

        click.echo("\n" + "-" * 60)
        click.echo(click.style("NEXT STEPS:", fg="green", bold=True))
        click.echo("-" * 60)

        click.echo("\n0. Review the full instructions:")
        click.echo(f"   {click.style(str(OUTPUT_DIR / 'instructions.md'), fg='cyan')}")

        click.echo("\n1. Edit the artifact file:")
        click.echo(f"   {click.style(str(artifact_path), fg='cyan')}")
        click.echo("\n   The bugs have comments showing what to fix:")
        click.echo("   - Change pop(0) to pop()")
        click.echo("   - Change [0] to [-1]")

        click.echo("\n2. Resume the workflow:")
        click.echo(
            f"   {click.style(f'uv run python examples/checkpoint_tdd/demo.py resume {short_id}', fg='cyan')}"
        )

        if verbose:
            click.echo("\n" + "-" * 60)
            click.echo(click.style("CONTEXT:", fg="blue", bold=True))
            click.echo("-" * 60)
            click.echo(f"\nSpecification: {checkpoint.specification}")
            click.echo(f"Constraints: {checkpoint.constraints}")
            click.echo(f"Completed artifacts: {dict(checkpoint.artifact_ids)}")

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
    click.echo("CHECKPOINT TDD DEMO - Resume Phase")
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
    click.echo(f"Completed steps: {list(checkpoint.completed_steps)}")

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

    # Show diff hint
    if "pop(0)" not in content and "pop()" in content:
        click.echo(click.style("  Detected fix: pop(0) -> pop()", fg="green"))
    if "[0]" not in content and "[-1]" in content:
        click.echo(click.style("  Detected fix: [0] -> [-1]", fg="green"))

    # Create HumanAmendment
    amendment = HumanAmendment(
        amendment_id=str(uuid.uuid4()),
        checkpoint_id=checkpoint.checkpoint_id,
        amendment_type=AmendmentType.ARTIFACT,
        created_at=datetime.now(UTC).isoformat(),
        created_by="cli",
        content=content,
        context="Human fixed the Stack implementation bugs",
        parent_artifact_id=checkpoint.failed_artifact_id,
        additional_rmax=0,
    )

    # Create workflow and resume
    workflow = create_workflow(artifact_dag, checkpoint_dag)
    click.echo("\nValidating human artifact against tests...")
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
        click.echo(f"All artifacts: {list(resume_result.artifacts.keys())}")

        # Show final implementation
        final_artifact = resume_result.artifacts.get("g_impl")
        if final_artifact:
            click.echo(
                f"\nFinal implementation (source: {final_artifact.source.value}):"
            )
            click.echo("-" * 40)
            # Show just the class definition, not the full content
            for line in final_artifact.content.split("\n")[:20]:
                click.echo(f"  {line}")

    elif resume_result.status == WorkflowStatus.CHECKPOINT:
        new_checkpoint = resume_result.checkpoint
        assert new_checkpoint is not None
        click.echo("\n" + "=" * 60)
        click.echo(click.style("[STILL FAILING]", fg="yellow", bold=True))
        click.echo("=" * 60)
        click.echo(f"\nNew checkpoint created: {new_checkpoint.checkpoint_id[:12]}")
        click.echo("\nTest failure output:")
        for line in new_checkpoint.failure_feedback.split("\n")[:10]:
            click.echo(f"  {line}")
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
        click.echo(f"    Completed: {list(cp.completed_steps)}")
        click.echo(f"    Created: {cp.created_at[:19]}")
        click.echo()


@cli.command()
def show_config() -> None:
    """Display the workflow.json and prompts.json configuration."""
    click.echo("=" * 60)
    click.echo("CONFIGURATION FILES")
    click.echo("=" * 60)

    click.echo(f"\n{click.style('workflow.json', fg='cyan', bold=True)}")
    click.echo("-" * 40)
    click.echo(WORKFLOW_PATH.read_text())

    click.echo(f"\n{click.style('prompts.json', fg='cyan', bold=True)}")
    click.echo("-" * 40)
    click.echo(PROMPTS_PATH.read_text())


if __name__ == "__main__":
    cli()
