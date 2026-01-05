#!/usr/bin/env python3
"""
Level 3 Checkpoint Demo: LLM-powered workflow with checkpoint/resume.

Demonstrates checkpoint/resume using a real LLM (Ollama) instead of mock generators.
This is the first example that shows actual AI generation with guard validation.

Task: Template Engine - write a render_template function that handles variable
substitution ({{ var }}) and conditionals ({% if var %}...{% endif %}).

Key differences from Level 2 (examples/checkpoint_tdd/):
- Uses OllamaGenerator (real LLM) instead of mock generators
- Single-step workflow (simpler than TDD's two-step)
- LLM may succeed or fail naturally (not deterministic like mocks)
- Test code provided via guard_config (not dependency artifact)

Prerequisites:
- Ollama running: ollama serve
- Model available: ollama pull qwen2.5-coder:7b

Usage:
    uv run python examples/checkpoint_llm/demo.py run --host http://localhost:11434
    # If checkpoint created, edit the artifact file, then:
    uv run python examples/checkpoint_llm/demo.py resume <checkpoint_id>
"""

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import click

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.workflow import ResumableWorkflow
from atomicguard.domain.models import (
    AmendmentType,
    HumanAmendment,
    WorkflowStatus,
)
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
        normalize_base_url,
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
        normalize_base_url,
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
WORKFLOW_PATH = SCRIPT_DIR / "workflow.json"
PROMPTS_PATH = SCRIPT_DIR / "prompts.json"


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
    host: str,
    model: str | None,
) -> ResumableWorkflow:
    """
    Create ResumableWorkflow from workflow.json and prompts.json.

    Key difference from Level 2: Uses OllamaGenerator and loads test code
    from external file to inject into guard_config.
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

    # Get effective model (CLI override > workflow.json)
    effective_model = model or workflow_config.get("model", "qwen2.5-coder:7b")

    # Build steps from action_pairs configuration
    action_pairs_config = workflow_config["action_pairs"]

    for step_id, ap_config in action_pairs_config.items():
        # Get generator from registry with LLM config
        generator_name = ap_config.get("generator")
        if generator_name:
            generator_config = {
                "model": effective_model,
                "base_url": normalize_base_url(host),
            }
            generator = GeneratorRegistry.create(generator_name, **generator_config)
        else:
            raise ValueError(f"No generator specified for step '{step_id}'")

        # Build guard from config
        # Special handling: if guard_config has test_file, load the test code
        guard_config = ap_config.get("guard_config", {})
        if "test_file" in guard_config:
            test_file_path = SCRIPT_DIR / guard_config["test_file"]
            if test_file_path.exists():
                guard_config = guard_config.copy()
                guard_config["test_code"] = test_file_path.read_text()
                del guard_config["test_file"]  # Replace with actual content

        # Build guard with updated config
        ap_config_with_test = ap_config.copy()
        ap_config_with_test["guard_config"] = guard_config
        guard = build_guard(ap_config_with_test)

        # Get prompt template
        prompt_template = prompts.get(step_id)

        # Create action pair
        action_pair = ActionPair(
            generator=generator,
            guard=guard,
            prompt_template=prompt_template,
        )

        # Add step (no dependencies for this single-step workflow)
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
    """Level 3 Checkpoint Demo: LLM-powered workflow with checkpoint/resume.

    Demonstrates checkpoint/resume using a real LLM (Ollama).
    """
    pass


@cli.command()
@click.option(
    "--host",
    default="http://localhost:11434",
    help="Ollama host URL",
)
@click.option(
    "--model",
    default=None,
    help="Model to use (default: from workflow.json)",
)
@click.option("--verbose", "-v", is_flag=True, help="Show full context inline")
def run(host: str, model: str | None, verbose: bool) -> None:
    """Execute workflow - LLM generates code, guard validates with tests."""
    click.echo("=" * 60)
    click.echo("CHECKPOINT LLM DEMO - Run Phase")
    click.echo("=" * 60)
    click.echo("\nThis demo uses a real LLM to generate code.")
    click.echo("The DynamicTestGuard runs pytest tests against the generated code.")
    click.echo(f"\nLLM Host: {host}")

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

    effective_model = model or workflow_config.get("model", "qwen2.5-coder:7b")
    click.echo(f"Model: {effective_model}\n")

    # Create and execute workflow
    try:
        workflow = create_workflow(artifact_dag, checkpoint_dag, host, model)
    except ImportError as e:
        click.echo(click.style(f"\n[ERROR] {e}", fg="red"))
        click.echo("\nMake sure openai is installed: pip install openai")
        raise SystemExit(1) from None

    click.echo("Executing workflow...")
    click.echo("  [g_template] Generating render_template function...")

    try:
        result = workflow.execute(workflow_config["specification"])
    except Exception as e:
        click.echo(click.style(f"\n[ERROR] LLM call failed: {e}", fg="red"))
        click.echo("\nMake sure Ollama is running:")
        click.echo("  ollama serve")
        click.echo(f"  ollama pull {effective_model}")
        raise SystemExit(1) from None

    click.echo(f"\nWorkflow Status: {result.status.value}")

    if result.status == WorkflowStatus.SUCCESS:
        click.echo("\n" + "=" * 60)
        click.echo(click.style("[SUCCESS]", fg="green", bold=True))
        click.echo("=" * 60)
        click.echo("\nThe LLM got it right! All tests passed.")
        click.echo(f"Artifacts: {list(result.artifacts.keys())}")

        final_artifact = result.artifacts.get("g_template")
        if final_artifact:
            click.echo("\nGenerated code:")
            click.echo("-" * 40)
            for line in final_artifact.content.split("\n")[:30]:
                click.echo(f"  {line}")

    elif result.status == WorkflowStatus.CHECKPOINT:
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
            resume_command="uv run python examples/checkpoint_llm/demo.py",
        )

        short_id = checkpoint.checkpoint_id[:12]

        click.echo("\n" + "=" * 60)
        click.echo(click.style("[CHECKPOINT CREATED]", fg="yellow", bold=True))
        click.echo("=" * 60)
        click.echo(f"\nCheckpoint ID: {short_id}")
        click.echo(f"Failed Step: {checkpoint.failed_step}")
        click.echo(f"Failure Type: {checkpoint.failure_type.value}")
        click.echo("\nTest failure output:")
        # Show first few lines of test output
        feedback_lines = checkpoint.failure_feedback.split("\n")[:8]
        for line in feedback_lines:
            click.echo(f"  {line}")
        if len(checkpoint.failure_feedback.split("\n")) > 8:
            click.echo("  ...")

        click.echo("\n" + "-" * 60)
        click.echo(click.style("NEXT STEPS:", fg="green", bold=True))
        click.echo("-" * 60)

        click.echo("\n0. Review the full instructions:")
        click.echo(f"   {click.style(str(OUTPUT_DIR / 'instructions.md'), fg='cyan')}")

        click.echo("\n1. Edit the artifact file:")
        click.echo(f"   {click.style(str(artifact_path), fg='cyan')}")
        click.echo("\n   Fix the render_template function to pass all tests.")

        click.echo("\n2. Resume the workflow:")
        click.echo(
            f"   {click.style(f'uv run python examples/checkpoint_llm/demo.py resume {short_id}', fg='cyan')}"
        )

        if verbose:
            click.echo("\n" + "-" * 60)
            click.echo(click.style("CONTEXT:", fg="blue", bold=True))
            click.echo("-" * 60)
            click.echo(f"\nSpecification: {checkpoint.specification}")
            click.echo(f"Constraints: {checkpoint.constraints}")

        click.echo("\n" + "=" * 60)

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
@click.option(
    "--host",
    default="http://localhost:11434",
    help="Ollama host URL (needed to rebuild workflow)",
)
@click.option(
    "--model",
    default=None,
    help="Model to use (default: from workflow.json)",
)
def resume(
    checkpoint_id: str, artifact: Path | None, host: str, model: str | None
) -> None:
    """Resume workflow from checkpoint using edited artifact."""
    click.echo("=" * 60)
    click.echo("CHECKPOINT LLM DEMO - Resume Phase")
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

    # Show preview
    preview_lines = content.split("\n")[:5]
    click.echo("\nContent preview:")
    for line in preview_lines:
        click.echo(f"  {line}")
    if len(content.split("\n")) > 5:
        click.echo("  ...")

    # Create HumanAmendment
    amendment = HumanAmendment(
        amendment_id=str(uuid.uuid4()),
        checkpoint_id=checkpoint.checkpoint_id,
        amendment_type=AmendmentType.ARTIFACT,
        created_at=datetime.now(UTC).isoformat(),
        created_by="cli",
        content=content,
        context="Human fixed the render_template implementation",
        parent_artifact_id=checkpoint.failed_artifact_id,
        additional_rmax=0,
    )

    # Create workflow and resume
    workflow = create_workflow(artifact_dag, checkpoint_dag, host, model)
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
        click.echo(f"Artifacts: {list(resume_result.artifacts.keys())}")

        # Show final implementation
        final_artifact = resume_result.artifacts.get("g_template")
        if final_artifact:
            click.echo(
                f"\nFinal implementation (source: {final_artifact.source.value}):"
            )
            click.echo("-" * 40)
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
        click.echo(f"    Type: {cp.failure_type.value}")
        click.echo(f"    Created: {cp.created_at[:19]}")
        click.echo()


@cli.command()
def show_config() -> None:
    """Display the workflow.json, prompts.json, and test file."""
    click.echo("=" * 60)
    click.echo("CONFIGURATION FILES")
    click.echo("=" * 60)

    click.echo(f"\n{click.style('workflow.json', fg='cyan', bold=True)}")
    click.echo("-" * 40)
    click.echo(WORKFLOW_PATH.read_text())

    click.echo(f"\n{click.style('prompts.json', fg='cyan', bold=True)}")
    click.echo("-" * 40)
    click.echo(PROMPTS_PATH.read_text())

    test_file = SCRIPT_DIR / "test_template.py"
    if test_file.exists():
        click.echo(f"\n{click.style('test_template.py', fg='cyan', bold=True)}")
        click.echo("-" * 40)
        click.echo(test_file.read_text())


if __name__ == "__main__":
    cli()
