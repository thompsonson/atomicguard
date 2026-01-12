#!/usr/bin/env python3
"""
Agent Design Process Demo: 7-step workflow with checkpoint/resume.

Produces design documentation and Dual-State Action Pair agent skeleton code.

Steps:
1. g_peas: Extract PEAS analysis (Performance, Environment, Actuators, Sensors)
2. g_environment: Classify environment properties (6 dimensions)
3. g_agent_function: Define agent function (percepts, actions, sequences)
4. g_agent_type: Select and justify agent type
5. g_atdd: Generate acceptance criteria (Given-When-Then)
6. g_action_pairs: Design Action Pairs (ρ, a_gen, G)
7. g_implementation: Generate implementation skeleton

Prerequisites:
- Ollama running: ollama serve
- Model available: ollama pull qwen2.5-coder:14b

Usage:
    uv run python -m examples.checkpoint.05_agent_design.demo run --host http://localhost:11434
    # If checkpoint created, edit the artifact file, then:
    uv run python -m examples.checkpoint.05_agent_design.demo resume <checkpoint_id>
"""

import json
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
from atomicguard.infrastructure.persistence.checkpoint import FilesystemCheckpointDAG
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

# Import from base - handle both direct execution and module execution
try:
    from examples.base import (
        find_checkpoint_by_prefix,
        load_prompts,
        load_workflow_config,
        normalize_base_url,
        write_checkpoint_output,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from examples.base import (
        find_checkpoint_by_prefix,
        load_prompts,
        load_workflow_config,
        normalize_base_url,
        write_checkpoint_output,
    )

# Import local generators and guards
from .generators import (
    ActionPairGenerator,
    AgentFunctionGenerator,
    AgentTypeGenerator,
    ATDDGenerator,
    EnvironmentPropertiesGenerator,
    ImplementationGenerator,
    PEASGenerator,
)
from .guards import (
    ActionPairsGuard,
    AgentFunctionGuard,
    AgentTypeGuard,
    ATDDGuard,
    EnvironmentPropertiesGuard,
    ImplementationGuard,
    PEASGuard,
)

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
ARTIFACT_DAG_DIR = OUTPUT_DIR / "artifact_dag"
CHECKPOINT_DAG_DIR = OUTPUT_DIR / "checkpoints"
HUMAN_ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
GENERATED_DIR = OUTPUT_DIR / "generated"
WORKFLOW_PATH = SCRIPT_DIR / "workflow.json"
PROMPTS_PATH = SCRIPT_DIR / "prompts.json"
SAMPLE_INPUT_DIR = SCRIPT_DIR / "sample_input"


# Generator and Guard registries for this example
GENERATOR_REGISTRY = {
    "PEASGenerator": PEASGenerator,
    "EnvironmentPropertiesGenerator": EnvironmentPropertiesGenerator,
    "AgentFunctionGenerator": AgentFunctionGenerator,
    "AgentTypeGenerator": AgentTypeGenerator,
    "ATDDGenerator": ATDDGenerator,
    "ActionPairGenerator": ActionPairGenerator,
    "ImplementationGenerator": ImplementationGenerator,
}

GUARD_REGISTRY = {
    "peas_complete": PEASGuard,
    "environment_classified": EnvironmentPropertiesGuard,
    "agent_function_valid": AgentFunctionGuard,
    "agent_type_justified": AgentTypeGuard,
    "atdd_valid": ATDDGuard,
    "action_pairs_valid": ActionPairsGuard,
    "implementation_valid": ImplementationGuard,
}


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


def load_specification() -> str:
    """Load specification from sample_input files."""
    problem_statement_path = SAMPLE_INPUT_DIR / "problem_statement.md"

    if not problem_statement_path.exists():
        raise click.ClickException(
            f"Problem statement not found: {problem_statement_path}\n"
            "Create a problem_statement.md in the sample_input directory."
        )

    return problem_statement_path.read_text()


def create_workflow(
    artifact_dag: FilesystemArtifactDAG,
    checkpoint_dag: FilesystemCheckpointDAG,
    host: str,
    model: str | None,
) -> ResumableWorkflow:
    """
    Create ResumableWorkflow from workflow.json and prompts.json.

    Uses local generators and guards from this example.
    """
    # Load configuration files
    workflow_config = load_workflow_config(
        WORKFLOW_PATH,
        required_fields=("name", "action_pairs"),
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
    effective_model = model or workflow_config.get("model", "qwen2.5-coder:14b")
    base_url = normalize_base_url(host)

    # Build steps from action_pairs configuration
    action_pairs_config = workflow_config["action_pairs"]

    for step_id, ap_config in action_pairs_config.items():
        # Get generator from local registry
        generator_name = ap_config.get("generator")
        if generator_name not in GENERATOR_REGISTRY:
            raise ValueError(f"Unknown generator: {generator_name}")

        generator_class = GENERATOR_REGISTRY[generator_name]
        generator_config = {
            "model": effective_model,
            "base_url": base_url,
        }
        generator = generator_class(**generator_config)

        # Get guard from local registry
        guard_name = ap_config.get("guard")
        if guard_name not in GUARD_REGISTRY:
            raise ValueError(f"Unknown guard: {guard_name}")

        guard_class = GUARD_REGISTRY[guard_name]
        guard_config = ap_config.get("guard_config", {})
        guard = guard_class(**guard_config)

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


def write_generated_files(impl_artifact_content: str) -> None:
    """Write generated implementation files to output/generated/."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(impl_artifact_content)
        files = data.get("files", [])

        for file_info in files:
            file_path = GENERATED_DIR / file_info["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file_info["content"])
            click.echo(f"  Written: {file_path}")

        # Also write workflow.json if present
        if "workflow_config" in data:
            workflow_path = GENERATED_DIR / "workflow.json"
            workflow_path.write_text(json.dumps(data["workflow_config"], indent=2))
            click.echo(f"  Written: {workflow_path}")

    except (json.JSONDecodeError, KeyError) as e:
        click.echo(f"  Warning: Could not write generated files: {e}")


# =============================================================================
# CLI Commands
# =============================================================================


@click.group()
def cli() -> None:
    """Agent Design Process Demo: 7-step workflow with checkpoint/resume.

    Produces design documentation and Dual-State Action Pair agent code.
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
    """Execute Agent Design Process workflow from problem statement."""
    click.echo("=" * 60)
    click.echo("AGENT DESIGN PROCESS - Run Phase")
    click.echo("=" * 60)
    click.echo("\nThis workflow runs 7 steps to design an agent:")
    click.echo("  1. g_peas: Extract PEAS analysis")
    click.echo("  2. g_environment: Classify environment properties")
    click.echo("  3. g_agent_function: Define agent function")
    click.echo("  4. g_agent_type: Select agent type")
    click.echo("  5. g_atdd: Generate acceptance criteria")
    click.echo("  6. g_action_pairs: Design Action Pairs")
    click.echo("  7. g_implementation: Generate skeleton code")
    click.echo(f"\nLLM Host: {host}")

    # Check if output exists
    if not check_output_exists():
        raise SystemExit(0)

    # Initialize DAGs
    artifact_dag, checkpoint_dag = get_dags()

    # Load workflow config for model info
    workflow_config = load_workflow_config(
        WORKFLOW_PATH,
        required_fields=("name", "action_pairs"),
    )

    effective_model = model or workflow_config.get("model", "qwen2.5-coder:14b")
    click.echo(f"Model: {effective_model}\n")

    # Load specification from sample_input
    try:
        specification = load_specification()
        click.echo(f"Loaded problem statement: {len(specification)} chars")
    except click.ClickException as e:
        click.echo(click.style(f"\n[ERROR] {e.message}", fg="red"))
        raise SystemExit(1) from None

    # Create and execute workflow
    try:
        workflow = create_workflow(artifact_dag, checkpoint_dag, host, model)
    except ImportError as e:
        click.echo(click.style(f"\n[ERROR] {e}", fg="red"))
        click.echo("\nMake sure openai is installed: pip install openai")
        raise SystemExit(1) from None

    click.echo("\nExecuting workflow...")

    try:
        result = workflow.execute(specification)
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
        click.echo("\nAll steps completed successfully!")
        click.echo(f"Artifacts: {list(result.artifacts.keys())}")

        # Show summary of generated content
        for step_id, artifact in result.artifacts.items():
            click.echo(f"\n--- {step_id} ---")
            try:
                data = json.loads(artifact.content)
                if step_id == "g_peas":
                    click.echo(
                        f"  Performance measures: {len(data.get('performance_measures', []))}"
                    )
                    click.echo(f"  Actuators: {len(data.get('actuators', []))}")
                    click.echo(f"  Sensors: {len(data.get('sensors', []))}")
                elif step_id == "g_environment":
                    click.echo(f"  Properties: {len(data.get('properties', []))}")
                    click.echo(
                        f"  Complexity: {data.get('overall_complexity', 'unknown')}"
                    )
                elif step_id == "g_agent_function":
                    click.echo(f"  Percepts: {len(data.get('percepts', []))}")
                    click.echo(f"  Actions: {len(data.get('actions', []))}")
                elif step_id == "g_agent_type":
                    click.echo(
                        f"  Selected type: {data.get('selected_type', 'unknown')}"
                    )
                elif step_id == "g_atdd":
                    click.echo(f"  Scenarios: {len(data.get('scenarios', []))}")
                elif step_id == "g_action_pairs":
                    click.echo(f"  Action pairs: {len(data.get('action_pairs', []))}")
                elif step_id == "g_implementation":
                    click.echo(f"  Files: {len(data.get('files', []))}")
                    click.echo(f"  Steps: {len(data.get('workflow_steps', []))}")
                    # Write generated files
                    click.echo("\nWriting generated files...")
                    write_generated_files(artifact.content)
            except json.JSONDecodeError:
                click.echo(f"  Content: {len(artifact.content)} chars")

        click.echo(f"\n\nGenerated files written to: {GENERATED_DIR}")

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
            resume_command="uv run python -m examples.checkpoint.05_agent_design.demo",
        )

        short_id = checkpoint.checkpoint_id[:12]

        click.echo("\n" + "=" * 60)
        click.echo(click.style("[CHECKPOINT CREATED]", fg="yellow", bold=True))
        click.echo("=" * 60)
        click.echo(f"\nCheckpoint ID: {short_id}")
        click.echo(f"Failed Step: {checkpoint.failed_step}")
        click.echo(f"Failure Type: {checkpoint.failure_type.value}")
        click.echo("\nFeedback:")
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

        click.echo("\n2. Resume the workflow:")
        click.echo(
            f"   {click.style(f'uv run python -m examples.checkpoint.05_agent_design.demo resume {short_id}', fg='cyan')}"
        )

        if verbose:
            click.echo("\n" + "-" * 60)
            click.echo(click.style("CONTEXT:", fg="blue", bold=True))
            click.echo("-" * 60)
            click.echo(f"\nSpecification preview: {checkpoint.specification[:200]}...")

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
    help="Path to artifact file (default: reads from output/artifacts/{step}.json)",
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
    click.echo("AGENT DESIGN PROCESS - Resume Phase")
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
        json_artifact = HUMAN_ARTIFACTS_DIR / f"{checkpoint.failed_step}.json"
        py_artifact = HUMAN_ARTIFACTS_DIR / f"{checkpoint.failed_step}.py"
        if json_artifact.exists():
            artifact = json_artifact
        elif py_artifact.exists():
            artifact = py_artifact
        else:
            artifact = json_artifact  # Will trigger error message

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
        context=f"Human fixed the {checkpoint.failed_step} artifact",
        parent_artifact_id=checkpoint.failed_artifact_id,
        additional_rmax=0,
    )

    # Create workflow and resume
    workflow = create_workflow(artifact_dag, checkpoint_dag, host, model)
    click.echo("\nValidating human artifact...")
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

        # Write generated files if implementation was completed
        if "g_implementation" in resume_result.artifacts:
            click.echo("\nWriting generated files...")
            write_generated_files(resume_result.artifacts["g_implementation"].content)
            click.echo(f"\nGenerated files written to: {GENERATED_DIR}")

    elif resume_result.status == WorkflowStatus.CHECKPOINT:
        new_checkpoint = resume_result.checkpoint
        assert new_checkpoint is not None
        click.echo("\n" + "=" * 60)
        click.echo(click.style("[STILL FAILING]", fg="yellow", bold=True))
        click.echo("=" * 60)
        click.echo(f"\nNew checkpoint created: {new_checkpoint.checkpoint_id[:12]}")
        click.echo("\nFeedback:")
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
    """Display the workflow.json and prompts.json."""
    click.echo("=" * 60)
    click.echo("CONFIGURATION FILES")
    click.echo("=" * 60)

    click.echo(f"\n{click.style('workflow.json', fg='cyan', bold=True)}")
    click.echo("-" * 40)
    click.echo(WORKFLOW_PATH.read_text())

    click.echo(f"\n{click.style('prompts.json', fg='cyan', bold=True)}")
    click.echo("-" * 40)
    click.echo(PROMPTS_PATH.read_text())


@cli.command()
def wizard() -> None:
    """Interactive wizard to create a problem statement."""
    click.echo("=" * 60)
    click.echo("AGENT DESIGN WIZARD")
    click.echo("=" * 60)
    click.echo(
        "\nThis wizard will help you create a problem statement for agent design."
    )

    # Gather information
    click.echo("\n" + "-" * 40)
    goal = click.prompt(
        "\n1. What is the GOAL of the agent?\n   (What should it accomplish?)",
        type=str,
    )

    click.echo("\n" + "-" * 40)
    context = click.prompt(
        "\n2. What is the CONTEXT/ENVIRONMENT?\n   (Where does the agent operate?)",
        type=str,
    )

    click.echo("\n" + "-" * 40)
    requirements = click.prompt(
        "\n3. What are the key REQUIREMENTS?\n   (Comma-separated list)",
        type=str,
    )

    click.echo("\n" + "-" * 40)
    constraints = click.prompt(
        "\n4. What CONSTRAINTS exist?\n   (Performance, technical, etc.)",
        type=str,
        default="None specified",
    )

    # Build problem statement
    req_list = [r.strip() for r in requirements.split(",")]
    req_formatted = "\n".join(f"{i+1}. {r}" for i, r in enumerate(req_list))

    problem_statement = f"""# Problem Statement: {goal.split()[0].title()} Agent

## Goal
{goal}

## Context
{context}

## Requirements
{req_formatted}

## Constraints
{constraints}
"""

    # Show preview
    click.echo("\n" + "=" * 60)
    click.echo("GENERATED PROBLEM STATEMENT")
    click.echo("=" * 60)
    click.echo(problem_statement)

    # Save
    if click.confirm("\nSave this problem statement?", default=True):
        SAMPLE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = SAMPLE_INPUT_DIR / "problem_statement.md"
        output_path.write_text(problem_statement)
        click.echo(f"\nSaved to: {output_path}")
        click.echo("\nNow run:")
        click.echo("  uv run python -m examples.checkpoint.05_agent_design.demo run")
    else:
        click.echo("\nNot saved. Copy the text above if needed.")


if __name__ == "__main__":
    cli()
