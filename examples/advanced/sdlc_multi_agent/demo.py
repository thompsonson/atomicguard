#!/usr/bin/env python3
"""
Multi-Agent SDLC Workflow - Proof of Concept.

Three-agent workflow demonstrating AtomicGuard's multi-agent coordination:
- DDD Agent: Generate domain documentation
- Coder Agent: Generate implementation from docs
- Tester Agent: Validate tests pass

Key design:
- DAG is the shared source of truth
- Workspaces are ephemeral but persisted for debugging
- Clear separation of concerns (WorkspaceService, Generators, Guards, Orchestrator)

Usage:
    # Run workflow
    uv run python -m examples.advanced.sdlc_multi_agent.demo run

    # List workspaces
    uv run python -m examples.advanced.sdlc_multi_agent.demo list-workspaces

    # Clean workspaces only
    uv run python -m examples.advanced.sdlc_multi_agent.demo clean-workspaces

    # Clean everything (workspaces + artifacts)
    uv run python -m examples.advanced.sdlc_multi_agent.demo clean
"""

import asyncio
import json
import shutil
from pathlib import Path

import click

from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

from .generators import CoderGenerator, DDDGenerator, IdentityGenerator
from .guards import AllTestsPassGuard, DocumentationGuard
from .orchestrator import SDLCOrchestrator
from .workspace_service import WorkspaceService

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
ARTIFACT_DAG_DIR = OUTPUT_DIR / "artifact_dag"
WORKSPACES_DIR = OUTPUT_DIR / "workspaces"
WORKFLOW_PATH = SCRIPT_DIR / "workflow.json"
PROMPTS_PATH = SCRIPT_DIR / "prompts.json"
SAMPLE_INPUT = SCRIPT_DIR / "sample_input" / "requirements.md"


# =============================================================================
# Helper Functions
# =============================================================================


def load_config() -> dict:
    """Load workflow configuration."""
    with open(WORKFLOW_PATH) as f:
        return json.load(f)


def load_prompts() -> dict:
    """Load prompts."""
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def initialize_services(config: dict) -> tuple:
    """Initialize all services.

    Returns:
        Tuple of (artifact_dag, workspace_service, orchestrator)
    """
    # Create directories
    ARTIFACT_DAG_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize DAG (source of truth)
    artifact_dag = FilesystemArtifactDAG(str(ARTIFACT_DAG_DIR))

    # Initialize workspace service
    workspace_service = WorkspaceService(
        base_dir=WORKSPACES_DIR,
        persist=config.get("workspace_config", {}).get("persist", True),
    )

    # Initialize generators
    model = config.get("model", "qwen2.5-coder:14b")
    base_url = "http://localhost:11434/v1"

    generators = {
        "g_ddd": DDDGenerator(model=model, base_url=base_url),
        "g_coder": CoderGenerator(model=model, base_url=base_url),
        "g_tester": IdentityGenerator(),
    }

    # Initialize guards
    guards = {
        "g_ddd": DocumentationGuard(),
        "g_coder": AllTestsPassGuard(timeout=60),
        "g_tester": AllTestsPassGuard(timeout=60),
    }

    # Load prompts
    prompts = load_prompts()

    # Initialize orchestrator
    orchestrator = SDLCOrchestrator(
        artifact_dag=artifact_dag,
        workspace_service=workspace_service,
        generators=generators,
        guards=guards,
        prompts=prompts,
    )

    return artifact_dag, workspace_service, orchestrator


# =============================================================================
# CLI Commands
# =============================================================================


@click.group()
def cli() -> None:
    """Multi-Agent SDLC Workflow - Proof of Concept.

    Demonstrates three-agent coordination: DDD → Coder → Tester
    """
    pass


@cli.command()
@click.option("--input", "input_file", type=click.Path(exists=True), default=None)
@click.option("--host", default="http://localhost:11434", help="Ollama host URL")
@click.option("--model", default=None, help="Model to use")
def run(input_file: str | None, host: str, model: str | None) -> None:
    """Execute the Multi-Agent SDLC workflow."""
    click.echo("=" * 60)
    click.echo("MULTI-AGENT SDLC WORKFLOW - Proof of Concept")
    click.echo("=" * 60)
    click.echo("\nThree-phase workflow:")
    click.echo("  1. DDD Agent: Generate domain documentation")
    click.echo("  2. Coder Agent: Generate implementation from docs")
    click.echo("  3. Tester Agent: Validate tests pass")
    click.echo(f"\nLLM Host: {host}")

    # Load configuration
    config = load_config()
    if model:
        config["model"] = model
    click.echo(f"Model: {config['model']}")
    click.echo(f"Total Retry Budget: {config['total_retry_budget']}")

    # Load user intent
    if input_file:
        user_intent = Path(input_file).read_text()
    elif SAMPLE_INPUT.exists():
        user_intent = SAMPLE_INPUT.read_text()
    else:
        click.echo(
            click.style(
                "\n[ERROR] No input file found. Create sample_input/requirements.md",
                fg="red",
            )
        )
        raise SystemExit(1)

    click.echo(f"\nUser Intent: {len(user_intent)} chars")

    # Initialize services
    artifact_dag, workspace_service, orchestrator = initialize_services(config)

    # Execute workflow
    click.echo("\n" + "=" * 60)
    click.echo("EXECUTING WORKFLOW")
    click.echo("=" * 60)

    try:
        result = asyncio.run(orchestrator.execute(user_intent))
    except Exception as e:
        click.echo(click.style(f"\n[ERROR] Workflow failed: {e}", fg="red"))
        click.echo("\nMake sure Ollama is running:")
        click.echo("  ollama serve")
        click.echo(f"  ollama pull {config['model']}")
        raise SystemExit(1) from None

    # Display results
    click.echo("\n" + "=" * 60)
    if result.success:
        click.echo(click.style("[SUCCESS]", fg="green", bold=True))
    else:
        click.echo(click.style("[FAILED]", fg="red", bold=True))
    click.echo("=" * 60)

    click.echo(f"\nCompleted Phases: {result.completed_phases}")
    if result.failed_phase:
        click.echo(f"Failed Phase: {click.style(result.failed_phase, fg='red')}")

    click.echo(f"\nAttempts Used: {result.total_attempts} / {config['total_retry_budget']}")
    click.echo(f"Budget Remaining: {result.budget_remaining}")

    # Show phase details
    click.echo("\n" + "-" * 60)
    click.echo("PHASE DETAILS")
    click.echo("-" * 60)

    for phase_id, phase_result in result.phase_results.items():
        status_symbol = "✓" if phase_result.success else "✗"
        status_color = "green" if phase_result.success else "red"
        click.echo(
            f"\n{status_symbol} {phase_id}: {click.style(('SUCCESS' if phase_result.success else 'FAILED'), fg=status_color)}"
        )
        click.echo(f"  Attempts: {phase_result.attempts}")
        if phase_result.artifact_id:
            click.echo(f"  Artifact ID: {phase_result.artifact_id[:16]}...")
        if phase_result.feedback:
            click.echo(f"  Feedback: {phase_result.feedback[:200]}...")

    # Show workspace locations
    click.echo("\n" + "-" * 60)
    click.echo("OUTPUT LOCATIONS")
    click.echo("-" * 60)
    click.echo(f"Artifacts: {click.style(str(ARTIFACT_DAG_DIR), fg='cyan')}")
    click.echo(f"Workspaces: {click.style(str(WORKSPACES_DIR), fg='cyan')}")

    if result.success:
        click.echo("\n" + "-" * 60)
        click.echo("NEXT STEPS")
        click.echo("-" * 60)
        click.echo("1. Explore workspaces:")
        click.echo(f"   ls {WORKSPACES_DIR}")
        click.echo("2. View generated code:")
        click.echo(f"   ls {WORKSPACES_DIR}/g_coder_attempt_*/src/")
        click.echo("3. List all workspaces:")
        click.echo("   uv run python -m examples.advanced.sdlc_multi_agent.demo list-workspaces")


@cli.command("list-workspaces")
def list_workspaces() -> None:
    """List all persisted workspaces."""
    click.echo("=" * 60)
    click.echo("PERSISTED WORKSPACES")
    click.echo("=" * 60)

    if not WORKSPACES_DIR.exists():
        click.echo("\nNo workspaces directory found.")
        click.echo("Run the workflow first: uv run python -m examples.advanced.sdlc_multi_agent.demo run")
        return

    workspaces = sorted(WORKSPACES_DIR.iterdir())

    if not workspaces:
        click.echo("\nNo workspaces found.")
        return

    click.echo(f"\nFound {len(workspaces)} workspace(s):\n")

    for workspace in workspaces:
        size_mb = sum(f.stat().st_size for f in workspace.rglob("*") if f.is_file()) / (1024 * 1024)
        file_count = len(list(workspace.rglob("*.py"))) + len(list(workspace.rglob("*.md")))

        click.echo(f"  {click.style(workspace.name, fg='cyan')}")
        click.echo(f"    Path: {workspace}")
        click.echo(f"    Size: {size_mb:.2f} MB")
        click.echo(f"    Files: {file_count} (*.py, *.md)")
        click.echo()


@cli.command("clean-workspaces")
@click.confirmation_option(prompt="Are you sure you want to delete all workspaces?")
def clean_workspaces() -> None:
    """Delete all persisted workspaces."""
    if not WORKSPACES_DIR.exists():
        click.echo("No workspaces directory found.")
        return

    workspaces = list(WORKSPACES_DIR.iterdir())
    count = len(workspaces)

    for workspace in workspaces:
        shutil.rmtree(workspace)

    click.echo(f"Deleted {count} workspace(s): {WORKSPACES_DIR}")


@cli.command()
@click.confirmation_option(
    prompt="Are you sure you want to delete all output (workspaces + artifacts)?"
)
def clean() -> None:
    """Delete all output (workspaces and artifacts)."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        click.echo(f"Deleted: {OUTPUT_DIR}")
    else:
        click.echo("Output directory does not exist.")


if __name__ == "__main__":
    cli()
