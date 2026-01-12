#!/usr/bin/env python3
"""
Enhanced SDLC Workflow Demo with Extension Support.

Extends the checkpoint/04_sdlc workflow with:
- Extension 01: Versioned Environment (W_ref, config_ref)
- Extension 02: Artifact Extraction (predicate-based queries)
- Extension 07: Incremental Execution (skip unchanged steps)

Prerequisites:
- Ollama running: ollama serve
- Model available: ollama pull qwen2.5-coder:14b

Usage:
    # Run with incremental execution (default)
    uv run python -m examples.advanced.sdlc_v2.demo run

    # Run full workflow (ignore cache)
    uv run python -m examples.advanced.sdlc_v2.demo run-full

    # Query artifacts
    uv run python -m examples.advanced.sdlc_v2.demo artifacts --status accepted

    # Resume from checkpoint
    uv run python -m examples.advanced.sdlc_v2.demo resume <checkpoint_id>
"""

import json
import logging
import shutil
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import click

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.checkpoint_service import CheckpointService
from atomicguard.application.resume_service import WorkflowResumeService
from atomicguard.application.workflow import Workflow
from atomicguard.domain.models import (
    AmendmentType,
    Artifact,
    ArtifactSource,
    ArtifactStatus,
    FailureType,
    HumanAmendment,
    WorkflowStatus,
)
from atomicguard.domain.workflow import compute_config_ref, compute_workflow_ref
from atomicguard.infrastructure.persistence.checkpoint import FilesystemCheckpointDAG
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

# Import from base
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

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from examples.base import (
        find_checkpoint_by_prefix,
        load_prompts,
        load_workflow_config,
        normalize_base_url,
        write_checkpoint_output,
    )

# Import local generators and guards
from .generators import (
    ADDGenerator,
    BDDGenerator,
    CoderGenerator,
    ConfigExtractorGenerator,
    IdentityGenerator,
    RulesExtractorGenerator,
)
from .guards import (
    AllTestsPassGuard,
    ArchitectureTestsGuard,
    ArchValidationGuard,
    BDDGuard,
    CompositeValidationGuard,
    ConfigGuard,
    MergeReadyGuard,
    QualityGatesGuard,
    RulesGuard,
)

# Import services (Extensions 02, 07)
from .services import (
    ArtifactExtractionService,
    FileExtractor,
    IncrementalExecutionService,
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
SAMPLE_INPUT_DIR = SCRIPT_DIR / "sample_input"

# Generator and Guard registries
GENERATOR_REGISTRY = {
    "ConfigExtractorGenerator": ConfigExtractorGenerator,
    "ADDGenerator": ADDGenerator,
    "BDDGenerator": BDDGenerator,
    "RulesExtractorGenerator": RulesExtractorGenerator,
    "CoderGenerator": CoderGenerator,
    # Phase 2: Validation pipeline
    "IdentityGenerator": IdentityGenerator,
}

GUARD_REGISTRY = {
    "config_extracted": ConfigGuard,
    "architecture_tests_valid": ArchitectureTestsGuard,
    "scenarios_valid": BDDGuard,
    "rules_valid": RulesGuard,
    "all_tests_pass": AllTestsPassGuard,
    # Phase 2: Validation pipeline
    "quality_gates": QualityGatesGuard,
    "arch_validation": ArchValidationGuard,
    "merge_ready": MergeReadyGuard,
    # Extension 08: Composite guards
    "composite_validation": CompositeValidationGuard,
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
    arch_doc = (SAMPLE_INPUT_DIR / "architecture.md").read_text()
    req_doc = (SAMPLE_INPUT_DIR / "requirements.md").read_text()

    return f"""# Architecture Documentation

{arch_doc}

# Requirements Documentation

{req_doc}
"""


def load_raw_prompts() -> dict:
    """Load raw prompts.json as dict for config_ref computation.

    Unlike load_prompts() which returns PromptTemplate objects,
    this returns the raw JSON dict which is JSON-serializable
    for use with compute_config_ref().
    """
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def create_workflow(
    artifact_dag: FilesystemArtifactDAG,
    host: str,
    model: str | None,
) -> Workflow:
    """Create Workflow from workflow.json and prompts.json.

    Note: checkpoint_dag is no longer passed to Workflow. Checkpointing is now
    handled explicitly via CheckpointService when needed.
    """
    workflow_config = load_workflow_config(
        WORKFLOW_PATH,
        required_fields=("name", "action_pairs"),
    )
    prompts = load_prompts(PROMPTS_PATH)

    workflow = Workflow(
        artifact_dag=artifact_dag,
        rmax=workflow_config.get("rmax", 3),
        constraints=workflow_config.get("constraints", ""),
    )

    effective_model = model or workflow_config.get("model", "qwen2.5-coder:14b")
    base_url = normalize_base_url(host)

    action_pairs_config = workflow_config["action_pairs"]

    for step_id, ap_config in action_pairs_config.items():
        generator_name = ap_config.get("generator")
        if generator_name not in GENERATOR_REGISTRY:
            raise ValueError(f"Unknown generator: {generator_name}")

        generator_class = GENERATOR_REGISTRY[generator_name]
        generator_config = {
            "model": effective_model,
            "base_url": base_url,
        }
        generator = generator_class(**generator_config)

        guard_name = ap_config.get("guard")
        if guard_name not in GUARD_REGISTRY:
            raise ValueError(f"Unknown guard: {guard_name}")

        guard_class = GUARD_REGISTRY[guard_name]
        guard_config = ap_config.get("guard_config", {})
        guard = guard_class(**guard_config)

        prompt_template = prompts.get(step_id)

        action_pair = ActionPair(
            generator=generator,
            guard=guard,
            prompt_template=prompt_template,
        )

        requires = tuple(ap_config.get("requires", []))
        workflow.add_step(
            guard_id=step_id,
            action_pair=action_pair,
            requires=requires,
            deps=requires,
        )

    return workflow


def check_output_exists(force_clean: bool = False) -> bool:
    """Check if output directory exists and prompt user."""
    if not OUTPUT_DIR.exists():
        return True

    if force_clean:
        shutil.rmtree(OUTPUT_DIR)
        click.echo(f"Cleaned: {OUTPUT_DIR}\n")
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
    """Enhanced SDLC Workflow with Extensions.

    Demonstrates Extensions 01, 02, and 07 in a complete SDLC workflow.
    """
    pass


@cli.command()
@click.option("--host", default="http://localhost:11434", help="Ollama host URL")
@click.option(
    "--model", default=None, help="Model to use (default: from workflow.json)"
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option(
    "--incremental/--full", default=True, help="Enable/disable incremental execution"
)
def run(host: str, model: str | None, verbose: bool, incremental: bool) -> None:
    """Execute workflow with incremental support (Extension 07)."""
    # Configure logging based on verbose flag
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Set the sdlc_checkpoint logger specifically (used by generators and guards)
    logging.getLogger("sdlc_checkpoint").setLevel(log_level)
    # Suppress noisy HTTP client logs
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    click.echo("=" * 60)
    click.echo("ENHANCED SDLC WORKFLOW - Extensions 01, 02, 07, 08")
    click.echo("=" * 60)
    click.echo("\nWorkflow steps:")
    click.echo("  Design Phase:")
    click.echo("    1. g_config: Extract project configuration")
    click.echo("    2. g_add: Generate architecture tests (pytest-arch)")
    click.echo("    3. g_bdd: Generate BDD scenarios (Gherkin)")
    click.echo("    4. g_rules: Extract structured architecture rules (deterministic)")
    click.echo("  Implementation Phase (Extension 08: Composite Guards):")
    click.echo("    5. g_coder: Generate + validate implementation")
    click.echo("       (CompositeValidationGuard: syntax → mypy/ruff → pytest-arch)")
    click.echo("  Merge Ready:")
    click.echo("    6. g_merge_ready: Final composite check")
    click.echo(f"\nLLM Host: {host}")
    click.echo(f"Incremental: {incremental}")

    if not check_output_exists():
        raise SystemExit(0)

    artifact_dag, checkpoint_dag = get_dags()

    workflow_config = load_workflow_config(
        WORKFLOW_PATH,
        required_fields=("name", "action_pairs"),
    )
    raw_prompts = load_raw_prompts()  # For JSON-serializable config_ref computation

    effective_model = model or workflow_config.get("model", "qwen2.5-coder:14b")
    click.echo(f"Model: {effective_model}")

    # Extension 01: Compute workflow reference
    w_ref = compute_workflow_ref(workflow_config)
    click.echo(f"W_ref: {w_ref[:16]}...")

    specification = load_specification()
    click.echo(f"Loaded specification: {len(specification)} chars")

    # Extension 07: Initialize incremental service
    incremental_service = IncrementalExecutionService(artifact_dag)

    if incremental:
        # Compute execution plan (use raw_prompts for JSON serialization)
        plan = incremental_service.compute_execution_plan(workflow_config, raw_prompts)

        skipped = [ap for ap, d in plan.items() if not d.should_execute]
        to_execute = [ap for ap, d in plan.items() if d.should_execute]

        if skipped:
            click.echo("\nIncremental execution plan:")
            click.echo(f"  Skip: {skipped}")
            click.echo(f"  Execute: {to_execute}")
        else:
            click.echo("\nNo cached artifacts found - executing full workflow")

    try:
        workflow = create_workflow(artifact_dag, host, model)
    except ImportError as e:
        click.echo(click.style(f"\n[ERROR] {e}", fg="red"))
        click.echo("\nMake sure openai is installed: pip install openai")
        raise SystemExit(1) from None

    # Initialize checkpoint service for explicit checkpoint creation
    checkpoint_service = CheckpointService(checkpoint_dag, artifact_dag)
    workflow_id = str(uuid.uuid4())

    click.echo("\nExecuting workflow...")

    try:
        result = workflow.execute(specification)
    except Exception as e:
        click.echo(click.style(f"\n[ERROR] LLM call failed: {e}", fg="red"))
        click.echo("\nMake sure Ollama is running:")
        click.echo("  ollama serve")
        click.echo(f"  ollama pull {effective_model}")
        raise SystemExit(1) from None

    # Update artifacts with config_ref (Extension 07)
    if result.status == WorkflowStatus.SUCCESS:
        upstream_artifacts = {}
        for step_id in [
            "g_config",
            "g_add",
            "g_bdd",
            "g_rules",
            "g_coder",
            "g_merge_ready",
        ]:
            if step_id in result.artifacts:
                artifact = result.artifacts[step_id]
                # Compute config_ref for this step (use raw_prompts for JSON serialization)
                upstream = dict(upstream_artifacts)
                config_ref = compute_config_ref(
                    step_id, workflow_config, raw_prompts, upstream or None
                )

                # Update artifact with config_ref
                updated = replace(artifact, config_ref=config_ref)
                artifact_dag.store(updated)
                upstream_artifacts[step_id] = updated

    click.echo(f"\nWorkflow Status: {result.status.value}")

    if result.status == WorkflowStatus.SUCCESS:
        click.echo("\n" + "=" * 60)
        click.echo(click.style("[SUCCESS]", fg="green", bold=True))
        click.echo("=" * 60)
        click.echo("\nAll steps completed successfully!")
        click.echo(f"Artifacts: {list(result.artifacts.keys())}")

        for step_id, artifact in result.artifacts.items():
            click.echo(f"\n--- {step_id} ---")
            try:
                data = json.loads(artifact.content)
                if "files" in data:
                    click.echo(f"  Files: {len(data['files'])}")
                elif "tests" in data:
                    click.echo(f"  Tests: {len(data['tests'])}")
                elif "scenarios" in data:
                    click.echo(f"  Scenarios: {len(data['scenarios'])}")
                elif "import_rules" in data:
                    click.echo(
                        f"  Rules: {len(data['import_rules'])} import, {len(data.get('folder_structure', []))} folders"
                    )
                elif "source_root" in data:
                    click.echo(f"  Config: {data.get('source_root')}")
            except json.JSONDecodeError:
                click.echo(f"  Content: {len(artifact.content)} chars")

        # Extract files to filesystem (post-workflow extraction)
        # Guards are sensing-only, so file extraction happens here after all guards pass
        click.echo("\n" + "-" * 60)
        click.echo("Extracting files to filesystem...")

        extractor = FileExtractor(OUTPUT_DIR)

        # Extract implementation files from g_coder
        if "g_coder" in result.artifacts:
            impl_paths = extractor.extract_implementation(
                result.artifacts["g_coder"].content
            )
            click.echo(f"  Implementation: {len(impl_paths)} files written")
            for path in impl_paths[:5]:  # Show first 5
                click.echo(f"    - {path.relative_to(OUTPUT_DIR)}")
            if len(impl_paths) > 5:
                click.echo(f"    ... and {len(impl_paths) - 5} more")

        # Extract architecture tests from g_add
        if "g_add" in result.artifacts:
            test_path = extractor.extract_tests(result.artifacts["g_add"].content)
            click.echo(f"  Tests: {test_path.relative_to(OUTPUT_DIR)}")

        click.echo(f"\nOutput directory: {click.style(str(OUTPUT_DIR), fg='cyan')}")

    elif result.status in (WorkflowStatus.FAILED, WorkflowStatus.ESCALATION):
        # Create checkpoint explicitly via CheckpointService (new API)
        checkpoint = checkpoint_service.create_checkpoint(
            workflow_definition=workflow.get_workflow_definition(),
            workflow_id=workflow_id,
            specification=specification,
            constraints=workflow_config.get("constraints", ""),
            rmax=workflow_config.get("rmax", 3),
            completed_steps=tuple(result.artifacts.keys()),
            artifact_ids=tuple(
                (gid, art.artifact_id) for gid, art in result.artifacts.items()
            ),
            failure_type=(
                FailureType.ESCALATION
                if result.status == WorkflowStatus.ESCALATION
                else FailureType.RMAX_EXHAUSTED
            ),
            failed_step=result.failed_step or "",
            failed_artifact_id=(
                result.escalation_artifact.artifact_id
                if result.escalation_artifact
                else None
            ),
            failure_feedback=result.escalation_feedback or "",
            provenance_ids=tuple(a.artifact_id for a, _ in result.provenance),
        )

        failed_artifact_content = ""
        if checkpoint.failed_artifact_id:
            try:
                failed_artifact = artifact_dag.get_artifact(
                    checkpoint.failed_artifact_id
                )
                failed_artifact_content = failed_artifact.content
            except KeyError:
                failed_artifact_content = "# Failed artifact not found"

        artifact_path = write_checkpoint_output(
            checkpoint=checkpoint,
            failed_artifact_content=failed_artifact_content,
            output_dir=OUTPUT_DIR,
            resume_command="uv run python -m examples.advanced.sdlc_v2.demo",
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

        click.echo("\n" + "-" * 60)
        click.echo(click.style("NEXT STEPS:", fg="green", bold=True))
        click.echo("-" * 60)
        click.echo(f"\n1. Edit: {click.style(str(artifact_path), fg='cyan')}")
        click.echo(
            f"2. Resume: {click.style(f'uv run python -m examples.advanced.sdlc_v2.demo resume {short_id}', fg='cyan')}"
        )


@cli.command("run-full")
@click.option("--host", default="http://localhost:11434", help="Ollama host URL")
@click.option("--model", default=None, help="Model to use")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def run_full(ctx: click.Context, host: str, model: str | None, verbose: bool) -> None:
    """Execute full workflow ignoring cache."""
    # Clean and run with incremental=False
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        click.echo(f"Cleaned: {OUTPUT_DIR}")

    ctx.invoke(run, host=host, model=model, verbose=verbose, incremental=False)


@cli.command()
@click.argument("checkpoint_id")
@click.option("--artifact", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--host", default="http://localhost:11434", help="Ollama host URL")
@click.option("--model", default=None, help="Model to use")
def resume(
    checkpoint_id: str, artifact: Path | None, host: str, model: str | None
) -> None:
    """Resume workflow from checkpoint using edited artifact."""
    click.echo("=" * 60)
    click.echo("ENHANCED SDLC WORKFLOW - Resume Phase")
    click.echo("=" * 60)

    artifact_dag, checkpoint_dag = get_dags()

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

    if artifact is None:
        json_artifact = HUMAN_ARTIFACTS_DIR / f"{checkpoint.failed_step}.json"
        py_artifact = HUMAN_ARTIFACTS_DIR / f"{checkpoint.failed_step}.py"
        if json_artifact.exists():
            artifact = json_artifact
        elif py_artifact.exists():
            artifact = py_artifact
        else:
            artifact = json_artifact

    if not artifact.exists():
        click.echo(
            click.style(f"\n[ERROR] Artifact file not found: {artifact}", fg="red")
        )
        raise SystemExit(1)

    content = artifact.read_text()
    click.echo(f"\nLoaded artifact from: {artifact}")

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

    # Create workflow to get definition and guard for the failed step
    workflow = create_workflow(artifact_dag, host, model)

    # Get the guard for the failed step
    failed_step = workflow.get_step(checkpoint.failed_step)
    guard = failed_step.action_pair.guard

    # Get dependencies for guard validation
    dependencies: dict[str, Artifact] = {}
    for dep_id in failed_step.requires:
        dep_artifact = artifact_dag.get_accepted(dep_id)
        if dep_artifact:
            dependencies[dep_id] = dep_artifact

    # Initialize resume service and resume with W_ref verification
    resume_service = WorkflowResumeService(checkpoint_dag, artifact_dag)
    click.echo("\nValidating human artifact...")

    resume_result = resume_service.resume(
        checkpoint_id=checkpoint.checkpoint_id,
        amendment=amendment,
        current_workflow_definition=workflow.get_workflow_definition(),
        guard=guard,
        dependencies=dependencies,
    )

    if resume_result.success and not resume_result.needs_retry:
        # Human artifact passed guard validation
        click.echo("\n" + "=" * 60)
        click.echo(click.style("[SUCCESS]", fg="green", bold=True))
        click.echo("=" * 60)
        click.echo(f"\nHuman artifact accepted for {checkpoint.failed_step}")
        if resume_result.artifact:
            click.echo(f"Artifact ID: {resume_result.artifact.artifact_id[:12]}")

        # Check if there are remaining steps to execute
        remaining_steps = [
            s
            for s in [
                "g_config",
                "g_add",
                "g_bdd",
                "g_rules",
                "g_coder",
                "g_merge_ready",
            ]
            if s not in checkpoint.completed_steps and s != checkpoint.failed_step
        ]
        if remaining_steps:
            click.echo(f"\nRemaining steps: {remaining_steps}")
            click.echo("Run the workflow again to continue from this point.")

    elif resume_result.success and resume_result.needs_retry:
        # Guard failed or feedback amendment
        if resume_result.guard_result and not resume_result.guard_result.passed:
            click.echo("\n" + "=" * 60)
            click.echo(click.style("[GUARD FAILED]", fg="yellow", bold=True))
            click.echo("=" * 60)
            click.echo("\nHuman artifact did not pass guard validation.")
            click.echo(f"Feedback: {resume_result.guard_result.feedback}")
            click.echo("\nEdit the artifact again and retry.")
        else:
            click.echo(
                "\nFeedback amendment requires agent retry (not implemented in CLI)."
            )

    else:
        # Resume failed (e.g., W_ref mismatch)
        click.echo("\n" + "=" * 60)
        click.echo(click.style("[RESUME FAILED]", fg="red", bold=True))
        click.echo("=" * 60)
        click.echo(f"\nError: {resume_result.error}")


@cli.command()
@click.option("--workflow", "workflow_id", default=None, help="Filter by workflow ID")
@click.option(
    "--status", type=click.Choice(["pending", "accepted", "rejected"]), default=None
)
@click.option("--step", "action_pair_id", default=None, help="Filter by action pair ID")
@click.option("--source", type=click.Choice(["generated", "human"]), default=None)
@click.option("--limit", default=20, help="Maximum results")
def artifacts(
    workflow_id: str | None,
    status: str | None,
    action_pair_id: str | None,
    source: str | None,
    limit: int,
) -> None:
    """Query artifacts in the repository (Extension 02)."""
    click.echo("=" * 60)
    click.echo("ARTIFACT QUERY - Extension 02")
    click.echo("=" * 60)

    if not ARTIFACT_DAG_DIR.exists():
        click.echo("\nNo artifacts found. Run the workflow first.")
        return

    artifact_dag = FilesystemArtifactDAG(str(ARTIFACT_DAG_DIR))
    extraction_service = ArtifactExtractionService(artifact_dag)

    # Build predicates based on filters
    from atomicguard.domain.extraction import (
        ActionPairPredicate,
        AndPredicate,
        SourcePredicate,
        StatusPredicate,
        WorkflowPredicate,
    )

    predicates = []

    if workflow_id:
        predicates.append(WorkflowPredicate(workflow_id))

    if status:
        status_map = {
            "pending": ArtifactStatus.PENDING,
            "accepted": ArtifactStatus.ACCEPTED,
            "rejected": ArtifactStatus.REJECTED,
        }
        predicates.append(StatusPredicate(status_map[status]))

    if action_pair_id:
        predicates.append(ActionPairPredicate(action_pair_id))

    if source:
        source_map = {
            "generated": ArtifactSource.GENERATED,
            "human": ArtifactSource.HUMAN,
        }
        predicates.append(SourcePredicate(source_map[source]))

    # Combine predicates
    predicate = None
    if predicates:
        predicate = predicates[0]
        for p in predicates[1:]:
            predicate = AndPredicate(predicate, p)

    results = extraction_service.query(
        predicate=predicate,
        limit=limit,
        order_by="-created_at",
    )

    if not results:
        click.echo("\nNo artifacts match the query.")
        return

    click.echo(f"\nFound {len(results)} artifact(s):\n")

    for artifact in results:
        status_color = {
            ArtifactStatus.PENDING: "yellow",
            ArtifactStatus.ACCEPTED: "green",
            ArtifactStatus.REJECTED: "red",
        }.get(artifact.status, "white")

        click.echo(f"  {click.style(artifact.artifact_id[:12], fg='cyan')}")
        click.echo(f"    Step: {artifact.action_pair_id}")
        click.echo(f"    Status: {click.style(artifact.status.value, fg=status_color)}")
        click.echo(f"    Source: {artifact.source.value}")
        click.echo(f"    Created: {artifact.created_at[:19]}")
        if artifact.config_ref:
            click.echo(f"    Config Ref: {artifact.config_ref[:16]}...")
        click.echo()

    # Show summary
    counts = extraction_service.count_by_status()
    click.echo("Summary:")
    for s, count in counts.items():
        click.echo(f"  {s.value}: {count}")


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


if __name__ == "__main__":
    cli()
