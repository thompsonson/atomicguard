"""Demo CLI for SWE-bench ablation study.

Uses the core AtomicGuard Workflow class for orchestration.

Usage:
    python -m examples.swe_bench_ablation.demo run --variant 01_baseline --problem astropy__astropy-12907
"""

import json
import logging
from pathlib import Path
from typing import Any

import click

from atomicguard import ActionPair, Workflow, WorkflowStatus
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

from .generators import (
    AnalysisGenerator,
    LocalizationGenerator,
    PatchGenerator,
    TestGenerator,
)
from .guards import AnalysisGuard, LocalizationGuard, PatchGuard, TestSyntaxGuard

logger = logging.getLogger("swe_bench_ablation")

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
ARTIFACT_DAG_DIR = OUTPUT_DIR / "artifact_dag"


# =============================================================================
# Registry
# =============================================================================


def get_generator_registry() -> dict[str, type]:
    """Get generator class registry."""
    return {
        "AnalysisGenerator": AnalysisGenerator,
        "LocalizationGenerator": LocalizationGenerator,
        "PatchGenerator": PatchGenerator,
        "TestGenerator": TestGenerator,
    }


def get_guard_registry() -> dict[str, type]:
    """Get guard class registry."""
    return {
        "analysis": AnalysisGuard,
        "localization": LocalizationGuard,
        "patch": PatchGuard,
        "test_syntax": TestSyntaxGuard,
    }


# =============================================================================
# Workflow Loading
# =============================================================================


def load_workflow_config(variant: str) -> dict[str, Any]:
    """Load workflow configuration from JSON file."""
    workflow_dir = Path(__file__).parent / "workflows"
    workflow_file = workflow_dir / f"{variant}.json"

    if not workflow_file.exists():
        raise FileNotFoundError(f"Workflow not found: {workflow_file}")

    return json.loads(workflow_file.read_text())


def load_prompts() -> dict[str, PromptTemplate]:
    """Load prompt templates from prompts.json."""
    prompts_file = Path(__file__).parent / "prompts.json"

    if not prompts_file.exists():
        return {}

    data = json.loads(prompts_file.read_text())

    templates = {}
    for key, value in data.items():
        templates[key] = PromptTemplate(
            role=value.get("role", ""),
            constraints=value.get("constraints", ""),
            task=value.get("task", ""),
            feedback_wrapper=value.get("feedback_wrapper", "Feedback: {feedback}"),
        )

    return templates


def build_workflow(
    config: dict[str, Any],
    prompts: dict[str, PromptTemplate],
    model: str,
    base_url: str,
    artifact_dag: FilesystemArtifactDAG,
    repo_root: str | None = None,
    api_key: str = "ollama",
    provider: str = "ollama",
) -> Workflow:
    """Build a Workflow from configuration.

    Args:
        config: Workflow configuration dict
        prompts: Prompt templates
        model: LLM model to use
        base_url: Ollama API base URL
        artifact_dag: Persistent artifact storage
        repo_root: Repository root for file validation
        provider: LLM provider identifier

    Returns:
        Configured Workflow instance
    """
    generator_registry = get_generator_registry()
    guard_registry = get_guard_registry()

    rmax = config.get("rmax", 3)
    workflow = Workflow(artifact_dag=artifact_dag, rmax=rmax)

    action_pairs = config.get("action_pairs", {})

    # Sort by dependencies (topological sort)
    sorted_pairs = _topological_sort(action_pairs)

    for ap_id in sorted_pairs:
        ap_config = action_pairs[ap_id]

        # Get generator class
        gen_name = ap_config["generator"]
        if gen_name not in generator_registry:
            raise ValueError(f"Unknown generator: {gen_name}")
        gen_cls = generator_registry[gen_name]

        # Get guard class
        guard_name = ap_config["guard"]
        if guard_name not in guard_registry:
            raise ValueError(f"Unknown guard: {guard_name}")
        guard_cls = guard_registry[guard_name]

        # Build generator with model config
        gen_kwargs: dict[str, Any] = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "provider": provider,
        }
        # AnalysisGenerator needs repo_root for code-aware analysis.
        if repo_root and issubclass(gen_cls, AnalysisGenerator):
            gen_kwargs["repo_root"] = repo_root
        # Patch generators need repo_root to produce unified diffs.
        if repo_root and issubclass(gen_cls, PatchGenerator):
            gen_kwargs["repo_root"] = repo_root
        generator = gen_cls(**gen_kwargs)

        # Build guard with config
        guard_config = ap_config.get("guard_config", {})
        if repo_root:
            guard_config["repo_root"] = repo_root
        guard = guard_cls(**guard_config)

        # Get prompt template
        template = prompts.get(ap_id)

        # Create action pair
        action_pair = ActionPair(
            generator=generator,
            guard=guard,
            prompt_template=template,
        )

        # Get dependencies
        requires = tuple(ap_config.get("requires", []))

        # Extension 09: Backtracking parameters
        r_patience = ap_config.get("r_patience")
        e_max = ap_config.get("e_max", 1)
        escalation = tuple(ap_config.get("escalation", []))

        # Extension 09: Guard-specific escalation routing
        raw_ebg = ap_config.get("escalation_by_guard")
        escalation_by_guard = (
            {k: tuple(v) for k, v in raw_ebg.items()} if raw_ebg else None
        )

        # Add step to workflow
        workflow.add_step(
            ap_id,
            action_pair,
            requires=requires,
            r_patience=r_patience,
            e_max=e_max,
            escalation=escalation,
            escalation_by_guard=escalation_by_guard,
        )

    return workflow


def _topological_sort(action_pairs: dict[str, Any]) -> list[str]:
    """Sort action pairs by dependencies."""
    result: list[str] = []
    visited: set[str] = set()

    def visit(ap_id: str) -> None:
        if ap_id in visited:
            return
        visited.add(ap_id)

        ap_config = action_pairs.get(ap_id, {})
        for dep in ap_config.get("requires", []):
            visit(dep)

        result.append(ap_id)

    for ap_id in action_pairs:
        visit(ap_id)

    return result


# =============================================================================
# CLI Commands
# =============================================================================


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool) -> None:
    """SWE-bench ablation study CLI."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@cli.command()
@click.option("--variant", default="01_baseline", help="Workflow variant")
@click.option(
    "--problem", required=True, help="Problem ID (e.g., astropy__astropy-12907)"
)
@click.option("--model", default="qwen2.5-coder:14b", help="LLM model")
@click.option(
    "--provider",
    default="ollama",
    help="LLM provider (ollama, openrouter, huggingface, openai)",
)
@click.option("--host", default="http://localhost:11434/v1", help="Ollama API URL")
@click.option("--repo-root", default=None, help="Repository root path")
def run(
    variant: str,
    problem: str,
    model: str,
    provider: str,
    host: str,
    repo_root: str | None,
) -> None:
    """Run a single problem with a workflow variant."""
    click.echo(f"Running {variant} on {problem}")

    # Load configuration
    config = load_workflow_config(variant)
    prompts = load_prompts()

    # Initialize persistent artifact DAG
    ARTIFACT_DAG_DIR.mkdir(parents=True, exist_ok=True)
    artifact_dag = FilesystemArtifactDAG(str(ARTIFACT_DAG_DIR))
    click.echo(f"Artifact storage: {ARTIFACT_DAG_DIR}")

    # Build workflow
    workflow = build_workflow(
        config, prompts, model, host, artifact_dag, repo_root, provider=provider
    )

    # For now, use a simple problem statement
    # In a real implementation, this would load from SWE-bench dataset
    specification = f"Fix the bug described in issue {problem}"

    # Execute workflow
    result = workflow.execute(specification)

    # Report results
    if result.status == WorkflowStatus.SUCCESS:
        click.echo(click.style("✓ Success!", fg="green"))
        for guard_id, artifact in result.artifacts.items():
            click.echo(f"  {guard_id}: {artifact.artifact_id[:8]}...")
    else:
        click.echo(click.style(f"✗ Failed at {result.failed_step}", fg="red"))
        if result.escalation_feedback:
            click.echo(f"  Feedback: {result.escalation_feedback}")


@cli.command()
def list_workflows() -> None:
    """List available workflow variants."""
    workflow_dir = Path(__file__).parent / "workflows"

    click.echo("Available workflows:")
    for f in sorted(workflow_dir.glob("*.json")):
        config = json.loads(f.read_text())
        name = config.get("name", f.stem)
        desc = config.get("description", "")
        click.echo(f"  {f.stem}: {name}")
        if desc:
            click.echo(f"    {desc}")


@cli.command()
@click.option(
    "--model",
    default="Qwen/Qwen2.5-Coder-32B-Instruct",
    help="HuggingFace model ID",
)
@click.option(
    "--provider",
    default="ollama",
    help="LLM provider (ollama, openrouter, huggingface, openai)",
)
@click.option(
    "--arms",
    default="singleshot,s1_direct,s1_tdd",
    help="Comma-separated arm names to run",
)
@click.option(
    "--output-dir",
    default="output/experiment_7_2",
    help="Directory for experiment results",
)
@click.option("--split", default="test", help="Dataset split")
@click.option("--max-instances", default=0, type=int, help="Max instances (0=all)")
@click.option("--resume", is_flag=True, help="Resume from existing results")
def experiment(
    model: str,
    provider: str,
    arms: str,
    output_dir: str,
    split: str,
    max_instances: int,
    resume: bool,
) -> None:
    """Run Experiment 7.2: Bug Fix Strategy Comparison on SWE-PolyBench."""
    from .experiment_runner import ExperimentRunner

    arm_map = {
        "singleshot": "02_singleshot",
        "s1_direct": "03_s1_direct",
        "s1_tdd": "04_s1_tdd",
    }
    arm_list = [arm_map[a.strip()] for a in arms.split(",") if a.strip() in arm_map]

    if not arm_list:
        click.echo(click.style("No valid arms specified", fg="red"))
        return

    click.echo(f"Running Experiment 7.2 with model={model}")
    click.echo(f"Arms: {arm_list}")
    click.echo(f"Output: {output_dir}")

    runner = ExperimentRunner(
        model=model,
        provider=provider,
        output_dir=output_dir,
    )
    runner.run_all(
        arms=arm_list,
        split=split,
        max_instances=max_instances if max_instances > 0 else None,
        resume_from=output_dir if resume else None,
    )


@cli.command()
@click.option(
    "--results",
    default="output/experiment_7_2/results.jsonl",
    help="Path to results.jsonl file",
)
@click.option(
    "--resolved",
    default=None,
    help="Path to resolved.json (instance_id -> bool) from swebench evaluation",
)
@click.option(
    "--output-dir",
    default="output/experiment_7_2",
    help="Directory for visualization output",
)
def visualize(results: str, resolved: str | None, output_dir: str) -> None:
    """Generate visualizations from experiment results."""
    from .analysis import generate_visualizations, load_results

    click.echo(f"Loading results from {results}")
    arm_results = load_results(results)

    if not arm_results:
        click.echo(click.style("No results found", fg="red"))
        return

    click.echo(f"Loaded {len(arm_results)} results")

    resolved_map: dict[str, bool] | None = None
    if resolved:
        resolved_path = Path(resolved)
        if resolved_path.exists():
            resolved_map = json.loads(resolved_path.read_text())
            click.echo(f"Loaded {len(resolved_map or {})} resolved entries")
        else:
            click.echo(click.style(f"Resolved file not found: {resolved}", fg="yellow"))

    paths = generate_visualizations(arm_results, resolved_map, output_dir)

    click.echo(click.style(f"Generated {len(paths)} visualizations:", fg="green"))
    for p in paths:
        click.echo(f"  {p}")


if __name__ == "__main__":
    cli()
