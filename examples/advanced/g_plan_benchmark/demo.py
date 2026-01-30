#!/usr/bin/env python3
"""
G_plan Validation Benchmark (ISMIS 2026).

Validates the G_plan taxonomy (Minimal/Medium/Expansive) against
multi-agent SDLC plans with injected defects.

All validation uses real AtomicGuard GuardInterface implementations.
Plans are loaded from the catalog via a deterministic PlanGenerator,
or generated via LLM using LLMPlanGenerator (epsilon estimation mode).

Usage:
    # Validate a plan at all rigor levels
    uv run python -m examples.advanced.g_plan_benchmark.demo validate

    # Run defect detection benchmark
    uv run python -m examples.advanced.g_plan_benchmark.demo benchmark

    # Run complexity scaling benchmark
    uv run python -m examples.advanced.g_plan_benchmark.demo complexity

    # Load plan from real sdlc_v2 workflow.json
    uv run python -m examples.advanced.g_plan_benchmark.demo validate --from-workflow

    # Estimate epsilon for LLM plan generation
    uv run python -m examples.advanced.g_plan_benchmark.demo epsilon --trials 20

    # Epsilon with specific model/host
    uv run python -m examples.advanced.g_plan_benchmark.demo epsilon \\
        --trials 20 --host http://localhost:11434 --model qwen2.5-coder:14b
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import click
from rich.console import Console
from rich.table import Table

from atomicguard.domain.models import (
    AmbientEnvironment,
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    GuardResult,
)
from atomicguard.domain.prompts import PromptTemplate

from .defects import DefectType, inject_defect
from .generators import LLMJsonGenerator, LLMPlanGenerator
from .guards import (
    AnalysisGuard,
    ExpansivePlanGuard,
    MediumPlanGuard,
    MinimalPlanGuard,
    ReconGuard,
    StrategyGuard,
)
from .models import PlanDefinition

console = Console()

SCRIPT_DIR = Path(__file__).parent
PLANS_DIR = SCRIPT_DIR / "plans"
PROMPTS_PATH = SCRIPT_DIR / "prompts.json"
SDLC_V2_WORKFLOW = SCRIPT_DIR.parent / "sdlc_v2" / "workflow.json"
SDLC_V2_SAMPLE_INPUT = SCRIPT_DIR.parent / "sdlc_v2" / "sample_input"


# =============================================================================
# HELPERS
# =============================================================================


def _make_artifact(content: str, action_pair_id: str = "g_plan") -> Artifact:
    """Wrap plan JSON as an Artifact for guard validation."""
    return Artifact(
        artifact_id=str(uuid4()),
        workflow_id="benchmark",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id=action_pair_id,
        created_at="2026-01-30T00:00:00",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=ContextSnapshot(
            workflow_id="benchmark",
            specification="G_plan benchmark",
            constraints="",
            feedback_history=(),
        ),
    )


def _load_plan(plan_source: str) -> dict[str, Any]:
    """Load a plan dict from the catalog or workflow.json."""
    plan_file = PLANS_DIR / f"{plan_source}.json"
    if plan_file.exists():
        with open(plan_file) as f:
            return json.load(f)
    raise FileNotFoundError(f"Plan not found: {plan_file}")


def _load_from_workflow() -> dict[str, Any]:
    """Load plan from the real sdlc_v2 workflow.json."""
    plan = PlanDefinition.from_workflow_json(SDLC_V2_WORKFLOW)
    return plan.to_dict()


def _load_specification() -> str:
    """Load specification from sdlc_v2 sample_input files."""
    arch_path = SDLC_V2_SAMPLE_INPUT / "architecture.md"
    req_path = SDLC_V2_SAMPLE_INPUT / "requirements.md"

    parts = []
    if arch_path.exists():
        parts.append(f"# Architecture Documentation\n\n{arch_path.read_text()}")
    if req_path.exists():
        parts.append(f"# Requirements Documentation\n\n{req_path.read_text()}")

    return "\n\n".join(parts) if parts else "Design a multi-agent SDLC workflow."


def _load_prompt_template(step_id: str) -> PromptTemplate:
    """Load a PromptTemplate from prompts.json."""
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    entry = prompts[step_id]
    return PromptTemplate(
        role=entry["role"],
        constraints=entry["constraints"],
        task=entry["task"],
        feedback_wrapper=entry["feedback_wrapper"],
    )


def _make_context(specification: str) -> Context:
    """Build a Context for the LLM generator."""
    return Context(
        ambient=AmbientEnvironment(repository=None, constraints=""),
        specification=specification,
        current_artifact=None,
        feedback_history=(),
        dependency_artifacts=(),
    )


def _generate_scaled_plan(num_steps: int) -> dict[str, Any]:
    """Generate a valid linear plan with specified number of steps."""
    guard_choices = [
        "syntax",
        "dynamic_test",
        "config_extracted",
        "architecture_tests_valid",
        "scenarios_valid",
    ]
    steps = []
    prev_effect = "start"

    for i in range(num_steps):
        effect = f"token_{i}"
        steps.append(
            {
                "step_id": f"step_{i}",
                "name": f"Step {i}",
                "generator": "OllamaGenerator",
                "guard": random.choice(guard_choices),
                "retry_budget": random.randint(1, 5),
                "preconditions": [prev_effect],
                "effects": [effect],
                "dependencies": [f"step_{i - 1}"] if i > 0 else [],
            }
        )
        prev_effect = effect

    return {
        "plan_id": f"scaled-{num_steps}",
        "initial_state": ["start"],
        "goal_state": [prev_effect],
        "total_retry_budget": num_steps * 5,
        "steps": steps,
    }


def _wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (95% CI by default)."""
    if trials == 0:
        return (0.0, 0.0)
    p_hat = successes / trials
    denom = 1 + z * z / trials
    centre = (p_hat + z * z / (2 * trials)) / denom
    spread = (
        z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * trials)) / trials) / denom
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# =============================================================================
# BENCHMARK DATA STRUCTURES
# =============================================================================


@dataclass
class DefectDetectionResult:
    """Result of running one defect type through all rigor levels."""

    defect_type: str
    minimal_detected: bool
    minimal_time_ms: float
    medium_detected: bool
    medium_time_ms: float
    expansive_detected: bool
    expansive_time_ms: float


@dataclass
class EpsilonTrialResult:
    """Result of a single LLM plan generation + validation trial."""

    trial: int
    minimal_passed: bool
    medium_passed: bool
    expansive_passed: bool
    generation_time_ms: float
    plan_steps: int
    errors: list[str]
    plan_content: str = ""
    minimal_feedback: str = ""
    medium_feedback: str = ""
    expansive_feedback: str = ""
    # Decomposed pipeline fields (None = step not run)
    analysis_passed: bool | None = None
    analysis_content: str = ""
    analysis_feedback: str = ""
    analysis_time_ms: float = 0.0
    recon_passed: bool | None = None
    recon_content: str = ""
    recon_feedback: str = ""
    recon_time_ms: float = 0.0
    strategy_passed: bool | None = None
    strategy_content: str = ""
    strategy_feedback: str = ""
    strategy_time_ms: float = 0.0


# =============================================================================
# CLI
# =============================================================================


@click.group()
def cli() -> None:
    """G_plan Validation Benchmark for ISMIS 2026 paper."""
    pass


@cli.command()
@click.option(
    "--plan",
    "plan_source",
    default="sdlc_v2",
    type=click.Choice(["sdlc_v2", "simple"]),
    help="Plan variant from catalog",
)
@click.option(
    "--from-workflow",
    is_flag=True,
    help="Load plan from real sdlc_v2/workflow.json instead of catalog",
)
def validate(plan_source: str, from_workflow: bool) -> None:
    """Validate a plan at all three rigor levels."""
    console.print("\n[bold]G_plan Validation Benchmark[/bold]")

    if from_workflow:
        if not SDLC_V2_WORKFLOW.exists():
            console.print(f"[red]workflow.json not found: {SDLC_V2_WORKFLOW}[/red]")
            raise SystemExit(1)
        plan_dict = _load_from_workflow()
        console.print(f"Loaded from: {SDLC_V2_WORKFLOW}")
    else:
        plan_dict = _load_plan(plan_source)
        console.print(f"Loaded from catalog: {plan_source}")

    plan_json = json.dumps(plan_dict, indent=2)
    artifact = _make_artifact(plan_json)

    plan = PlanDefinition.from_dict(plan_dict)
    console.print(f"Plan: {plan.plan_id} ({len(plan.steps)} steps)")

    guards = [
        ("Minimal", MinimalPlanGuard()),
        ("Medium", MediumPlanGuard()),
        ("Expansive", ExpansivePlanGuard()),
    ]

    for level_name, guard in guards:
        start = time.perf_counter()
        result: GuardResult = guard.validate(artifact)
        elapsed_ms = (time.perf_counter() - start) * 1000

        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(f"\n{level_name}: {status} ({elapsed_ms:.3f}ms)")
        if result.feedback:
            console.print(f"  {result.feedback}")


@cli.command()
@click.option(
    "--plan",
    "plan_source",
    default="sdlc_v2",
    type=click.Choice(["sdlc_v2", "simple"]),
    help="Base plan to inject defects into",
)
@click.option("--trials", default=100, help="Trials per defect type")
@click.option("--output", default=None, help="Output JSON file")
@click.option(
    "--from-workflow",
    is_flag=True,
    help="Load base plan from real sdlc_v2/workflow.json",
)
def benchmark(
    plan_source: str,
    trials: int,
    output: str | None,
    from_workflow: bool,
) -> None:
    """Run defect detection benchmark across all rigor levels."""
    console.print("\n[bold]G_plan Defect Detection Benchmark[/bold]")

    if from_workflow:
        base_plan_dict = _load_from_workflow()
        console.print("Base plan: sdlc_v2/workflow.json")
    else:
        base_plan_dict = _load_plan(plan_source)
        console.print(f"Base plan: {plan_source}")

    plan = PlanDefinition.from_dict(base_plan_dict)
    console.print(f"Steps: {len(plan.steps)}, Trials: {trials}")

    minimal_guard = MinimalPlanGuard()
    medium_guard = MediumPlanGuard()
    expansive_guard = ExpansivePlanGuard()

    results: list[DefectDetectionResult] = []

    for defect_type in DefectType:
        minimal_detections = 0
        medium_detections = 0
        expansive_detections = 0
        minimal_times: list[float] = []
        medium_times: list[float] = []
        expansive_times: list[float] = []

        for _ in range(trials):
            defective = inject_defect(base_plan_dict, defect_type)
            content = json.dumps(defective, indent=2)
            artifact = _make_artifact(content)

            # Minimal
            t0 = time.perf_counter()
            min_r = minimal_guard.validate(artifact)
            minimal_times.append((time.perf_counter() - t0) * 1000)
            if not min_r.passed:
                minimal_detections += 1

            # Medium
            t0 = time.perf_counter()
            med_r = medium_guard.validate(artifact)
            medium_times.append((time.perf_counter() - t0) * 1000)
            if not med_r.passed:
                medium_detections += 1

            # Expansive
            t0 = time.perf_counter()
            exp_r = expansive_guard.validate(artifact)
            expansive_times.append((time.perf_counter() - t0) * 1000)
            if not exp_r.passed:
                expansive_detections += 1

        results.append(
            DefectDetectionResult(
                defect_type=defect_type.value,
                minimal_detected=minimal_detections == trials,
                minimal_time_ms=sum(minimal_times) / len(minimal_times),
                medium_detected=medium_detections == trials,
                medium_time_ms=sum(medium_times) / len(medium_times),
                expansive_detected=expansive_detections == trials,
                expansive_time_ms=sum(expansive_times) / len(expansive_times),
            )
        )

    # Display results
    _display_detection_results(results, trials, output)


@cli.command()
@click.option("--trials", default=1000, help="Trials per plan size")
@click.option("--output", default=None, help="Output JSON file")
def complexity(trials: int, output: str | None) -> None:
    """Run complexity scaling benchmark across plan sizes."""
    console.print("\n[bold]G_plan Complexity Scaling Benchmark[/bold]")
    console.print(f"Trials: {trials}")

    minimal_guard = MinimalPlanGuard()
    medium_guard = MediumPlanGuard()
    expansive_guard = ExpansivePlanGuard()

    timing_results: dict[str, dict[int, float]] = {
        "minimal": {},
        "medium": {},
        "expansive": {},
    }

    for num_steps in [5, 10, 20, 50, 100]:
        plan_dict = _generate_scaled_plan(num_steps)
        content = json.dumps(plan_dict, indent=2)
        artifact = _make_artifact(content)

        for label, guard in [
            ("minimal", minimal_guard),
            ("medium", medium_guard),
            ("expansive", expansive_guard),
        ]:
            times: list[float] = []
            for _ in range(trials):
                t0 = time.perf_counter()
                guard.validate(artifact)
                times.append((time.perf_counter() - t0) * 1000)
            timing_results[label][num_steps] = sum(times) / len(times)

    # Display
    table = Table(title="Validation Time by Plan Size (ms)")
    table.add_column("Steps", justify="right")
    table.add_column("Minimal", justify="right")
    table.add_column("Medium", justify="right")
    table.add_column("Expansive", justify="right")

    for num_steps in sorted(timing_results["minimal"].keys()):
        table.add_row(
            str(num_steps),
            f"{timing_results['minimal'][num_steps]:.3f}",
            f"{timing_results['medium'][num_steps]:.3f}",
            f"{timing_results['expansive'][num_steps]:.3f}",
        )

    console.print(table)

    if output:
        # Convert int keys to strings for JSON
        serializable = {
            level: {str(k): v for k, v in times.items()}
            for level, times in timing_results.items()
        }
        with open(output, "w") as f:
            json.dump(serializable, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


@cli.command()
@click.option("--trials", default=20, help="Number of LLM generation trials")
@click.option(
    "--backend",
    default="ollama",
    type=click.Choice(["ollama", "huggingface"]),
    help="LLM backend to use",
)
@click.option(
    "--host",
    default="http://localhost:11434",
    help="Ollama host URL (ollama backend only)",
)
@click.option("--model", default=None, help="Model to use (default depends on backend)")
@click.option(
    "--pipeline",
    default="single",
    type=click.Choice(["single", "classify-then-plan", "full"]),
    help="Pipeline mode: single, classify-then-plan (Option A), or full (4-step decomposition)",
)
@click.option("--output", default=None, help="Output JSON file")
@click.option("--verbose", "-v", is_flag=True, help="Show per-trial details")
def epsilon(
    trials: int,
    backend: str,
    host: str,
    model: str | None,
    pipeline: str,
    output: str | None,
    verbose: bool,
) -> None:
    """Estimate epsilon for LLM plan generation against G_plan guards.

    Generates N plans via LLM, validates each through Minimal/Medium/Expansive,
    and reports epsilon-hat (pass rate) with 95% Wilson confidence intervals.

    Pipeline modes:
      single              One-shot plan generation (g_plan_llm only)
      classify-then-plan  Two-step: g_analysis → g_plan_conditioned (Option A)
      full                Four-step: g_analysis → g_recon → g_strategy → g_plan_full

    Supports --backend ollama (default) or --backend huggingface.
    """
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    # Apply backend-specific defaults
    if model is None:
        model = (
            "Qwen/Qwen2.5-Coder-32B-Instruct"
            if backend == "huggingface"
            else "qwen2.5-coder:14b"
        )

    pipeline_label = {
        "single": "Single-shot",
        "classify-then-plan": "Classify-then-Plan (Option A)",
        "full": "Full Decomposition (analysis → recon → strategy → plan)",
    }[pipeline]

    console.print("\n[bold]G_plan Epsilon Estimation (LLM Plan Generation)[/bold]")
    console.print(f"Pipeline: {pipeline_label}")
    console.print(f"Backend: {backend}")
    console.print(f"Model: {model}")
    if backend == "ollama":
        console.print(f"Host: {host}")
    console.print(f"Trials: {trials}")

    has_presteps = pipeline in ("classify-then-plan", "full")
    has_recon = pipeline == "full"

    # Create LLM generators
    try:
        if backend == "huggingface":
            plan_generator = LLMPlanGenerator(model=model, backend="huggingface")
            if has_presteps:
                json_generator = LLMJsonGenerator(model=model, backend="huggingface")
        else:
            base_url = host.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            plan_generator = LLMPlanGenerator(model=model, base_url=base_url)
            if has_presteps:
                json_generator = LLMJsonGenerator(model=model, base_url=base_url)
    except (ImportError, ValueError) as err:
        console.print(f"[red]{err}[/red]")
        raise SystemExit(1) from err

    # Load specification and prompt templates
    specification = _load_specification()
    plan_template = _load_prompt_template("g_plan_llm")
    base_context = _make_context(specification)

    if has_presteps:
        analysis_template = _load_prompt_template("g_analysis")
        analysis_guard = AnalysisGuard()
    if has_recon:
        recon_template = _load_prompt_template("g_recon")
        strategy_template = _load_prompt_template("g_strategy")
        recon_guard = ReconGuard()
        strategy_guard = StrategyGuard()

    console.print(f"Specification: {len(specification)} chars")

    # Plan guards
    minimal_guard = MinimalPlanGuard()
    medium_guard = MediumPlanGuard()
    expansive_guard = ExpansivePlanGuard()

    # Run trials
    trial_results: list[EpsilonTrialResult] = []

    for i in range(trials):
        console.print(f"\n--- Trial {i + 1}/{trials} ---")

        trial = _run_epsilon_trial(
            trial_num=i + 1,
            pipeline=pipeline,
            base_context=base_context,
            plan_generator=plan_generator,
            json_generator=json_generator if has_presteps else None,
            plan_template=plan_template,
            analysis_template=analysis_template if has_presteps else None,
            recon_template=recon_template if has_recon else None,
            strategy_template=strategy_template if has_recon else None,
            analysis_guard=analysis_guard if has_presteps else None,
            recon_guard=recon_guard if has_recon else None,
            strategy_guard=strategy_guard if has_recon else None,
            minimal_guard=minimal_guard,
            medium_guard=medium_guard,
            expansive_guard=expansive_guard,
            verbose=verbose,
        )
        trial_results.append(trial)

    # Compute and display results
    _display_epsilon_results(trial_results, model, pipeline, output)


def _run_prestep(
    step_name: str,
    generator: LLMJsonGenerator,
    guard: AnalysisGuard | ReconGuard | StrategyGuard,
    context: Context,
    template: PromptTemplate,
    action_pair_id: str,
    verbose: bool,
) -> tuple[bool, str, str, float]:
    """Run a pre-planning step (analysis, recon, or strategy).

    Returns: (passed, content, feedback, time_ms)
    """
    t0 = time.perf_counter()
    try:
        artifact = generator.generate(
            context=context,
            template=template,
            action_pair_id=action_pair_id,
            workflow_id="epsilon_benchmark",
        )
        elapsed = (time.perf_counter() - t0) * 1000
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        console.print(f"  [red]{step_name} generation failed: {e}[/red]")
        return False, "", str(e), elapsed

    content = artifact.content
    result = guard.validate(artifact)
    feedback = result.feedback or ""

    status = "[green]P[/green]" if result.passed else "[red]F[/red]"
    console.print(f"  {step_name}: {status} ({elapsed:.0f}ms)")
    if verbose and not result.passed:
        console.print(f"    {feedback}")

    return result.passed, content, feedback, elapsed


def _run_epsilon_trial(
    trial_num: int,
    pipeline: str,
    base_context: Context,
    plan_generator: LLMPlanGenerator,
    json_generator: LLMJsonGenerator | None,
    plan_template: PromptTemplate,
    analysis_template: PromptTemplate | None,
    recon_template: PromptTemplate | None,
    strategy_template: PromptTemplate | None,
    analysis_guard: AnalysisGuard | None,
    recon_guard: ReconGuard | None,
    strategy_guard: StrategyGuard | None,
    minimal_guard: MinimalPlanGuard,
    medium_guard: MediumPlanGuard,
    expansive_guard: ExpansivePlanGuard,
    verbose: bool,
) -> EpsilonTrialResult:
    """Run a single epsilon trial for any pipeline mode."""
    # Track pre-step results
    a_passed: bool | None = None
    a_content = ""
    a_feedback = ""
    a_time = 0.0
    r_passed: bool | None = None
    r_content = ""
    r_feedback = ""
    r_time = 0.0
    s_passed: bool | None = None
    s_content = ""
    s_feedback = ""
    s_time = 0.0

    plan_context = base_context
    total_prestep_time = 0.0
    has_presteps = pipeline in ("classify-then-plan", "full")
    has_recon = pipeline == "full"

    def _fail_result(errors: list[str]) -> EpsilonTrialResult:
        """Build a failed trial result at the current point in the pipeline."""
        return EpsilonTrialResult(
            trial=trial_num,
            minimal_passed=False,
            medium_passed=False,
            expansive_passed=False,
            generation_time_ms=total_prestep_time,
            plan_steps=0,
            errors=errors,
            analysis_passed=a_passed,
            analysis_content=a_content,
            analysis_feedback=a_feedback,
            analysis_time_ms=a_time,
            recon_passed=r_passed,
            recon_content=r_content,
            recon_feedback=r_feedback,
            recon_time_ms=r_time,
            strategy_passed=s_passed,
            strategy_content=s_content,
            strategy_feedback=s_feedback,
            strategy_time_ms=s_time,
        )

    # --- Step 1: g_analysis (classify-then-plan and full) ---
    if has_presteps:
        assert json_generator is not None
        assert analysis_template is not None
        assert analysis_guard is not None

        a_passed, a_content, a_feedback, a_time = _run_prestep(
            "Analysis", json_generator, analysis_guard,
            base_context, analysis_template, "g_analysis", verbose,
        )
        total_prestep_time += a_time
        if not a_passed:
            return _fail_result([f"Analysis: {a_feedback}"])

        plan_context = base_context.amend(
            delta_constraints=f"## Problem Analysis (from g_analysis)\n{a_content}"
        )

    # --- Step 2: g_recon (full only) ---
    if has_recon:
        assert recon_template is not None
        assert recon_guard is not None

        r_passed, r_content, r_feedback, r_time = _run_prestep(
            "Recon", json_generator, recon_guard,
            plan_context, recon_template, "g_recon", verbose,
        )
        total_prestep_time += r_time
        if not r_passed:
            return _fail_result([f"Recon: {r_feedback}"])

        plan_context = plan_context.amend(
            delta_constraints=f"## Codebase Reconnaissance (from g_recon)\n{r_content}"
        )

    # --- Step 3: g_strategy (full only) ---
    if has_recon:
        assert strategy_template is not None
        assert strategy_guard is not None

        s_passed, s_content, s_feedback, s_time = _run_prestep(
            "Strategy", json_generator, strategy_guard,
            plan_context, strategy_template, "g_strategy", verbose,
        )
        total_prestep_time += s_time
        if not s_passed:
            return _fail_result([f"Strategy: {s_feedback}"])

        plan_context = plan_context.amend(
            delta_constraints=f"## Selected Strategy (from g_strategy)\n{s_content}"
        )

    # --- Step 4: Generate plan ---
    plan_action_pair = {
        "single": "g_plan_llm",
        "classify-then-plan": "g_plan_conditioned",
        "full": "g_plan_full",
    }[pipeline]

    t0 = time.perf_counter()
    try:
        plan_artifact = plan_generator.generate(
            context=plan_context,
            template=plan_template,
            action_pair_id=plan_action_pair,
            workflow_id="epsilon_benchmark",
        )
        plan_gen_time = (time.perf_counter() - t0) * 1000
    except Exception as e:
        plan_gen_time = (time.perf_counter() - t0) * 1000
        console.print(f"  [red]Plan generation failed: {e}[/red]")
        total_prestep_time += plan_gen_time
        return _fail_result([f"Plan generation failed: {e}"])

    total_gen_time = total_prestep_time + plan_gen_time

    # Count steps in generated plan
    plan_content = plan_artifact.content
    plan_steps = 0
    try:
        plan_data = json.loads(plan_content)
        plan_steps = len(plan_data.get("steps", []))
    except (json.JSONDecodeError, TypeError):
        pass

    # Validate plan at all three levels
    min_r = minimal_guard.validate(plan_artifact)
    med_r = medium_guard.validate(plan_artifact)
    exp_r = expansive_guard.validate(plan_artifact)

    errors: list[str] = []
    if not min_r.passed and min_r.feedback:
        errors.append(f"Minimal: {min_r.feedback}")
    if not med_r.passed and med_r.feedback and min_r.passed:
        errors.append(f"Medium: {med_r.feedback}")
    if not exp_r.passed and exp_r.feedback and med_r.passed:
        errors.append(f"Expansive: {exp_r.feedback}")

    # Per-trial summary
    min_s = "[green]P[/green]" if min_r.passed else "[red]F[/red]"
    med_s = "[green]P[/green]" if med_r.passed else "[red]F[/red]"
    exp_s = "[green]P[/green]" if exp_r.passed else "[red]F[/red]"
    console.print(
        f"  Min:{min_s} Med:{med_s} Exp:{exp_s} "
        f"| {plan_steps} steps | {total_gen_time:.0f}ms"
    )
    if verbose and errors:
        for err in errors:
            console.print(f"    {err}")

    return EpsilonTrialResult(
        trial=trial_num,
        minimal_passed=min_r.passed,
        medium_passed=med_r.passed,
        expansive_passed=exp_r.passed,
        generation_time_ms=total_gen_time,
        plan_steps=plan_steps,
        errors=errors,
        plan_content=plan_content,
        minimal_feedback=min_r.feedback or "",
        medium_feedback=med_r.feedback or "",
        expansive_feedback=exp_r.feedback or "",
        analysis_passed=a_passed,
        analysis_content=a_content,
        analysis_feedback=a_feedback,
        analysis_time_ms=a_time,
        recon_passed=r_passed,
        recon_content=r_content,
        recon_feedback=r_feedback,
        recon_time_ms=r_time,
        strategy_passed=s_passed,
        strategy_content=s_content,
        strategy_feedback=s_feedback,
        strategy_time_ms=s_time,
    )


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def _display_detection_results(
    results: list[DefectDetectionResult],
    trials: int,
    output: str | None,
) -> None:
    """Display defect detection benchmark results."""
    table = Table(title="Defect Detection by Rigor Level")
    table.add_column("Defect Type")
    table.add_column("Minimal", justify="center")
    table.add_column("Medium", justify="center")
    table.add_column("Expansive", justify="center")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Med (ms)", justify="right")
    table.add_column("Exp (ms)", justify="right")

    for r in results:
        min_det = "[green]Y[/green]" if r.minimal_detected else "[red]N[/red]"
        med_det = "[green]Y[/green]" if r.medium_detected else "[red]N[/red]"
        exp_det = "[green]Y[/green]" if r.expansive_detected else "[red]N[/red]"

        table.add_row(
            r.defect_type,
            min_det,
            med_det,
            exp_det,
            f"{r.minimal_time_ms:.3f}",
            f"{r.medium_time_ms:.3f}",
            f"{r.expansive_time_ms:.3f}",
        )

    console.print(table)

    # Summary
    total = len(results)
    min_count = sum(1 for r in results if r.minimal_detected)
    med_count = sum(1 for r in results if r.medium_detected)
    exp_count = sum(1 for r in results if r.expansive_detected)

    console.print(f"\n[bold]Detection Summary ({trials} trials per defect):[/bold]")
    console.print(f"  Minimal:   {min_count}/{total} ({min_count / total:.0%})")
    console.print(f"  Medium:    {med_count}/{total} ({med_count / total:.0%})")
    console.print(f"  Expansive: {exp_count}/{total} ({exp_count / total:.0%})")

    avg_min = sum(r.minimal_time_ms for r in results) / total
    avg_med = sum(r.medium_time_ms for r in results) / total
    avg_exp = sum(r.expansive_time_ms for r in results) / total

    console.print("\n[bold]Average Validation Time:[/bold]")
    console.print(f"  Minimal:   {avg_min:.3f}ms")
    console.print(f"  Medium:    {avg_med:.3f}ms")
    console.print(f"  Expansive: {avg_exp:.3f}ms")

    if output:
        output_data = {
            "trials": trials,
            "results": [
                {
                    "defect_type": r.defect_type,
                    "minimal": {
                        "detected": r.minimal_detected,
                        "time_ms": r.minimal_time_ms,
                    },
                    "medium": {
                        "detected": r.medium_detected,
                        "time_ms": r.medium_time_ms,
                    },
                    "expansive": {
                        "detected": r.expansive_detected,
                        "time_ms": r.expansive_time_ms,
                    },
                }
                for r in results
            ],
            "summary": {
                "detection_rates": {
                    "minimal": min_count / total,
                    "medium": med_count / total,
                    "expansive": exp_count / total,
                },
                "avg_times_ms": {
                    "minimal": avg_min,
                    "medium": avg_med,
                    "expansive": avg_exp,
                },
            },
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


def _display_epsilon_results(
    results: list[EpsilonTrialResult],
    model: str,
    pipeline: str,
    output: str | None,
) -> None:
    """Display epsilon estimation results."""
    k = len(results)

    min_pass = sum(1 for r in results if r.minimal_passed)
    med_pass = sum(1 for r in results if r.medium_passed)
    exp_pass = sum(1 for r in results if r.expansive_passed)

    min_eps = min_pass / k if k > 0 else 0.0
    med_eps = med_pass / k if k > 0 else 0.0
    exp_eps = exp_pass / k if k > 0 else 0.0

    min_ci = _wilson_ci(min_pass, k)
    med_ci = _wilson_ci(med_pass, k)
    exp_ci = _wilson_ci(exp_pass, k)

    avg_gen_time = sum(r.generation_time_ms for r in results) / k if k > 0 else 0.0
    avg_steps = sum(r.plan_steps for r in results) / k if k > 0 else 0.0

    # Summary table
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    pipeline_label = {
        "single": "Single-shot",
        "classify-then-plan": "Classify-then-Plan (Option A)",
        "full": "Full Decomposition (analysis → recon → strategy → plan)",
    }.get(pipeline, pipeline)
    console.print(f"[bold]Epsilon Estimation Results — {pipeline_label}[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]")

    # Pre-step epsilon table (for decomposed pipelines)
    prestep_rows: list[tuple[str, str, int, float, tuple[float, float], float]] = []

    is_decomposed = any(r.analysis_passed is not None for r in results)
    if is_decomposed:
        analysis_pass = sum(1 for r in results if r.analysis_passed is True)
        analysis_eps = analysis_pass / k if k > 0 else 0.0
        analysis_ci = _wilson_ci(analysis_pass, k)
        avg_analysis_time = (
            sum(r.analysis_time_ms for r in results) / k if k > 0 else 0.0
        )
        prestep_rows.append((
            "g_analysis", "analysis_valid",
            analysis_pass, analysis_eps, analysis_ci, avg_analysis_time,
        ))

    has_recon = any(r.recon_passed is not None for r in results)
    if has_recon:
        recon_pass = sum(1 for r in results if r.recon_passed is True)
        recon_eps = recon_pass / k if k > 0 else 0.0
        recon_ci = _wilson_ci(recon_pass, k)
        avg_recon_time = (
            sum(r.recon_time_ms for r in results) / k if k > 0 else 0.0
        )
        prestep_rows.append((
            "g_recon", "recon_valid",
            recon_pass, recon_eps, recon_ci, avg_recon_time,
        ))

    has_strategy = any(r.strategy_passed is not None for r in results)
    if has_strategy:
        strategy_pass = sum(1 for r in results if r.strategy_passed is True)
        strategy_eps = strategy_pass / k if k > 0 else 0.0
        strategy_ci = _wilson_ci(strategy_pass, k)
        avg_strategy_time = (
            sum(r.strategy_time_ms for r in results) / k if k > 0 else 0.0
        )
        prestep_rows.append((
            "g_strategy", "strategy_valid",
            strategy_pass, strategy_eps, strategy_ci, avg_strategy_time,
        ))

    if prestep_rows:
        ps_table = Table(title=f"Pre-step Epsilon (k={k})")
        ps_table.add_column("Step")
        ps_table.add_column("Guard")
        ps_table.add_column("Pass", justify="right")
        ps_table.add_column("epsilon-hat", justify="right")
        ps_table.add_column("95% CI", justify="center")
        ps_table.add_column("Avg (ms)", justify="right")

        for step, guard, passed, eps, ci, avg_t in prestep_rows:
            ps_table.add_row(
                step, guard, str(passed),
                f"{eps:.2f}",
                f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                f"{avg_t:.0f}",
            )

        console.print(ps_table)
        console.print()

    # Plan epsilon table
    table = Table(title=f"G_plan Epsilon (k={k}, model={model})")
    table.add_column("Rigor Level")
    table.add_column("Pass", justify="right")
    table.add_column("Fail", justify="right")
    table.add_column("epsilon-hat", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("E[attempts]", justify="right")

    for label, passed, eps, ci in [
        ("Minimal", min_pass, min_eps, min_ci),
        ("Medium", med_pass, med_eps, med_ci),
        ("Expansive", exp_pass, exp_eps, exp_ci),
    ]:
        e_attempts = f"{1 / eps:.1f}" if eps > 0 else "inf"
        table.add_row(
            label,
            str(passed),
            str(k - passed),
            f"{eps:.2f}",
            f"[{ci[0]:.2f}, {ci[1]:.2f}]",
            e_attempts,
        )

    console.print(table)

    console.print(f"\nAvg total generation time: {avg_gen_time:.0f}ms")
    console.print(f"Avg plan steps: {avg_steps:.1f}")

    # Error frequency analysis
    error_counts: dict[str, int] = {}
    for r in results:
        for err in r.errors:
            key = err.split("\n")[0][:80]
            error_counts[key] = error_counts.get(key, 0) + 1

    if error_counts:
        console.print("\n[bold]Common Failure Modes:[/bold]")
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
            console.print(f"  {count}x  {err}")

    if output:
        output_data: dict[str, Any] = {
            "model": model,
            "pipeline": pipeline,
            "trials": k,
            "epsilon": {
                "minimal": {
                    "pass": min_pass,
                    "epsilon_hat": min_eps,
                    "ci_95": list(min_ci),
                },
                "medium": {
                    "pass": med_pass,
                    "epsilon_hat": med_eps,
                    "ci_95": list(med_ci),
                },
                "expansive": {
                    "pass": exp_pass,
                    "epsilon_hat": exp_eps,
                    "ci_95": list(exp_ci),
                },
            },
            "avg_generation_time_ms": avg_gen_time,
            "avg_plan_steps": avg_steps,
        }

        # Include pre-step epsilon for decomposed pipelines
        if prestep_rows:
            output_data["prestep_epsilon"] = {
                step: {
                    "pass": passed,
                    "epsilon_hat": eps,
                    "ci_95": list(ci),
                    "avg_time_ms": avg_t,
                }
                for step, _guard, passed, eps, ci, avg_t in prestep_rows
            }

        output_data["trials_detail"] = [
            {
                "trial": r.trial,
                "minimal_passed": r.minimal_passed,
                "medium_passed": r.medium_passed,
                "expansive_passed": r.expansive_passed,
                "generation_time_ms": r.generation_time_ms,
                "plan_steps": r.plan_steps,
                "errors": r.errors,
                "plan_content": r.plan_content,
                "guard_feedback": {
                    "minimal": r.minimal_feedback,
                    "medium": r.medium_feedback,
                    "expansive": r.expansive_feedback,
                },
                **(
                    {
                        "analysis_passed": r.analysis_passed,
                        "analysis_content": r.analysis_content,
                        "analysis_feedback": r.analysis_feedback,
                        "analysis_time_ms": r.analysis_time_ms,
                    }
                    if r.analysis_passed is not None
                    else {}
                ),
                **(
                    {
                        "recon_passed": r.recon_passed,
                        "recon_content": r.recon_content,
                        "recon_feedback": r.recon_feedback,
                        "recon_time_ms": r.recon_time_ms,
                    }
                    if r.recon_passed is not None
                    else {}
                ),
                **(
                    {
                        "strategy_passed": r.strategy_passed,
                        "strategy_content": r.strategy_content,
                        "strategy_feedback": r.strategy_feedback,
                        "strategy_time_ms": r.strategy_time_ms,
                    }
                    if r.strategy_passed is not None
                    else {}
                ),
            }
            for r in results
        ]
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


@cli.command()
@click.option(
    "--problems",
    required=True,
    type=click.Path(exists=True),
    help="Path to problems JSON file or directory of JSON files",
)
@click.option(
    "--pipeline",
    "pipelines",
    multiple=True,
    default=["single", "full"],
    type=click.Choice(["single", "classify-then-plan", "full"]),
    help="Pipeline modes to evaluate (repeat for multiple)",
)
@click.option("--trials", default=1, help="Trials per problem per pipeline")
@click.option(
    "--backend",
    default="ollama",
    type=click.Choice(["ollama", "huggingface"]),
    help="LLM backend to use",
)
@click.option(
    "--host",
    default="http://localhost:11434",
    help="Ollama host URL (ollama backend only)",
)
@click.option("--model", default=None, help="Model to use (default depends on backend)")
@click.option("--output", default=None, help="Output JSON file for full results")
@click.option("--verbose", "-v", is_flag=True, help="Show per-trial details")
def evaluate(
    problems: str,
    pipelines: tuple[str, ...],
    trials: int,
    backend: str,
    host: str,
    model: str | None,
    output: str | None,
    verbose: bool,
) -> None:
    """Run evaluation harness across problems, pipelines, and trials.

    Loads a problem set from a JSON file or directory, runs each problem
    through each pipeline mode, and produces a scorecard with epsilon,
    strategy alignment, and per-category breakdowns.

    Example:
        uv run python -m examples.advanced.g_plan_benchmark.demo evaluate \\
            --problems problems/catalog.json \\
            --pipeline single --pipeline full \\
            --trials 3 --output eval_results.json
    """
    from .evaluation import ExperimentConfig, ExperimentRunner, ProblemSet, score_experiment

    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    if model is None:
        model = (
            "Qwen/Qwen2.5-Coder-32B-Instruct"
            if backend == "huggingface"
            else "qwen2.5-coder:14b"
        )

    # Load problems
    problem_set = ProblemSet.load(problems)
    console.print(f"\n[bold]Evaluation Harness[/bold]")
    console.print(f"Problems: {len(problem_set)} from {problems}")
    console.print(f"Pipelines: {', '.join(pipelines)}")
    console.print(f"Trials per problem: {trials}")
    console.print(f"Model: {model} ({backend})")
    total = len(problem_set) * len(pipelines) * trials
    console.print(f"Total trials: {total}")

    # Configure and run
    config = ExperimentConfig(
        pipelines=list(pipelines),
        trials_per_problem=trials,
        model=model,
        backend=backend,
        host=host,
    )

    runner = ExperimentRunner(config)

    def _on_trial(problem_id: str, pipeline: str, trial_num: int, result) -> None:
        status = "[green]OK[/green]" if result.pipeline_succeeded else "[red]FAIL[/red]"
        console.print(
            f"  {problem_id} | {pipeline} | trial {trial_num}: {status} "
            f"({result.total_time_ms:.0f}ms)"
        )
        if verbose and result.errors:
            for err in result.errors:
                console.print(f"    {err}")

    result = runner.run(problem_set, on_trial=_on_trial)

    # Score
    scorecard = score_experiment(result, problem_set)

    # Display scorecard
    _display_scorecard(scorecard)

    # Save results
    if output:
        output_data = {
            "scorecard": scorecard.to_dict(),
            "raw_results": result.to_dict(),
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


def _display_scorecard(scorecard) -> None:
    """Display the experiment scorecard."""
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print("[bold]Evaluation Scorecard[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]")

    for name, card in scorecard.pipelines.items():
        console.print(f"\n[bold]{name}[/bold]")

        # Pipeline epsilon table
        table = Table(title=f"Epsilon — {name}")
        table.add_column("Metric")
        table.add_column("Pass", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("epsilon-hat", justify="right")
        table.add_column("95% CI", justify="center")
        table.add_column("E[attempts]", justify="right")

        for label, score in [
            ("Pipeline (end-to-end)", card.pipeline_epsilon),
            ("Minimal", card.minimal_epsilon),
            ("Medium", card.medium_epsilon),
            ("Expansive", card.expansive_epsilon),
        ]:
            ci = score.ci_95
            e_att = f"{score.e_attempts:.1f}" if score.e_attempts != float("inf") else "inf"
            table.add_row(
                label,
                str(score.passed),
                str(score.total),
                f"{score.epsilon_hat:.2f}",
                f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                e_att,
            )

        # Pre-step rows
        for label, score in [
            ("g_analysis", card.analysis_epsilon),
            ("g_recon", card.recon_epsilon),
            ("g_strategy", card.strategy_epsilon),
        ]:
            if score is not None:
                ci = score.ci_95
                e_att = f"{score.e_attempts:.1f}" if score.e_attempts != float("inf") else "inf"
                table.add_row(
                    label,
                    str(score.passed),
                    str(score.total),
                    f"{score.epsilon_hat:.2f}",
                    f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                    e_att,
                )

        console.print(table)

        # Strategy alignment
        if card.strategy_alignment.total > 0:
            sa = card.strategy_alignment
            ci = sa.ci_95
            console.print(
                f"  Strategy alignment: {sa.correct}/{sa.total} "
                f"({sa.accuracy:.0%}) CI [{ci[0]:.2f}, {ci[1]:.2f}]"
            )

        # Per-category breakdown
        if card.categories:
            cat_table = Table(title=f"Per-Category — {name}")
            cat_table.add_column("Category")
            cat_table.add_column("Pipeline eps", justify="right")
            cat_table.add_column("Medium eps", justify="right")
            cat_table.add_column("Strategy align", justify="right")
            cat_table.add_column("n", justify="right")

            for cat in card.categories:
                align_str = (
                    f"{cat.strategy_alignment.accuracy:.0%}"
                    if cat.strategy_alignment.total > 0
                    else "—"
                )
                cat_table.add_row(
                    cat.category,
                    f"{cat.pipeline_epsilon.epsilon_hat:.2f}",
                    f"{cat.medium_epsilon.epsilon_hat:.2f}",
                    align_str,
                    str(cat.pipeline_epsilon.total),
                )

            console.print(cat_table)

        console.print(f"  Avg time: {card.avg_time_ms:.0f}ms")

    # Cross-pipeline comparison
    if scorecard.comparison:
        console.print(f"\n[bold]Cross-Pipeline Comparison[/bold]")
        for key, delta in scorecard.comparison.get("delta_vs_baseline", {}).items():
            sign = "+" if delta["delta_pp"] >= 0 else ""
            console.print(
                f"  {delta['other']} vs {delta['baseline']}: "
                f"{sign}{delta['delta_pp']}pp "
                f"({delta['baseline_epsilon']:.2f} → {delta['other_epsilon']:.2f})"
            )


if __name__ == "__main__":
    cli()
