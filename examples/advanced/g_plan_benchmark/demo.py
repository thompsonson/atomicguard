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
from .guards import AnalysisGuard, ExpansivePlanGuard, MediumPlanGuard, MinimalPlanGuard
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
    # Option A: Classify-then-Plan pipeline fields
    analysis_passed: bool | None = None  # None = single pipeline (no analysis)
    analysis_content: str = ""
    analysis_feedback: str = ""
    analysis_time_ms: float = 0.0


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
    type=click.Choice(["single", "classify-then-plan"]),
    help="Pipeline mode: single (one-shot) or classify-then-plan (Option A decomposition)",
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
    }[pipeline]

    console.print("\n[bold]G_plan Epsilon Estimation (LLM Plan Generation)[/bold]")
    console.print(f"Pipeline: {pipeline_label}")
    console.print(f"Backend: {backend}")
    console.print(f"Model: {model}")
    if backend == "ollama":
        console.print(f"Host: {host}")
    console.print(f"Trials: {trials}")

    # Create LLM generators
    try:
        if backend == "huggingface":
            plan_generator = LLMPlanGenerator(model=model, backend="huggingface")
            if pipeline == "classify-then-plan":
                analysis_generator = LLMJsonGenerator(model=model, backend="huggingface")
        else:
            base_url = host.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            plan_generator = LLMPlanGenerator(model=model, base_url=base_url)
            if pipeline == "classify-then-plan":
                analysis_generator = LLMJsonGenerator(model=model, base_url=base_url)
    except (ImportError, ValueError) as err:
        console.print(f"[red]{err}[/red]")
        raise SystemExit(1) from err

    # Load specification and prompt templates
    specification = _load_specification()
    plan_template = _load_prompt_template("g_plan_llm")
    base_context = _make_context(specification)

    if pipeline == "classify-then-plan":
        analysis_template = _load_prompt_template("g_analysis")
        analysis_guard = AnalysisGuard()

    console.print(f"Specification: {len(specification)} chars")

    # Plan guards
    minimal_guard = MinimalPlanGuard()
    medium_guard = MediumPlanGuard()
    expansive_guard = ExpansivePlanGuard()

    # Run trials
    trial_results: list[EpsilonTrialResult] = []

    for i in range(trials):
        console.print(f"\n--- Trial {i + 1}/{trials} ---")

        analysis_passed: bool | None = None
        analysis_content = ""
        analysis_feedback_str = ""
        analysis_time = 0.0
        plan_context = base_context

        # Step 1 (classify-then-plan only): Run g_analysis
        if pipeline == "classify-then-plan":
            t0 = time.perf_counter()
            try:
                analysis_artifact = analysis_generator.generate(
                    context=base_context,
                    template=analysis_template,
                    action_pair_id="g_analysis",
                    workflow_id="epsilon_benchmark",
                )
                analysis_time = (time.perf_counter() - t0) * 1000
            except Exception as e:
                analysis_time = (time.perf_counter() - t0) * 1000
                console.print(f"  [red]Analysis generation failed: {e}[/red]")
                trial_results.append(
                    EpsilonTrialResult(
                        trial=i + 1,
                        minimal_passed=False,
                        medium_passed=False,
                        expansive_passed=False,
                        generation_time_ms=analysis_time,
                        plan_steps=0,
                        errors=[f"Analysis generation failed: {e}"],
                        analysis_passed=False,
                        analysis_content="",
                        analysis_feedback=str(e),
                        analysis_time_ms=analysis_time,
                    )
                )
                continue

            analysis_content = analysis_artifact.content
            analysis_r = analysis_guard.validate(analysis_artifact)
            analysis_passed = analysis_r.passed
            analysis_feedback_str = analysis_r.feedback or ""

            a_status = "[green]P[/green]" if analysis_r.passed else "[red]F[/red]"
            console.print(f"  Analysis: {a_status} ({analysis_time:.0f}ms)")

            if not analysis_r.passed:
                # Analysis failed — plan generation skipped
                if verbose:
                    console.print(f"    {analysis_feedback_str}")
                trial_results.append(
                    EpsilonTrialResult(
                        trial=i + 1,
                        minimal_passed=False,
                        medium_passed=False,
                        expansive_passed=False,
                        generation_time_ms=analysis_time,
                        plan_steps=0,
                        errors=[f"Analysis: {analysis_feedback_str}"],
                        analysis_passed=False,
                        analysis_content=analysis_content,
                        analysis_feedback=analysis_feedback_str,
                        analysis_time_ms=analysis_time,
                    )
                )
                continue

            # Enrich context with analysis results for plan generation
            plan_context = base_context.amend(
                delta_constraints=(
                    f"## Problem Analysis (from g_analysis)\n{analysis_content}"
                )
            )

        # Step 2: Generate plan (single-shot or conditioned on analysis)
        t0 = time.perf_counter()
        try:
            plan_artifact = plan_generator.generate(
                context=plan_context,
                template=plan_template,
                action_pair_id=(
                    "g_plan_conditioned"
                    if pipeline == "classify-then-plan"
                    else "g_plan_llm"
                ),
                workflow_id="epsilon_benchmark",
            )
            plan_gen_time = (time.perf_counter() - t0) * 1000
        except Exception as e:
            plan_gen_time = (time.perf_counter() - t0) * 1000
            console.print(f"  [red]Plan generation failed: {e}[/red]")
            trial_results.append(
                EpsilonTrialResult(
                    trial=i + 1,
                    minimal_passed=False,
                    medium_passed=False,
                    expansive_passed=False,
                    generation_time_ms=analysis_time + plan_gen_time,
                    plan_steps=0,
                    errors=[f"Plan generation failed: {e}"],
                    analysis_passed=analysis_passed,
                    analysis_content=analysis_content,
                    analysis_feedback=analysis_feedback_str,
                    analysis_time_ms=analysis_time,
                )
            )
            continue

        total_gen_time = analysis_time + plan_gen_time

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

        trial_results.append(
            EpsilonTrialResult(
                trial=i + 1,
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
                analysis_passed=analysis_passed,
                analysis_content=analysis_content,
                analysis_feedback=analysis_feedback_str,
                analysis_time_ms=analysis_time,
            )
        )

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

    # Compute and display results
    _display_epsilon_results(trial_results, model, pipeline, output)


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
    }.get(pipeline, pipeline)
    console.print(f"[bold]Epsilon Estimation Results — {pipeline_label}[/bold]")
    console.print(f"[bold]{'=' * 60}[/bold]")

    # Analysis epsilon (classify-then-plan only)
    is_decomposed = any(r.analysis_passed is not None for r in results)
    if is_decomposed:
        analysis_pass = sum(
            1 for r in results if r.analysis_passed is True
        )
        analysis_eps = analysis_pass / k if k > 0 else 0.0
        analysis_ci = _wilson_ci(analysis_pass, k)
        avg_analysis_time = (
            sum(r.analysis_time_ms for r in results) / k if k > 0 else 0.0
        )

        a_table = Table(title=f"g_analysis Epsilon (k={k})")
        a_table.add_column("Guard")
        a_table.add_column("Pass", justify="right")
        a_table.add_column("Fail", justify="right")
        a_table.add_column("epsilon-hat", justify="right")
        a_table.add_column("95% CI", justify="center")

        e_a = f"{1 / analysis_eps:.1f}" if analysis_eps > 0 else "inf"
        a_table.add_row(
            "analysis_valid",
            str(analysis_pass),
            str(k - analysis_pass),
            f"{analysis_eps:.2f}",
            f"[{analysis_ci[0]:.2f}, {analysis_ci[1]:.2f}]",
        )
        console.print(a_table)
        console.print(f"Avg analysis time: {avg_analysis_time:.0f}ms")
        console.print(f"E[analysis attempts]: {e_a}")
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

        # Include analysis epsilon for decomposed pipeline
        if is_decomposed:
            output_data["analysis_epsilon"] = {
                "pass": analysis_pass,
                "epsilon_hat": analysis_eps,
                "ci_95": list(analysis_ci),
                "avg_time_ms": avg_analysis_time,
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
            }
            for r in results
        ]
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


if __name__ == "__main__":
    cli()
