"""Experiment runner: executes problems × pipelines × trials."""

from __future__ import annotations

import json
import logging
import time

from atomicguard.domain.models import (
    AmbientEnvironment,
    Context,
)
from atomicguard.domain.prompts import PromptTemplate

from ..generators import LLMJsonGenerator, LLMPlanGenerator
from ..guards import (
    AnalysisGuard,
    ExpansivePlanGuard,
    MediumPlanGuard,
    MinimalPlanGuard,
    ReconGuard,
    StrategyGuard,
)
from .problem import Problem, ProblemSet
from .results import (
    ExperimentConfig,
    ExperimentResult,
    PipelineResult,
    PlanValidation,
    ProblemTrialResult,
    StepResult,
)

logger = logging.getLogger(__name__)

_PROMPTS_PATH = None  # set at import from parent package


def _get_prompts_path():
    """Resolve the prompts.json path relative to the benchmark package."""
    from pathlib import Path

    return Path(__file__).parent.parent / "prompts.json"


def _load_prompt_template(step_id: str) -> PromptTemplate:
    """Load a PromptTemplate from prompts.json."""
    path = _get_prompts_path()
    with open(path) as f:
        prompts = json.load(f)
    entry = prompts[step_id]
    return PromptTemplate(
        role=entry["role"],
        constraints=entry["constraints"],
        task=entry["task"],
        feedback_wrapper=entry["feedback_wrapper"],
    )


def _make_context(specification: str) -> Context:
    """Build a Context from a problem specification."""
    return Context(
        ambient=AmbientEnvironment(repository=None, constraints=""),
        specification=specification,
        current_artifact=None,
        feedback_history=(),
        dependency_artifacts=(),
    )


def _extract_json_field(content: str, field: str) -> str:
    """Extract a field value from a JSON string, returning '' on failure."""
    try:
        data = json.loads(content)
        return str(data.get(field, ""))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""


class ExperimentRunner:
    """Runs evaluation experiments across problems, pipelines, and trials.

    Usage::

        config = ExperimentConfig(
            pipelines=["single", "full"],
            trials_per_problem=3,
            model="qwen2.5-coder:14b",
        )
        runner = ExperimentRunner(config)
        result = runner.run(problem_set)
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._plan_generator: LLMPlanGenerator | None = None
        self._json_generator: LLMJsonGenerator | None = None

        # Guards (instantiated once, reused across all trials)
        self._minimal_guard = MinimalPlanGuard()
        self._medium_guard = MediumPlanGuard()
        self._expansive_guard = ExpansivePlanGuard()
        self._analysis_guard = AnalysisGuard()
        self._recon_guard = ReconGuard()
        self._strategy_guard = StrategyGuard()

        # Prompt templates (loaded once)
        self._plan_template = _load_prompt_template("g_plan_llm")
        self._analysis_template = _load_prompt_template("g_analysis")
        self._recon_template = _load_prompt_template("g_recon")
        self._strategy_template = _load_prompt_template("g_strategy")

    def _ensure_generators(self) -> None:
        """Lazily initialise LLM generators."""
        if self._plan_generator is not None:
            return
        cfg = self.config
        if cfg.backend == "huggingface":
            self._plan_generator = LLMPlanGenerator(
                model=cfg.model, backend="huggingface"
            )
            self._json_generator = LLMJsonGenerator(
                model=cfg.model, backend="huggingface"
            )
        else:
            base_url = cfg.host.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            self._plan_generator = LLMPlanGenerator(
                model=cfg.model, base_url=base_url
            )
            self._json_generator = LLMJsonGenerator(
                model=cfg.model, base_url=base_url
            )

    def run(
        self,
        problem_set: ProblemSet,
        *,
        on_trial: callable | None = None,
    ) -> ExperimentResult:
        """Run the full experiment.

        Args:
            problem_set: Problems to evaluate.
            on_trial: Optional callback(problem_id, pipeline, trial_num, result)
                called after each trial completes. Useful for progress reporting.

        Returns:
            ExperimentResult with per-pipeline aggregated results.
        """
        self._ensure_generators()

        result = ExperimentResult(config=self.config)
        for pipeline in self.config.pipelines:
            result.pipeline_results[pipeline] = PipelineResult(pipeline=pipeline)

        total = len(problem_set) * len(self.config.pipelines) * self.config.trials_per_problem
        completed = 0

        for problem in problem_set:
            for pipeline in self.config.pipelines:
                for trial_num in range(1, self.config.trials_per_problem + 1):
                    completed += 1
                    logger.info(
                        "[%d/%d] %s | %s | trial %d",
                        completed, total,
                        problem.problem_id, pipeline, trial_num,
                    )

                    trial_result = self._run_trial(problem, pipeline, trial_num)
                    result.pipeline_results[pipeline].trials.append(trial_result)

                    if on_trial is not None:
                        on_trial(
                            problem.problem_id, pipeline, trial_num, trial_result,
                        )

        return result

    def _run_trial(
        self,
        problem: Problem,
        pipeline: str,
        trial_num: int,
    ) -> ProblemTrialResult:
        """Run a single trial: one problem through one pipeline."""
        assert self._plan_generator is not None
        assert self._json_generator is not None

        base_context = _make_context(problem.description)
        plan_context = base_context
        total_start = time.perf_counter()

        trial = ProblemTrialResult(
            problem_id=problem.problem_id,
            pipeline=pipeline,
            trial_num=trial_num,
        )

        has_presteps = pipeline in ("classify-then-plan", "full")
        has_recon = pipeline == "full"

        # --- Step 1: g_analysis ---
        if has_presteps:
            analysis = self._run_step(
                "g_analysis",
                self._json_generator,
                self._analysis_guard,
                plan_context,
                self._analysis_template,
            )
            trial.analysis = analysis
            if not analysis.passed:
                trial.errors.append(f"Analysis: {analysis.feedback}")
                trial.total_time_ms = (time.perf_counter() - total_start) * 1000
                return trial

            trial.classified_type = _extract_json_field(analysis.content, "problem_type")
            plan_context = plan_context.amend(
                delta_constraints=f"## Problem Analysis (from g_analysis)\n{analysis.content}"
            )

        # --- Step 2: g_recon ---
        if has_recon:
            recon = self._run_step(
                "g_recon",
                self._json_generator,
                self._recon_guard,
                plan_context,
                self._recon_template,
            )
            trial.recon = recon
            if not recon.passed:
                trial.errors.append(f"Recon: {recon.feedback}")
                trial.total_time_ms = (time.perf_counter() - total_start) * 1000
                return trial

            plan_context = plan_context.amend(
                delta_constraints=f"## Codebase Reconnaissance (from g_recon)\n{recon.content}"
            )

        # --- Step 3: g_strategy ---
        if has_recon:
            strategy = self._run_step(
                "g_strategy",
                self._json_generator,
                self._strategy_guard,
                plan_context,
                self._strategy_template,
            )
            trial.strategy = strategy
            if not strategy.passed:
                trial.errors.append(f"Strategy: {strategy.feedback}")
                trial.total_time_ms = (time.perf_counter() - total_start) * 1000
                return trial

            trial.selected_strategy = _extract_json_field(strategy.content, "strategy_id")
            plan_context = plan_context.amend(
                delta_constraints=f"## Selected Strategy (from g_strategy)\n{strategy.content}"
            )

        # --- Step 4: Plan generation ---
        plan_action_pair = {
            "single": "g_plan_llm",
            "classify-then-plan": "g_plan_conditioned",
            "full": "g_plan_full",
        }[pipeline]

        plan_step = self._run_step(
            plan_action_pair,
            self._plan_generator,
            None,  # we validate with all three guards below
            plan_context,
            self._plan_template,
        )

        if not plan_step.passed:
            # Generation itself failed (exception)
            trial.errors.append(f"Plan generation: {plan_step.feedback}")
            trial.total_time_ms = (time.perf_counter() - total_start) * 1000
            return trial

        # Validate plan at all three rigor levels
        from atomicguard.domain.models import Artifact, ArtifactStatus, ContextSnapshot
        from uuid import uuid4

        plan_artifact = Artifact(
            artifact_id=str(uuid4()),
            workflow_id="evaluation",
            content=plan_step.content,
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=plan_action_pair,
            created_at="",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=ContextSnapshot(
                workflow_id="evaluation",
                specification=problem.description[:200],
                constraints="",
                feedback_history=(),
            ),
        )

        min_r = self._minimal_guard.validate(plan_artifact)
        med_r = self._medium_guard.validate(plan_artifact)
        exp_r = self._expansive_guard.validate(plan_artifact)

        plan_steps = 0
        try:
            plan_data = json.loads(plan_step.content)
            plan_steps = len(plan_data.get("steps", []))
        except (json.JSONDecodeError, TypeError):
            pass

        trial.plan = PlanValidation(
            minimal_passed=min_r.passed,
            medium_passed=med_r.passed,
            expansive_passed=exp_r.passed,
            minimal_feedback=min_r.feedback or "",
            medium_feedback=med_r.feedback or "",
            expansive_feedback=exp_r.feedback or "",
            plan_steps=plan_steps,
            plan_content=plan_step.content,
        )

        if not min_r.passed:
            trial.errors.append(f"Minimal: {min_r.feedback}")
        if min_r.passed and not med_r.passed:
            trial.errors.append(f"Medium: {med_r.feedback}")
        if med_r.passed and not exp_r.passed:
            trial.errors.append(f"Expansive: {exp_r.feedback}")

        # Pipeline succeeds if all pre-steps passed AND plan passes Medium
        trial.pipeline_succeeded = med_r.passed
        trial.total_time_ms = (time.perf_counter() - total_start) * 1000

        return trial

    def _run_step(
        self,
        action_pair_id: str,
        generator: LLMPlanGenerator | LLMJsonGenerator,
        guard: AnalysisGuard | ReconGuard | StrategyGuard | None,
        context: Context,
        template: PromptTemplate,
    ) -> StepResult:
        """Run a single pipeline step (generation + optional guard validation).

        When guard is None, only generation is performed (plan step uses
        separate three-level validation).
        """
        t0 = time.perf_counter()
        try:
            artifact = generator.generate(
                context=context,
                template=template,
                action_pair_id=action_pair_id,
                workflow_id="evaluation",
            )
            elapsed = (time.perf_counter() - t0) * 1000
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.warning("Step %s failed: %s", action_pair_id, e)
            return StepResult(passed=False, feedback=str(e), time_ms=elapsed)

        if guard is None:
            # No guard — just return the generated content
            return StepResult(passed=True, content=artifact.content, time_ms=elapsed)

        result = guard.validate(artifact)
        return StepResult(
            passed=result.passed,
            content=artifact.content,
            feedback=result.feedback or "",
            time_ms=elapsed,
        )
