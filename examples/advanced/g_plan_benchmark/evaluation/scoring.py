"""Scoring and aggregation for evaluation experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .problem import TYPE_TO_STRATEGY, ProblemSet
from .results import ExperimentResult, PipelineResult, ProblemTrialResult


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


@dataclass
class EpsilonScore:
    """Epsilon (pass rate) with Wilson confidence interval."""

    passed: int
    total: int

    @property
    def epsilon_hat(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def ci_95(self) -> tuple[float, float]:
        return _wilson_ci(self.passed, self.total)

    @property
    def e_attempts(self) -> float:
        """Expected number of attempts to get one pass: 1/epsilon."""
        return 1.0 / self.epsilon_hat if self.epsilon_hat > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        ci = self.ci_95
        return {
            "passed": self.passed,
            "total": self.total,
            "epsilon_hat": round(self.epsilon_hat, 4),
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
            "e_attempts": round(self.e_attempts, 2) if self.e_attempts != float("inf") else None,
        }


@dataclass
class StrategyAlignmentScore:
    """Strategy alignment: how often does the pipeline select the expected strategy?"""

    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def ci_95(self) -> tuple[float, float]:
        return _wilson_ci(self.correct, self.total)

    def to_dict(self) -> dict[str, Any]:
        ci = self.ci_95
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
        }


@dataclass
class CategoryBreakdown:
    """Per-category breakdown of epsilon and strategy alignment."""

    category: str
    pipeline_epsilon: EpsilonScore
    minimal_epsilon: EpsilonScore
    medium_epsilon: EpsilonScore
    expansive_epsilon: EpsilonScore
    strategy_alignment: StrategyAlignmentScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "pipeline_epsilon": self.pipeline_epsilon.to_dict(),
            "minimal_epsilon": self.minimal_epsilon.to_dict(),
            "medium_epsilon": self.medium_epsilon.to_dict(),
            "expansive_epsilon": self.expansive_epsilon.to_dict(),
            "strategy_alignment": self.strategy_alignment.to_dict(),
        }


@dataclass
class PipelineScorecard:
    """Scorecard for one pipeline mode."""

    pipeline: str

    # Overall epsilon (pipeline success = all presteps + Medium pass)
    pipeline_epsilon: EpsilonScore = field(default_factory=lambda: EpsilonScore(0, 0))

    # Guard-level epsilon
    minimal_epsilon: EpsilonScore = field(default_factory=lambda: EpsilonScore(0, 0))
    medium_epsilon: EpsilonScore = field(default_factory=lambda: EpsilonScore(0, 0))
    expansive_epsilon: EpsilonScore = field(default_factory=lambda: EpsilonScore(0, 0))

    # Pre-step epsilon (None if pipeline doesn't have that step)
    analysis_epsilon: EpsilonScore | None = None
    recon_epsilon: EpsilonScore | None = None
    strategy_epsilon: EpsilonScore | None = None

    # Strategy alignment (only for pipelines with g_strategy)
    strategy_alignment: StrategyAlignmentScore = field(
        default_factory=StrategyAlignmentScore
    )

    # Per-category breakdown
    categories: list[CategoryBreakdown] = field(default_factory=list)

    # Timing
    avg_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "pipeline": self.pipeline,
            "pipeline_epsilon": self.pipeline_epsilon.to_dict(),
            "minimal_epsilon": self.minimal_epsilon.to_dict(),
            "medium_epsilon": self.medium_epsilon.to_dict(),
            "expansive_epsilon": self.expansive_epsilon.to_dict(),
            "avg_time_ms": round(self.avg_time_ms, 1),
        }
        if self.analysis_epsilon is not None:
            d["analysis_epsilon"] = self.analysis_epsilon.to_dict()
        if self.recon_epsilon is not None:
            d["recon_epsilon"] = self.recon_epsilon.to_dict()
        if self.strategy_epsilon is not None:
            d["strategy_epsilon"] = self.strategy_epsilon.to_dict()
        if self.strategy_alignment.total > 0:
            d["strategy_alignment"] = self.strategy_alignment.to_dict()
        if self.categories:
            d["categories"] = [c.to_dict() for c in self.categories]
        return d


@dataclass
class ExperimentScorecard:
    """Complete scorecard for an experiment across all pipelines."""

    pipelines: dict[str, PipelineScorecard] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "pipelines": {k: v.to_dict() for k, v in self.pipelines.items()},
        }
        if self.comparison:
            d["comparison"] = self.comparison
        return d


def _score_pipeline(
    pipeline_result: PipelineResult,
    problem_set: ProblemSet,
) -> PipelineScorecard:
    """Score a single pipeline's results."""
    trials = pipeline_result.trials
    n = len(trials)
    pipeline = pipeline_result.pipeline

    card = PipelineScorecard(pipeline=pipeline)

    if n == 0:
        return card

    # --- Overall epsilon ---
    pipeline_pass = sum(1 for t in trials if t.pipeline_succeeded)
    card.pipeline_epsilon = EpsilonScore(pipeline_pass, n)

    # --- Guard-level epsilon (only count trials that reached plan generation) ---
    # A trial "reaches plan generation" if it has a non-empty plan_content
    plan_trials = [t for t in trials if t.plan.plan_content]
    np = len(plan_trials)
    card.minimal_epsilon = EpsilonScore(
        sum(1 for t in plan_trials if t.plan.minimal_passed), np
    )
    card.medium_epsilon = EpsilonScore(
        sum(1 for t in plan_trials if t.plan.medium_passed), np
    )
    card.expansive_epsilon = EpsilonScore(
        sum(1 for t in plan_trials if t.plan.expansive_passed), np
    )

    # --- Pre-step epsilon ---
    analysis_trials = [t for t in trials if t.analysis is not None]
    if analysis_trials:
        card.analysis_epsilon = EpsilonScore(
            sum(1 for t in analysis_trials if t.analysis.passed),  # type: ignore[union-attr]
            len(analysis_trials),
        )

    recon_trials = [t for t in trials if t.recon is not None]
    if recon_trials:
        card.recon_epsilon = EpsilonScore(
            sum(1 for t in recon_trials if t.recon.passed),  # type: ignore[union-attr]
            len(recon_trials),
        )

    strategy_trials = [t for t in trials if t.strategy is not None]
    if strategy_trials:
        card.strategy_epsilon = EpsilonScore(
            sum(1 for t in strategy_trials if t.strategy.passed),  # type: ignore[union-attr]
            len(strategy_trials),
        )

    # --- Strategy alignment ---
    # Build lookup from problem_id → expected_strategy
    problem_lookup: dict[str, str] = {}
    for problem in problem_set:
        expected = problem.expected_strategy
        if not expected and problem.expected_type in TYPE_TO_STRATEGY:
            expected = TYPE_TO_STRATEGY[problem.expected_type]
        if expected:
            problem_lookup[problem.problem_id] = expected

    alignment = StrategyAlignmentScore()
    for t in trials:
        if t.selected_strategy and t.problem_id in problem_lookup:
            alignment.total += 1
            if t.selected_strategy == problem_lookup[t.problem_id]:
                alignment.correct += 1
    card.strategy_alignment = alignment

    # --- Per-category breakdown ---
    # Group trials by expected_type
    categories_seen: set[str] = set()
    trial_by_cat: dict[str, list[ProblemTrialResult]] = {}
    for t in trials:
        # Look up the problem's expected_type
        try:
            prob = problem_set[t.problem_id]
            cat = prob.expected_type
        except KeyError:
            cat = "unknown"
        categories_seen.add(cat)
        trial_by_cat.setdefault(cat, []).append(t)

    for cat in sorted(categories_seen):
        cat_trials = trial_by_cat[cat]
        nc = len(cat_trials)
        cat_plan_trials = [t for t in cat_trials if t.plan.plan_content]
        ncp = len(cat_plan_trials)

        expected_strat = TYPE_TO_STRATEGY.get(cat, "")
        cat_align = StrategyAlignmentScore()
        for t in cat_trials:
            if t.selected_strategy and expected_strat:
                cat_align.total += 1
                if t.selected_strategy == expected_strat:
                    cat_align.correct += 1

        card.categories.append(
            CategoryBreakdown(
                category=cat,
                pipeline_epsilon=EpsilonScore(
                    sum(1 for t in cat_trials if t.pipeline_succeeded), nc,
                ),
                minimal_epsilon=EpsilonScore(
                    sum(1 for t in cat_plan_trials if t.plan.minimal_passed), ncp,
                ),
                medium_epsilon=EpsilonScore(
                    sum(1 for t in cat_plan_trials if t.plan.medium_passed), ncp,
                ),
                expansive_epsilon=EpsilonScore(
                    sum(1 for t in cat_plan_trials if t.plan.expansive_passed), ncp,
                ),
                strategy_alignment=cat_align,
            )
        )

    # --- Timing ---
    card.avg_time_ms = sum(t.total_time_ms for t in trials) / n

    return card


def score_experiment(
    result: ExperimentResult,
    problem_set: ProblemSet,
) -> ExperimentScorecard:
    """Score an entire experiment, producing per-pipeline and comparison scorecards.

    Args:
        result: The raw experiment results.
        problem_set: The problem set used (needed for expected_type lookups).

    Returns:
        ExperimentScorecard with per-pipeline scores and cross-pipeline comparison.
    """
    scorecard = ExperimentScorecard()

    for pipeline_name, pipeline_result in result.pipeline_results.items():
        scorecard.pipelines[pipeline_name] = _score_pipeline(pipeline_result, problem_set)

    # --- Cross-pipeline comparison ---
    if len(scorecard.pipelines) >= 2:
        comparison: dict[str, Any] = {}
        pipeline_names = sorted(scorecard.pipelines.keys())

        # Epsilon comparison
        comparison["pipeline_epsilon"] = {
            name: scorecard.pipelines[name].pipeline_epsilon.to_dict()
            for name in pipeline_names
        }

        # Delta: each pipeline vs the first (usually "single")
        baseline_name = pipeline_names[0]
        baseline_eps = scorecard.pipelines[baseline_name].pipeline_epsilon.epsilon_hat
        comparison["delta_vs_baseline"] = {}
        for name in pipeline_names[1:]:
            other_eps = scorecard.pipelines[name].pipeline_epsilon.epsilon_hat
            comparison["delta_vs_baseline"][f"{name}_vs_{baseline_name}"] = {
                "delta_pp": round((other_eps - baseline_eps) * 100, 2),
                "baseline": baseline_name,
                "baseline_epsilon": round(baseline_eps, 4),
                "other": name,
                "other_epsilon": round(other_eps, 4),
            }

        scorecard.comparison = comparison

    return scorecard
