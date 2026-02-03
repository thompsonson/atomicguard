"""Tests for evaluation scoring and aggregation."""

from __future__ import annotations

import pytest

from examples.advanced.g_plan_benchmark.evaluation.problem import Problem, ProblemSet
from examples.advanced.g_plan_benchmark.evaluation.results import (
    ExperimentConfig,
    ExperimentResult,
    PipelineResult,
    PlanValidation,
    ProblemTrialResult,
    StepResult,
)
from examples.advanced.g_plan_benchmark.evaluation.scoring import (
    EpsilonScore,
    ExperimentScorecard,
    StrategyAlignmentScore,
    score_experiment,
)


# =============================================================================
# EpsilonScore
# =============================================================================


class TestEpsilonScore:
    def test_perfect_epsilon(self):
        s = EpsilonScore(passed=10, total=10)
        assert s.epsilon_hat == 1.0
        assert s.e_attempts == 1.0

    def test_zero_epsilon(self):
        s = EpsilonScore(passed=0, total=10)
        assert s.epsilon_hat == 0.0
        assert s.e_attempts == float("inf")

    def test_partial_epsilon(self):
        s = EpsilonScore(passed=7, total=10)
        assert abs(s.epsilon_hat - 0.7) < 1e-9
        assert abs(s.e_attempts - 1 / 0.7) < 1e-6

    def test_empty_trials(self):
        s = EpsilonScore(passed=0, total=0)
        assert s.epsilon_hat == 0.0
        assert s.ci_95 == (0.0, 0.0)

    def test_ci_bounds(self):
        s = EpsilonScore(passed=5, total=10)
        lo, hi = s.ci_95
        assert 0.0 <= lo <= s.epsilon_hat <= hi <= 1.0

    def test_to_dict(self):
        s = EpsilonScore(passed=8, total=10)
        d = s.to_dict()
        assert d["passed"] == 8
        assert d["total"] == 10
        assert d["epsilon_hat"] == 0.8
        assert len(d["ci_95"]) == 2
        assert d["e_attempts"] == 1.25


class TestStrategyAlignmentScore:
    def test_perfect_alignment(self):
        s = StrategyAlignmentScore(correct=10, total=10)
        assert s.accuracy == 1.0

    def test_zero_alignment(self):
        s = StrategyAlignmentScore(correct=0, total=10)
        assert s.accuracy == 0.0

    def test_empty(self):
        s = StrategyAlignmentScore()
        assert s.accuracy == 0.0

    def test_to_dict(self):
        s = StrategyAlignmentScore(correct=7, total=10)
        d = s.to_dict()
        assert d["correct"] == 7
        assert d["total"] == 10
        assert abs(d["accuracy"] - 0.7) < 1e-3


# =============================================================================
# score_experiment
# =============================================================================


def _make_problem_set() -> ProblemSet:
    return ProblemSet([
        Problem(problem_id="BUG-1", description="Fix crash", expected_type="bug_fix"),
        Problem(problem_id="FEAT-1", description="Add feature", expected_type="feature"),
    ])


def _make_trial(
    problem_id: str,
    pipeline: str,
    trial_num: int = 1,
    *,
    pipeline_succeeded: bool = True,
    minimal_passed: bool = True,
    medium_passed: bool = True,
    expansive_passed: bool = True,
    plan_content: str = '{"steps":[]}',
    analysis_passed: bool | None = None,
    classified_type: str = "",
    selected_strategy: str = "",
) -> ProblemTrialResult:
    trial = ProblemTrialResult(
        problem_id=problem_id,
        pipeline=pipeline,
        trial_num=trial_num,
        total_time_ms=100.0,
        pipeline_succeeded=pipeline_succeeded,
        classified_type=classified_type,
        selected_strategy=selected_strategy,
        plan=PlanValidation(
            minimal_passed=minimal_passed,
            medium_passed=medium_passed,
            expansive_passed=expansive_passed,
            plan_content=plan_content,
        ),
    )
    if analysis_passed is not None:
        trial.analysis = StepResult(passed=analysis_passed)
    return trial


class TestScoreExperiment:
    def test_single_pipeline_all_pass(self):
        ps = _make_problem_set()
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["single"]),
            pipeline_results={
                "single": PipelineResult(
                    pipeline="single",
                    trials=[
                        _make_trial("BUG-1", "single"),
                        _make_trial("FEAT-1", "single"),
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["single"]

        assert card.pipeline_epsilon.passed == 2
        assert card.pipeline_epsilon.total == 2
        assert card.pipeline_epsilon.epsilon_hat == 1.0

    def test_single_pipeline_some_fail(self):
        ps = _make_problem_set()
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["single"]),
            pipeline_results={
                "single": PipelineResult(
                    pipeline="single",
                    trials=[
                        _make_trial("BUG-1", "single", pipeline_succeeded=True),
                        _make_trial("FEAT-1", "single", pipeline_succeeded=False, medium_passed=False),
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["single"]
        assert card.pipeline_epsilon.passed == 1
        assert card.pipeline_epsilon.total == 2

    def test_guard_level_epsilon(self):
        ps = _make_problem_set()
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["single"]),
            pipeline_results={
                "single": PipelineResult(
                    pipeline="single",
                    trials=[
                        _make_trial("BUG-1", "single", minimal_passed=True, medium_passed=True, expansive_passed=False),
                        _make_trial("FEAT-1", "single", minimal_passed=True, medium_passed=False, expansive_passed=False),
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["single"]
        assert card.minimal_epsilon.passed == 2
        assert card.medium_epsilon.passed == 1
        assert card.expansive_epsilon.passed == 0

    def test_prestep_epsilon(self):
        ps = _make_problem_set()
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["full"]),
            pipeline_results={
                "full": PipelineResult(
                    pipeline="full",
                    trials=[
                        _make_trial("BUG-1", "full", analysis_passed=True),
                        _make_trial("FEAT-1", "full", analysis_passed=False, pipeline_succeeded=False, plan_content=""),
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["full"]
        assert card.analysis_epsilon is not None
        assert card.analysis_epsilon.passed == 1
        assert card.analysis_epsilon.total == 2

    def test_strategy_alignment(self):
        ps = _make_problem_set()
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["full"]),
            pipeline_results={
                "full": PipelineResult(
                    pipeline="full",
                    trials=[
                        _make_trial("BUG-1", "full", selected_strategy="S1_locate_and_fix"),
                        _make_trial("FEAT-1", "full", selected_strategy="S1_locate_and_fix"),  # wrong
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["full"]
        # BUG-1 expects S1 (correct), FEAT-1 expects S2 (wrong)
        assert card.strategy_alignment.correct == 1
        assert card.strategy_alignment.total == 2

    def test_per_category_breakdown(self):
        ps = _make_problem_set()
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["single"]),
            pipeline_results={
                "single": PipelineResult(
                    pipeline="single",
                    trials=[
                        _make_trial("BUG-1", "single", pipeline_succeeded=True),
                        _make_trial("FEAT-1", "single", pipeline_succeeded=False, medium_passed=False),
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["single"]
        assert len(card.categories) == 2
        cat_dict = {c.category: c for c in card.categories}
        assert cat_dict["bug_fix"].pipeline_epsilon.epsilon_hat == 1.0
        assert cat_dict["feature"].pipeline_epsilon.epsilon_hat == 0.0

    def test_cross_pipeline_comparison(self):
        ps = _make_problem_set()
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["single", "full"]),
            pipeline_results={
                "single": PipelineResult(
                    pipeline="single",
                    trials=[
                        _make_trial("BUG-1", "single", pipeline_succeeded=False, medium_passed=False),
                        _make_trial("FEAT-1", "single", pipeline_succeeded=False, medium_passed=False),
                    ],
                ),
                "full": PipelineResult(
                    pipeline="full",
                    trials=[
                        _make_trial("BUG-1", "full", pipeline_succeeded=True),
                        _make_trial("FEAT-1", "full", pipeline_succeeded=True),
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        assert "delta_vs_baseline" in scorecard.comparison
        # Sorted pipeline names: full (baseline, eps=1.0), single (eps=0.0)
        # single_vs_full: 0% - 100% = -100pp
        delta = scorecard.comparison["delta_vs_baseline"]["single_vs_full"]
        assert delta["delta_pp"] == -100.0

    def test_empty_experiment(self):
        ps = ProblemSet([])
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["single"]),
            pipeline_results={
                "single": PipelineResult(pipeline="single"),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["single"]
        assert card.pipeline_epsilon.total == 0

    def test_scorecard_to_dict(self):
        ps = _make_problem_set()
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["single"]),
            pipeline_results={
                "single": PipelineResult(
                    pipeline="single",
                    trials=[_make_trial("BUG-1", "single")],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        d = scorecard.to_dict()
        assert "pipelines" in d
        assert "single" in d["pipelines"]
        assert d["pipelines"]["single"]["pipeline_epsilon"]["epsilon_hat"] == 1.0

    def test_multiple_trials_per_problem(self):
        ps = ProblemSet([Problem(problem_id="P1", description="d", expected_type="bug_fix")])
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["single"], trials_per_problem=3),
            pipeline_results={
                "single": PipelineResult(
                    pipeline="single",
                    trials=[
                        _make_trial("P1", "single", trial_num=1, pipeline_succeeded=True),
                        _make_trial("P1", "single", trial_num=2, pipeline_succeeded=False, medium_passed=False),
                        _make_trial("P1", "single", trial_num=3, pipeline_succeeded=True),
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["single"]
        assert card.pipeline_epsilon.passed == 2
        assert card.pipeline_epsilon.total == 3

    def test_guard_epsilon_excludes_trials_without_plan(self):
        """Guard-level epsilon should only count trials that reached plan generation.

        If a pre-step fails (plan_content is empty), that trial should not
        contribute to minimal/medium/expansive totals — only to pipeline_epsilon.
        """
        ps = ProblemSet([
            Problem(problem_id="P1", description="d", expected_type="bug_fix"),
            Problem(problem_id="P2", description="d", expected_type="bug_fix"),
            Problem(problem_id="P3", description="d", expected_type="bug_fix"),
        ])
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["full"]),
            pipeline_results={
                "full": PipelineResult(
                    pipeline="full",
                    trials=[
                        # P1: analysis failed, never reached plan generation
                        _make_trial(
                            "P1", "full",
                            analysis_passed=False,
                            pipeline_succeeded=False,
                            plan_content="",  # no plan generated
                            minimal_passed=False,
                            medium_passed=False,
                            expansive_passed=False,
                        ),
                        # P2: all pre-steps passed, plan passed Medium
                        _make_trial(
                            "P2", "full",
                            analysis_passed=True,
                            pipeline_succeeded=True,
                            plan_content='{"steps":[{"step_id":"s1"}]}',
                            minimal_passed=True,
                            medium_passed=True,
                            expansive_passed=True,
                        ),
                        # P3: all pre-steps passed, plan failed Minimal
                        _make_trial(
                            "P3", "full",
                            analysis_passed=True,
                            pipeline_succeeded=False,
                            plan_content='{"bad_plan": true}',
                            minimal_passed=False,
                            medium_passed=False,
                            expansive_passed=False,
                        ),
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["full"]

        # Pipeline epsilon: 1 out of 3 total trials succeeded
        assert card.pipeline_epsilon.passed == 1
        assert card.pipeline_epsilon.total == 3

        # Guard-level epsilon: only 2 trials reached plan generation (P2 and P3)
        # P1 has empty plan_content, so it's excluded
        assert card.minimal_epsilon.total == 2
        assert card.minimal_epsilon.passed == 1  # only P2
        assert card.medium_epsilon.total == 2
        assert card.medium_epsilon.passed == 1  # only P2

    def test_strategy_alignment_ignores_unknown_types(self):
        """Strategy alignment should skip problems with no expected strategy."""
        ps = ProblemSet([
            Problem(problem_id="P1", description="d", expected_type="bug_fix"),
            Problem(problem_id="P2", description="d", expected_type="unknown"),
        ])
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["full"]),
            pipeline_results={
                "full": PipelineResult(
                    pipeline="full",
                    trials=[
                        _make_trial("P1", "full", selected_strategy="S1_locate_and_fix"),
                        _make_trial("P2", "full", selected_strategy="S1_locate_and_fix"),
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["full"]
        # Only P1 has an expected strategy (bug_fix → S1).
        # P2 is "unknown" with no mapping, so it should be excluded.
        assert card.strategy_alignment.total == 1
        assert card.strategy_alignment.correct == 1

    def test_strategy_alignment_wrong_selection(self):
        """Strategy alignment correctly detects mismatched strategy selection."""
        ps = ProblemSet([
            Problem(problem_id="BUG", description="d", expected_type="bug_fix"),
            Problem(problem_id="FEAT", description="d", expected_type="feature"),
        ])
        result = ExperimentResult(
            config=ExperimentConfig(pipelines=["full"]),
            pipeline_results={
                "full": PipelineResult(
                    pipeline="full",
                    trials=[
                        # BUG selects S2 instead of expected S1 — wrong
                        _make_trial("BUG", "full", selected_strategy="S2_tdd_feature"),
                        # FEAT selects S2 — correct (feature → S2)
                        _make_trial("FEAT", "full", selected_strategy="S2_tdd_feature"),
                    ],
                ),
            },
        )

        scorecard = score_experiment(result, ps)
        card = scorecard.pipelines["full"]
        assert card.strategy_alignment.total == 2
        assert card.strategy_alignment.correct == 1  # only FEAT is correct
