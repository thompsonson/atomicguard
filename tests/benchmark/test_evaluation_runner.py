"""Tests for the evaluation experiment runner."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    ContextSnapshot,
)

from examples.advanced.g_plan_benchmark.evaluation.problem import Problem, ProblemSet
from examples.advanced.g_plan_benchmark.evaluation.results import (
    ExperimentConfig,
)
from examples.advanced.g_plan_benchmark.evaluation.runner import ExperimentRunner


# =============================================================================
# Helpers
# =============================================================================


def _make_artifact(content: str, action_pair_id: str = "g_plan_llm") -> Artifact:
    return Artifact(
        artifact_id="test-id",
        workflow_id="evaluation",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id=action_pair_id,
        created_at="",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=ContextSnapshot(
            workflow_id="evaluation",
            specification="test",
            constraints="",
            feedback_history=(),
        ),
    )


def _valid_plan_json() -> str:
    """A plan that passes MinimalPlanGuard and MediumPlanGuard."""
    return json.dumps({
        "plan_id": "test-plan",
        "initial_state": ["intent_received"],
        "goal_state": ["syntax"],
        "total_retry_budget": 10,
        "steps": [
            {
                "step_id": "s1",
                "name": "Write code",
                "generator": "OllamaGenerator",
                "guard": "syntax",
                "retry_budget": 3,
                "preconditions": ["intent_received"],
                "effects": ["syntax"],
                "dependencies": [],
            }
        ],
    })


def _valid_analysis_json() -> str:
    return json.dumps({
        "problem_type": "bug_fix",
        "language": "python",
        "severity": "medium",
        "key_signals": ["crash on startup"],
        "affected_area": "config parser",
        "rationale": "KeyError indicates missing config key",
    })


def _valid_recon_json() -> str:
    return json.dumps({
        "mentioned_files": ["config.py"],
        "stack_traces": [],
        "apis_involved": [],
        "test_references": [],
        "reproduction_steps": ["run app"],
        "constraints_mentioned": [],
    })


def _valid_strategy_json() -> str:
    return json.dumps({
        "strategy_id": "S1_locate_and_fix",
        "strategy_name": "Locate and Fix",
        "rationale": "Bug fix — locate the error source and patch it",
        "key_steps": ["find error", "write test", "fix"],
        "expected_guards": ["syntax", "test"],
        "risk_factors": [],
    })


def _make_problem_set() -> ProblemSet:
    return ProblemSet([
        Problem(
            problem_id="BUG-1",
            description="KeyError in parse_config()",
            expected_type="bug_fix",
            language="python",
        ),
    ])


# =============================================================================
# Tests
# =============================================================================


class TestExperimentRunnerSinglePipeline:
    """Test the runner with the 'single' pipeline (plan generation only)."""

    def test_single_pipeline_success(self):
        config = ExperimentConfig(
            pipelines=["single"],
            trials_per_problem=1,
            model="test-model",
        )
        runner = ExperimentRunner(config)

        # Mock generators
        mock_plan_gen = MagicMock()
        mock_plan_gen.generate.return_value = _make_artifact(_valid_plan_json())
        runner._plan_generator = mock_plan_gen
        runner._json_generator = MagicMock()

        ps = _make_problem_set()
        result = runner.run(ps)

        assert "single" in result.pipeline_results
        trials = result.pipeline_results["single"].trials
        assert len(trials) == 1
        assert trials[0].pipeline_succeeded is True
        assert trials[0].plan.medium_passed is True

    def test_single_pipeline_plan_fails_validation(self):
        config = ExperimentConfig(pipelines=["single"], trials_per_problem=1)
        runner = ExperimentRunner(config)

        mock_plan_gen = MagicMock()
        mock_plan_gen.generate.return_value = _make_artifact('{"invalid": true}')
        runner._plan_generator = mock_plan_gen
        runner._json_generator = MagicMock()

        result = runner.run(_make_problem_set())

        trial = result.pipeline_results["single"].trials[0]
        assert trial.pipeline_succeeded is False
        assert trial.plan.minimal_passed is False

    def test_single_pipeline_generation_exception(self):
        config = ExperimentConfig(pipelines=["single"], trials_per_problem=1)
        runner = ExperimentRunner(config)

        mock_plan_gen = MagicMock()
        mock_plan_gen.generate.side_effect = RuntimeError("LLM timeout")
        runner._plan_generator = mock_plan_gen
        runner._json_generator = MagicMock()

        result = runner.run(_make_problem_set())

        trial = result.pipeline_results["single"].trials[0]
        assert trial.pipeline_succeeded is False
        assert any("Plan generation" in e for e in trial.errors)


class TestExperimentRunnerFullPipeline:
    """Test the runner with the 'full' pipeline."""

    def _setup_runner(self):
        config = ExperimentConfig(pipelines=["full"], trials_per_problem=1)
        runner = ExperimentRunner(config)

        mock_json_gen = MagicMock()
        mock_plan_gen = MagicMock()
        runner._json_generator = mock_json_gen
        runner._plan_generator = mock_plan_gen

        return runner, mock_json_gen, mock_plan_gen

    def test_full_pipeline_success(self):
        runner, mock_json_gen, mock_plan_gen = self._setup_runner()

        # Each call to json_generator returns analysis, then recon, then strategy
        mock_json_gen.generate.side_effect = [
            _make_artifact(_valid_analysis_json(), "g_analysis"),
            _make_artifact(_valid_recon_json(), "g_recon"),
            _make_artifact(_valid_strategy_json(), "g_strategy"),
        ]
        mock_plan_gen.generate.return_value = _make_artifact(_valid_plan_json(), "g_plan_full")

        result = runner.run(_make_problem_set())

        trial = result.pipeline_results["full"].trials[0]
        assert trial.pipeline_succeeded is True
        assert trial.analysis is not None
        assert trial.analysis.passed is True
        assert trial.recon is not None
        assert trial.recon.passed is True
        assert trial.strategy is not None
        assert trial.strategy.passed is True
        assert trial.classified_type == "bug_fix"
        assert trial.selected_strategy == "S1_locate_and_fix"

    def test_full_pipeline_analysis_failure_aborts(self):
        runner, mock_json_gen, mock_plan_gen = self._setup_runner()

        # Analysis returns invalid JSON
        mock_json_gen.generate.return_value = _make_artifact(
            '{"bad": "data"}', "g_analysis"
        )

        result = runner.run(_make_problem_set())

        trial = result.pipeline_results["full"].trials[0]
        assert trial.pipeline_succeeded is False
        assert trial.analysis is not None
        assert trial.analysis.passed is False
        # Recon should not have been attempted
        assert trial.recon is None
        assert trial.strategy is None

    def test_full_pipeline_recon_failure_aborts(self):
        runner, mock_json_gen, mock_plan_gen = self._setup_runner()

        mock_json_gen.generate.side_effect = [
            _make_artifact(_valid_analysis_json(), "g_analysis"),
            # Recon returns invalid — all lists empty
            _make_artifact(json.dumps({
                "mentioned_files": [], "stack_traces": [],
                "apis_involved": [], "test_references": [],
                "reproduction_steps": [], "constraints_mentioned": [],
            }), "g_recon"),
        ]

        result = runner.run(_make_problem_set())

        trial = result.pipeline_results["full"].trials[0]
        assert trial.pipeline_succeeded is False
        assert trial.analysis.passed is True  # type: ignore[union-attr]
        assert trial.recon is not None
        assert trial.recon.passed is False
        assert trial.strategy is None

    def test_full_pipeline_strategy_failure_aborts(self):
        runner, mock_json_gen, mock_plan_gen = self._setup_runner()

        mock_json_gen.generate.side_effect = [
            _make_artifact(_valid_analysis_json(), "g_analysis"),
            _make_artifact(_valid_recon_json(), "g_recon"),
            # Strategy returns invalid strategy_id
            _make_artifact(json.dumps({
                "strategy_id": "INVALID",
                "strategy_name": "Bad",
                "rationale": "r",
                "key_steps": ["s"],
                "expected_guards": ["g"],
                "risk_factors": [],
            }), "g_strategy"),
        ]

        result = runner.run(_make_problem_set())

        trial = result.pipeline_results["full"].trials[0]
        assert trial.pipeline_succeeded is False
        assert trial.analysis.passed is True  # type: ignore[union-attr]
        assert trial.recon.passed is True  # type: ignore[union-attr]
        assert trial.strategy is not None
        assert trial.strategy.passed is False


class TestExperimentRunnerMultiPipeline:
    """Test running multiple pipelines in one experiment."""

    def test_two_pipelines(self):
        config = ExperimentConfig(
            pipelines=["single", "full"],
            trials_per_problem=1,
        )
        runner = ExperimentRunner(config)

        mock_json_gen = MagicMock()
        mock_plan_gen = MagicMock()
        runner._json_generator = mock_json_gen
        runner._plan_generator = mock_plan_gen

        plan_artifact = _make_artifact(_valid_plan_json())
        mock_plan_gen.generate.return_value = plan_artifact

        mock_json_gen.generate.side_effect = [
            _make_artifact(_valid_analysis_json(), "g_analysis"),
            _make_artifact(_valid_recon_json(), "g_recon"),
            _make_artifact(_valid_strategy_json(), "g_strategy"),
        ]

        result = runner.run(_make_problem_set())

        assert "single" in result.pipeline_results
        assert "full" in result.pipeline_results
        assert len(result.pipeline_results["single"].trials) == 1
        assert len(result.pipeline_results["full"].trials) == 1


class TestExperimentRunnerCallback:
    """Test the on_trial callback."""

    def test_callback_called(self):
        config = ExperimentConfig(pipelines=["single"], trials_per_problem=1)
        runner = ExperimentRunner(config)

        mock_plan_gen = MagicMock()
        mock_plan_gen.generate.return_value = _make_artifact(_valid_plan_json())
        runner._plan_generator = mock_plan_gen
        runner._json_generator = MagicMock()

        calls = []

        def on_trial(problem_id, pipeline, trial_num, result):
            calls.append((problem_id, pipeline, trial_num))

        runner.run(_make_problem_set(), on_trial=on_trial)
        assert len(calls) == 1
        assert calls[0] == ("BUG-1", "single", 1)


class TestExperimentRunnerClassifyThenPlan:
    """Test the 'classify-then-plan' pipeline (analysis + plan, no recon/strategy)."""

    def test_classify_then_plan_success(self):
        config = ExperimentConfig(pipelines=["classify-then-plan"], trials_per_problem=1)
        runner = ExperimentRunner(config)

        mock_json_gen = MagicMock()
        mock_plan_gen = MagicMock()
        runner._json_generator = mock_json_gen
        runner._plan_generator = mock_plan_gen

        mock_json_gen.generate.return_value = _make_artifact(
            _valid_analysis_json(), "g_analysis"
        )
        mock_plan_gen.generate.return_value = _make_artifact(
            _valid_plan_json(), "g_plan_conditioned"
        )

        result = runner.run(_make_problem_set())

        trial = result.pipeline_results["classify-then-plan"].trials[0]
        assert trial.pipeline_succeeded is True
        assert trial.analysis is not None
        assert trial.analysis.passed is True
        # classify-then-plan skips recon and strategy
        assert trial.recon is None
        assert trial.strategy is None
        assert trial.classified_type == "bug_fix"

    def test_classify_then_plan_analysis_failure(self):
        config = ExperimentConfig(pipelines=["classify-then-plan"], trials_per_problem=1)
        runner = ExperimentRunner(config)

        mock_json_gen = MagicMock()
        mock_plan_gen = MagicMock()
        runner._json_generator = mock_json_gen
        runner._plan_generator = mock_plan_gen

        # Analysis returns garbage
        mock_json_gen.generate.return_value = _make_artifact(
            '{"not_valid": true}', "g_analysis"
        )

        result = runner.run(_make_problem_set())

        trial = result.pipeline_results["classify-then-plan"].trials[0]
        assert trial.pipeline_succeeded is False
        assert trial.analysis is not None
        assert trial.analysis.passed is False
        # Plan should not have been attempted
        assert trial.plan.plan_content == ""


class TestExperimentRunnerMultipleProblems:
    """Test running across multiple problems."""

    def test_two_problems_single_pipeline(self):
        config = ExperimentConfig(pipelines=["single"], trials_per_problem=1)
        runner = ExperimentRunner(config)

        mock_plan_gen = MagicMock()
        mock_plan_gen.generate.return_value = _make_artifact(_valid_plan_json())
        runner._plan_generator = mock_plan_gen
        runner._json_generator = MagicMock()

        ps = ProblemSet([
            Problem(problem_id="BUG-1", description="Fix crash", expected_type="bug_fix"),
            Problem(problem_id="FEAT-1", description="Add search", expected_type="feature"),
        ])
        result = runner.run(ps)

        trials = result.pipeline_results["single"].trials
        assert len(trials) == 2
        assert {t.problem_id for t in trials} == {"BUG-1", "FEAT-1"}
        # Both get the same valid plan, so both should succeed
        assert all(t.pipeline_succeeded for t in trials)

    def test_two_problems_three_trials_each(self):
        config = ExperimentConfig(pipelines=["single"], trials_per_problem=3)
        runner = ExperimentRunner(config)

        mock_plan_gen = MagicMock()
        mock_plan_gen.generate.return_value = _make_artifact(_valid_plan_json())
        runner._plan_generator = mock_plan_gen
        runner._json_generator = MagicMock()

        ps = ProblemSet([
            Problem(problem_id="P1", description="d1"),
            Problem(problem_id="P2", description="d2"),
        ])
        result = runner.run(ps)

        trials = result.pipeline_results["single"].trials
        assert len(trials) == 6  # 2 problems × 3 trials
        # Check that each problem gets 3 trials
        p1_trials = [t for t in trials if t.problem_id == "P1"]
        p2_trials = [t for t in trials if t.problem_id == "P2"]
        assert len(p1_trials) == 3
        assert len(p2_trials) == 3

    def test_problem_description_reaches_context(self):
        """Verify each problem's description is used as the specification."""
        config = ExperimentConfig(pipelines=["single"], trials_per_problem=1)
        runner = ExperimentRunner(config)

        captured_contexts = []

        def _capture_generate(**kwargs):
            captured_contexts.append(kwargs.get("context"))
            return _make_artifact(_valid_plan_json())

        mock_plan_gen = MagicMock()
        mock_plan_gen.generate.side_effect = lambda **kwargs: _capture_generate(**kwargs)
        runner._plan_generator = mock_plan_gen
        runner._json_generator = MagicMock()

        ps = ProblemSet([
            Problem(problem_id="P1", description="Fix the KeyError bug"),
            Problem(problem_id="P2", description="Add dark mode toggle"),
        ])
        runner.run(ps)

        assert len(captured_contexts) == 2
        assert captured_contexts[0].specification == "Fix the KeyError bug"
        assert captured_contexts[1].specification == "Add dark mode toggle"


class TestExperimentRunnerMultipleTrials:
    """Test multiple trials per problem."""

    def test_three_trials(self):
        config = ExperimentConfig(pipelines=["single"], trials_per_problem=3)
        runner = ExperimentRunner(config)

        mock_plan_gen = MagicMock()
        mock_plan_gen.generate.return_value = _make_artifact(_valid_plan_json())
        runner._plan_generator = mock_plan_gen
        runner._json_generator = MagicMock()

        result = runner.run(_make_problem_set())

        trials = result.pipeline_results["single"].trials
        assert len(trials) == 3
        assert [t.trial_num for t in trials] == [1, 2, 3]
        assert all(t.problem_id == "BUG-1" for t in trials)
