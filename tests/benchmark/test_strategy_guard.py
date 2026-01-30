"""Tests for StrategyGuard (strategy_valid)."""

import json

import pytest

from atomicguard.domain.models import Artifact, ArtifactStatus, ContextSnapshot

from examples.advanced.g_plan_benchmark.guards.strategy import (
    VALID_STRATEGY_IDS,
    StrategyGuard,
)


@pytest.fixture
def guard() -> StrategyGuard:
    return StrategyGuard()


@pytest.fixture
def snapshot() -> ContextSnapshot:
    return ContextSnapshot(
        workflow_id="test",
        specification="test",
        constraints="",
        feedback_history=(),
    )


def _make_artifact(content: str, snapshot: ContextSnapshot) -> Artifact:
    return Artifact(
        artifact_id="test-001",
        workflow_id="test",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_strategy",
        created_at="2026-01-30T00:00:00",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=snapshot,
    )


VALID_STRATEGY = {
    "strategy_id": "S1_locate_and_fix",
    "strategy_name": "Locate and Fix Bug",
    "rationale": "Problem is a TypeError in the auth module, classic bug fix.",
    "key_steps": [
        "Locate the defect using stack trace",
        "Write characterization test",
        "Fix the TypeError",
        "Run regression tests",
    ],
    "expected_guards": ["syntax", "dynamic_test", "composite_validation"],
    "risk_factors": ["Fix might break other auth flows"],
}


class TestStrategyGuardValid:
    def test_valid_strategy_passes(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_STRATEGY), snapshot)
        result = guard.validate(artifact)
        assert result.passed is True
        assert "StrategyGuard" == result.guard_name

    def test_all_strategy_ids(self, guard, snapshot):
        for sid in VALID_STRATEGY_IDS:
            data = {**VALID_STRATEGY, "strategy_id": sid}
            artifact = _make_artifact(json.dumps(data), snapshot)
            result = guard.validate(artifact)
            assert result.passed is True, f"strategy_id={sid} should pass"

    def test_feedback_includes_id_and_name(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_STRATEGY), snapshot)
        result = guard.validate(artifact)
        assert "S1_locate_and_fix" in result.feedback
        assert "Locate and Fix Bug" in result.feedback

    def test_feedback_includes_counts(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_STRATEGY), snapshot)
        result = guard.validate(artifact)
        assert "4 steps" in result.feedback
        assert "3 guards" in result.feedback

    def test_empty_risk_factors_ok(self, guard, snapshot):
        """risk_factors may be empty (low-risk strategy)."""
        data = {**VALID_STRATEGY, "risk_factors": []}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is True


class TestStrategyGuardInvalid:
    def test_not_json(self, guard, snapshot):
        artifact = _make_artifact("not json", snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "not parseable" in result.feedback

    def test_not_object(self, guard, snapshot):
        artifact = _make_artifact("42", snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "JSON object" in result.feedback

    def test_missing_strategy_id(self, guard, snapshot):
        data = {k: v for k, v in VALID_STRATEGY.items() if k != "strategy_id"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "strategy_id" in result.feedback

    def test_invalid_strategy_id(self, guard, snapshot):
        data = {**VALID_STRATEGY, "strategy_id": "S99_invalid"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "Invalid strategy_id" in result.feedback

    def test_missing_strategy_name(self, guard, snapshot):
        data = {k: v for k, v in VALID_STRATEGY.items() if k != "strategy_name"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "strategy_name" in result.feedback

    def test_empty_strategy_name(self, guard, snapshot):
        data = {**VALID_STRATEGY, "strategy_name": "  "}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False

    def test_missing_rationale(self, guard, snapshot):
        data = {k: v for k, v in VALID_STRATEGY.items() if k != "rationale"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "rationale" in result.feedback

    def test_empty_key_steps(self, guard, snapshot):
        data = {**VALID_STRATEGY, "key_steps": []}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "non-empty" in result.feedback

    def test_key_steps_not_list(self, guard, snapshot):
        data = {**VALID_STRATEGY, "key_steps": "not a list"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "must be a list" in result.feedback

    def test_empty_expected_guards(self, guard, snapshot):
        data = {**VALID_STRATEGY, "expected_guards": []}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "non-empty" in result.feedback

    def test_missing_risk_factors(self, guard, snapshot):
        data = {k: v for k, v in VALID_STRATEGY.items() if k != "risk_factors"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "risk_factors" in result.feedback

    def test_risk_factors_not_list(self, guard, snapshot):
        data = {**VALID_STRATEGY, "risk_factors": "not a list"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "must be a list" in result.feedback

    def test_multiple_errors(self, guard, snapshot):
        data = {"strategy_id": "S1_locate_and_fix"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "strategy_name" in result.feedback
        assert "key_steps" in result.feedback


class TestStrategyGuardEdgeCases:
    def test_extra_fields_allowed(self, guard, snapshot):
        data = {**VALID_STRATEGY, "extra": "ignored"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_guard_name(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_STRATEGY), snapshot)
        result = guard.validate(artifact)
        assert result.guard_name == "StrategyGuard"

    def test_guard_name_on_failure(self, guard, snapshot):
        artifact = _make_artifact("bad", snapshot)
        result = guard.validate(artifact)
        assert result.guard_name == "StrategyGuard"
