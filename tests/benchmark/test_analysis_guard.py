"""Tests for AnalysisGuard (analysis_valid)."""

import json

import pytest

from atomicguard.domain.models import Artifact, ArtifactStatus, ContextSnapshot

from examples.advanced.g_plan_benchmark.guards.analysis import (
    VALID_LANGUAGES,
    VALID_PROBLEM_TYPES,
    VALID_SEVERITIES,
    AnalysisGuard,
)


@pytest.fixture
def guard() -> AnalysisGuard:
    """Create an AnalysisGuard instance."""
    return AnalysisGuard()


@pytest.fixture
def snapshot() -> ContextSnapshot:
    """Minimal context snapshot for test artifacts."""
    return ContextSnapshot(
        workflow_id="test",
        specification="test",
        constraints="",
        feedback_history=(),
    )


def _make_artifact(content: str, snapshot: ContextSnapshot) -> Artifact:
    """Helper to wrap content as an Artifact."""
    return Artifact(
        artifact_id="test-001",
        workflow_id="test",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_analysis",
        created_at="2026-01-30T00:00:00",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=snapshot,
    )


VALID_ANALYSIS = {
    "problem_type": "bug_fix",
    "language": "python",
    "severity": "high",
    "key_signals": ["TypeError in line 42", "stack trace points to auth module"],
    "affected_area": "authentication middleware",
    "rationale": "Stack trace indicates a type error in the auth module during login.",
}


class TestAnalysisGuardValid:
    """Tests for valid analysis artifacts."""

    def test_valid_analysis_passes(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_ANALYSIS), snapshot)
        result = guard.validate(artifact)
        assert result.passed is True
        assert "AnalysisGuard" == result.guard_name

    def test_all_problem_types(self, guard, snapshot):
        for pt in VALID_PROBLEM_TYPES:
            data = {**VALID_ANALYSIS, "problem_type": pt}
            artifact = _make_artifact(json.dumps(data), snapshot)
            result = guard.validate(artifact)
            assert result.passed is True, f"problem_type={pt} should pass"

    def test_all_severities(self, guard, snapshot):
        for sev in VALID_SEVERITIES:
            data = {**VALID_ANALYSIS, "severity": sev}
            artifact = _make_artifact(json.dumps(data), snapshot)
            result = guard.validate(artifact)
            assert result.passed is True, f"severity={sev} should pass"

    def test_all_languages(self, guard, snapshot):
        for lang in VALID_LANGUAGES:
            data = {**VALID_ANALYSIS, "language": lang}
            artifact = _make_artifact(json.dumps(data), snapshot)
            result = guard.validate(artifact)
            assert result.passed is True, f"language={lang} should pass"

    def test_feedback_includes_classification(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_ANALYSIS), snapshot)
        result = guard.validate(artifact)
        assert "bug_fix" in result.feedback
        assert "python" in result.feedback

    def test_feedback_includes_signal_count(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_ANALYSIS), snapshot)
        result = guard.validate(artifact)
        assert "2 signals" in result.feedback

    def test_single_signal(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "key_signals": ["single signal"]}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is True
        assert "1 signals" in result.feedback


class TestAnalysisGuardInvalid:
    """Tests for invalid analysis artifacts."""

    def test_not_json(self, guard, snapshot):
        artifact = _make_artifact("this is not json", snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "not parseable" in result.feedback

    def test_empty_string(self, guard, snapshot):
        artifact = _make_artifact("", snapshot)
        result = guard.validate(artifact)
        assert result.passed is False

    def test_json_array_not_object(self, guard, snapshot):
        artifact = _make_artifact("[1, 2, 3]", snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "JSON object" in result.feedback

    def test_missing_problem_type(self, guard, snapshot):
        data = {k: v for k, v in VALID_ANALYSIS.items() if k != "problem_type"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "problem_type" in result.feedback

    def test_invalid_problem_type(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "problem_type": "unknown_type"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "Invalid problem_type" in result.feedback

    def test_missing_language(self, guard, snapshot):
        data = {k: v for k, v in VALID_ANALYSIS.items() if k != "language"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "language" in result.feedback

    def test_empty_language(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "language": ""}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "language" in result.feedback

    def test_invalid_language(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "language": "rust"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "Invalid language" in result.feedback

    def test_missing_severity(self, guard, snapshot):
        data = {k: v for k, v in VALID_ANALYSIS.items() if k != "severity"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "severity" in result.feedback

    def test_invalid_severity(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "severity": "critical"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "Invalid severity" in result.feedback

    def test_missing_key_signals(self, guard, snapshot):
        data = {k: v for k, v in VALID_ANALYSIS.items() if k != "key_signals"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "key_signals" in result.feedback

    def test_key_signals_not_list(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "key_signals": "not a list"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "must be a list" in result.feedback

    def test_key_signals_empty_list(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "key_signals": []}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "non-empty" in result.feedback

    def test_missing_affected_area(self, guard, snapshot):
        data = {k: v for k, v in VALID_ANALYSIS.items() if k != "affected_area"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "affected_area" in result.feedback

    def test_empty_affected_area(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "affected_area": "  "}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "affected_area" in result.feedback

    def test_missing_rationale(self, guard, snapshot):
        data = {k: v for k, v in VALID_ANALYSIS.items() if k != "rationale"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "rationale" in result.feedback

    def test_empty_rationale(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "rationale": ""}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "rationale" in result.feedback

    def test_multiple_errors_reported(self, guard, snapshot):
        """Missing multiple fields should report all errors."""
        data = {"problem_type": "bug_fix"}  # Missing everything else
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        # Should mention multiple missing fields
        assert "language" in result.feedback
        assert "key_signals" in result.feedback
        assert "severity" in result.feedback


class TestAnalysisGuardEdgeCases:
    """Edge case tests."""

    def test_extra_fields_allowed(self, guard, snapshot):
        """Extra fields beyond the schema should not cause failure."""
        data = {**VALID_ANALYSIS, "extra_field": "ignored"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_none_content(self, guard, snapshot):
        """None content should fail gracefully."""
        artifact = _make_artifact("null", snapshot)
        result = guard.validate(artifact)
        assert result.passed is False

    def test_numeric_content(self, guard, snapshot):
        """Numeric JSON should fail."""
        artifact = _make_artifact("42", snapshot)
        result = guard.validate(artifact)
        assert result.passed is False

    def test_language_whitespace_only(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "language": "   "}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False

    def test_affected_area_not_string(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "affected_area": 42}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False

    def test_rationale_not_string(self, guard, snapshot):
        data = {**VALID_ANALYSIS, "rationale": ["not", "a", "string"]}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False

    def test_guard_name_set(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_ANALYSIS), snapshot)
        result = guard.validate(artifact)
        assert result.guard_name == "AnalysisGuard"

    def test_guard_name_on_failure(self, guard, snapshot):
        artifact = _make_artifact("not json", snapshot)
        result = guard.validate(artifact)
        assert result.guard_name == "AnalysisGuard"
