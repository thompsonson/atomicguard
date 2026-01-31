"""Tests for ReconGuard (recon_valid)."""

import json

import pytest

from atomicguard.domain.models import Artifact, ArtifactStatus, ContextSnapshot

from examples.advanced.g_plan_benchmark.guards.recon import (
    REQUIRED_LIST_FIELDS,
    ReconGuard,
)


@pytest.fixture
def guard() -> ReconGuard:
    return ReconGuard()


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
        action_pair_id="g_recon",
        created_at="2026-01-30T00:00:00",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=snapshot,
    )


VALID_RECON = {
    "mentioned_files": ["src/auth/handler.py", "tests/test_auth.py"],
    "stack_traces": ["TypeError: NoneType has no attribute 'login'"],
    "apis_involved": ["AuthHandler.login", "SessionManager.create"],
    "test_references": ["test_login_success", "test_login_failure"],
    "reproduction_steps": ["1. Call login with None user", "2. Observe TypeError"],
    "constraints_mentioned": ["Must support Python 3.8+"],
}


class TestReconGuardValid:
    def test_valid_recon_passes(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_RECON), snapshot)
        result = guard.validate(artifact)
        assert result.passed is True
        assert "ReconGuard" == result.guard_name

    def test_feedback_includes_counts(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_RECON), snapshot)
        result = guard.validate(artifact)
        total = sum(len(VALID_RECON[f]) for f in REQUIRED_LIST_FIELDS)
        assert str(total) in result.feedback
        assert "6 fields" in result.feedback

    def test_single_field_populated(self, guard, snapshot):
        """Only mentioned_files populated — should still pass."""
        data = {f: [] for f in REQUIRED_LIST_FIELDS}
        data["mentioned_files"] = ["foo.py"]
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is True
        assert "1 items" in result.feedback
        assert "1 fields" in result.feedback

    def test_sparse_recon(self, guard, snapshot):
        """Multiple fields empty, only two populated."""
        data = {f: [] for f in REQUIRED_LIST_FIELDS}
        data["apis_involved"] = ["foo"]
        data["stack_traces"] = ["bar"]
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is True


class TestReconGuardInvalid:
    def test_not_json(self, guard, snapshot):
        artifact = _make_artifact("not json", snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "not parseable" in result.feedback

    def test_not_object(self, guard, snapshot):
        artifact = _make_artifact("[1, 2]", snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "JSON object" in result.feedback

    def test_missing_field(self, guard, snapshot):
        data = {f: [] for f in REQUIRED_LIST_FIELDS if f != "mentioned_files"}
        data["mentioned_files_typo"] = []  # wrong key
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "mentioned_files" in result.feedback

    def test_field_not_list(self, guard, snapshot):
        data = {f: [] for f in REQUIRED_LIST_FIELDS}
        data["mentioned_files"] = "not a list"
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "must be a list" in result.feedback

    def test_all_fields_empty(self, guard, snapshot):
        data = {f: [] for f in REQUIRED_LIST_FIELDS}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "at least one signal" in result.feedback

    def test_multiple_missing_fields(self, guard, snapshot):
        data = {"mentioned_files": ["foo.py"]}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        # Should mention multiple missing fields
        assert "stack_traces" in result.feedback
        assert "apis_involved" in result.feedback


class TestReconGuardEdgeCases:
    def test_extra_fields_allowed(self, guard, snapshot):
        data = {**VALID_RECON, "extra": "ignored"}
        artifact = _make_artifact(json.dumps(data), snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_empty_string(self, guard, snapshot):
        artifact = _make_artifact("", snapshot)
        result = guard.validate(artifact)
        assert result.passed is False

    def test_guard_name(self, guard, snapshot):
        artifact = _make_artifact(json.dumps(VALID_RECON), snapshot)
        result = guard.validate(artifact)
        assert result.guard_name == "ReconGuard"
