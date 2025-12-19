"""Tests for domain models."""

import pytest

from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    ContextSnapshot,
    FeedbackEntry,
    GuardResult,
    WorkflowResult,
    WorkflowState,
    WorkflowStatus,
)


class TestArtifactStatus:
    """Tests for ArtifactStatus enum."""

    def test_status_values(self):
        """Verify all expected status values exist."""
        assert ArtifactStatus.PENDING.value == "pending"
        assert ArtifactStatus.REJECTED.value == "rejected"
        assert ArtifactStatus.ACCEPTED.value == "accepted"
        assert ArtifactStatus.SUPERSEDED.value == "superseded"

    def test_all_statuses_accounted(self):
        """Ensure we have exactly 4 statuses."""
        assert len(ArtifactStatus) == 4


class TestFeedbackEntry:
    """Tests for FeedbackEntry dataclass."""

    def test_feedback_entry_creation(self):
        """Test creating a FeedbackEntry."""
        entry = FeedbackEntry(
            artifact_id="art-001",
            feedback="Syntax error on line 5",
        )
        assert entry.artifact_id == "art-001"
        assert entry.feedback == "Syntax error on line 5"

    def test_feedback_entry_immutable(self):
        """FeedbackEntry should be immutable (frozen)."""
        entry = FeedbackEntry(artifact_id="art-001", feedback="test")
        with pytest.raises(AttributeError):
            entry.artifact_id = "changed"


class TestContextSnapshot:
    """Tests for ContextSnapshot dataclass."""

    def test_context_snapshot_creation(self, sample_context_snapshot):
        """Test ContextSnapshot creation and access."""
        assert "adds two numbers" in sample_context_snapshot.specification
        assert sample_context_snapshot.feedback_history == ()

    def test_context_snapshot_immutable(self, sample_context_snapshot):
        """ContextSnapshot should be immutable."""
        with pytest.raises(AttributeError):
            sample_context_snapshot.specification = "changed"

    def test_context_snapshot_with_feedback(self):
        """Test ContextSnapshot with feedback history."""
        feedback = (
            FeedbackEntry("art-001", "First error"),
            FeedbackEntry("art-002", "Second error"),
        )
        snapshot = ContextSnapshot(
            specification="Test spec",
            constraints="Test constraints",
            feedback_history=feedback,
            dependency_ids=("dep-001",),
        )
        assert len(snapshot.feedback_history) == 2
        assert len(snapshot.dependency_ids) == 1


class TestArtifact:
    """Tests for Artifact dataclass."""

    def test_artifact_creation(self, sample_artifact):
        """Test Artifact creation."""
        assert sample_artifact.artifact_id == "test-artifact-001"
        assert "def add" in sample_artifact.content
        assert sample_artifact.status == ArtifactStatus.PENDING

    def test_artifact_immutable(self, sample_artifact):
        """Artifact should be immutable."""
        with pytest.raises(AttributeError):
            sample_artifact.content = "changed"

    def test_artifact_with_previous_attempt(self, sample_context_snapshot):
        """Test artifact that references a previous attempt."""
        artifact = Artifact(
            artifact_id="retry-001",
            content="def add(a, b): return a + b",
            previous_attempt_id="original-001",
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:01Z",
            attempt_number=2,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )
        assert artifact.previous_attempt_id == "original-001"
        assert artifact.attempt_number == 2


class TestGuardResult:
    """Tests for GuardResult dataclass."""

    def test_passed_result(self):
        """Test creating a passing result."""
        result = GuardResult(passed=True, feedback="All checks passed")
        assert result.passed is True
        assert result.feedback == "All checks passed"

    def test_failed_result(self):
        """Test creating a failing result."""
        result = GuardResult(passed=False, feedback="Syntax error")
        assert result.passed is False
        assert result.feedback == "Syntax error"

    def test_default_feedback(self):
        """Test default empty feedback."""
        result = GuardResult(passed=True)
        assert result.feedback == ""

    def test_guard_result_immutable(self):
        """GuardResult should be immutable."""
        result = GuardResult(passed=True)
        with pytest.raises(AttributeError):
            result.passed = False

    def test_fatal_default_false(self):
        """Fatal defaults to False for backward compatibility."""
        result = GuardResult(passed=False, feedback="Error")
        assert result.fatal is False

    def test_fatal_explicit_true(self):
        """Fatal can be set explicitly."""
        result = GuardResult(passed=False, feedback="Fatal error", fatal=True)
        assert result.fatal is True


class TestWorkflowStatus:
    """Tests for WorkflowStatus enum."""

    def test_status_values(self):
        """Verify all expected status values exist."""
        assert WorkflowStatus.SUCCESS.value == "success"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.ESCALATION.value == "escalation"

    def test_all_statuses_accounted(self):
        """Ensure we have exactly 3 statuses."""
        assert len(WorkflowStatus) == 3


class TestWorkflowState:
    """Tests for WorkflowState."""

    def test_initial_state(self):
        """Test initial workflow state."""
        state = WorkflowState()
        assert state.is_satisfied("g_test") is False
        assert state.get_artifact_id("g_test") is None

    def test_satisfy_guard(self):
        """Test satisfying a guard."""
        state = WorkflowState()
        state.satisfy("g_test", "artifact-001")
        assert state.is_satisfied("g_test") is True
        assert state.get_artifact_id("g_test") == "artifact-001"

    def test_multiple_guards(self):
        """Test satisfying multiple guards."""
        state = WorkflowState()
        state.satisfy("g_syntax", "art-001")
        state.satisfy("g_test", "art-002")

        assert state.is_satisfied("g_syntax") is True
        assert state.is_satisfied("g_test") is True
        assert state.is_satisfied("g_other") is False

    def test_workflow_state_mutable(self):
        """WorkflowState should be mutable (unlike other models)."""
        state = WorkflowState()
        state.satisfy("g_test", "art-001")
        # Should not raise - state is mutable
        state.guards["g_test"] = False
        assert state.is_satisfied("g_test") is False


class TestWorkflowResult:
    """Tests for WorkflowResult."""

    def test_successful_result(self, sample_artifact):
        """Test successful workflow result."""
        result = WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            artifacts={"g_impl": sample_artifact},
        )
        assert result.status == WorkflowStatus.SUCCESS
        assert "g_impl" in result.artifacts
        assert result.failed_step is None

    def test_failed_result(self):
        """Test failed workflow result."""
        result = WorkflowResult(
            status=WorkflowStatus.FAILED,
            artifacts={},
            failed_step="g_impl",
        )
        assert result.status == WorkflowStatus.FAILED
        assert result.failed_step == "g_impl"

    def test_result_with_provenance(self, sample_artifact):
        """Test result with provenance chain."""
        result = WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            artifacts={"g_impl": sample_artifact},
            provenance=((sample_artifact, "step_1"),),
        )
        assert len(result.provenance) == 1
        assert result.provenance[0][1] == "step_1"

    def test_escalation_result(self, sample_artifact):
        """Test escalation workflow result."""
        result = WorkflowResult(
            status=WorkflowStatus.ESCALATION,
            artifacts={},
            failed_step="g_impl",
            escalation_artifact=sample_artifact,
            escalation_feedback="Non-recoverable error",
        )
        assert result.status == WorkflowStatus.ESCALATION
        assert result.escalation_artifact == sample_artifact
        assert result.escalation_feedback == "Non-recoverable error"
