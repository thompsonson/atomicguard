"""Tests for domain models."""

import pytest

from atomicguard.domain.models import (
    AmendmentType,
    Artifact,
    ArtifactSource,
    ArtifactStatus,
    ContextSnapshot,
    FailureType,
    FeedbackEntry,
    GuardResult,
    HumanAmendment,
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
            workflow_id="test-workflow-001",
            specification="Test spec",
            constraints="Test constraints",
            feedback_history=feedback,
            dependency_artifacts=(("dep-key", "dep-001"),),
        )
        assert len(snapshot.feedback_history) == 2
        assert len(snapshot.dependency_artifacts) == 1


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
            workflow_id="test-workflow-001",
            content="def add(a, b): return a + b",
            previous_attempt_id="original-001",
            parent_action_pair_id=None,
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
        assert WorkflowStatus.CHECKPOINT.value == "checkpoint"

    def test_all_statuses_accounted(self):
        """Ensure we have exactly 4 statuses."""
        assert len(WorkflowStatus) == 4


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

    def test_checkpoint_result(self, sample_checkpoint):
        """Test checkpoint workflow result."""
        result = WorkflowResult(
            status=WorkflowStatus.CHECKPOINT,
            artifacts={},
            failed_step="g_impl",
            checkpoint=sample_checkpoint,
        )
        assert result.status == WorkflowStatus.CHECKPOINT
        assert result.checkpoint == sample_checkpoint
        assert result.checkpoint.checkpoint_id == "chk-test-001"


# =============================================================================
# CHECKPOINT AND AMENDMENT MODEL TESTS
# =============================================================================


class TestArtifactSource:
    """Tests for ArtifactSource enum."""

    def test_source_values(self):
        """Verify all expected source values exist."""
        assert ArtifactSource.GENERATED.value == "generated"
        assert ArtifactSource.HUMAN.value == "human"
        assert ArtifactSource.IMPORTED.value == "imported"

    def test_all_sources_accounted(self):
        """Ensure we have exactly 3 sources."""
        assert len(ArtifactSource) == 3

    def test_artifact_default_source(self, sample_context_snapshot):
        """Verify artifact defaults to GENERATED source."""
        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test content",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )
        assert artifact.source == ArtifactSource.GENERATED

    def test_artifact_with_human_source(self, sample_context_snapshot):
        """Verify artifact can be created with HUMAN source."""
        artifact = Artifact(
            artifact_id="human-001",
            workflow_id="wf-001",
            content="human provided content",
            previous_attempt_id="failed-001",
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=3,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
            source=ArtifactSource.HUMAN,
        )
        assert artifact.source == ArtifactSource.HUMAN


class TestFailureType:
    """Tests for FailureType enum."""

    def test_failure_type_values(self):
        """Verify all expected failure type values exist."""
        assert FailureType.ESCALATION.value == "escalation"
        assert FailureType.RMAX_EXHAUSTED.value == "rmax_exhausted"

    def test_all_failure_types_accounted(self):
        """Ensure we have exactly 2 failure types."""
        assert len(FailureType) == 2


class TestAmendmentType:
    """Tests for AmendmentType enum."""

    def test_amendment_type_values(self):
        """Verify all expected amendment type values exist."""
        assert AmendmentType.ARTIFACT.value == "artifact"
        assert AmendmentType.FEEDBACK.value == "feedback"
        assert AmendmentType.SKIP.value == "skip"

    def test_all_amendment_types_accounted(self):
        """Ensure we have exactly 3 amendment types."""
        assert len(AmendmentType) == 3


class TestWorkflowCheckpoint:
    """Tests for WorkflowCheckpoint dataclass."""

    def test_checkpoint_creation(self, sample_checkpoint):
        """Test creating a WorkflowCheckpoint with all fields."""
        assert sample_checkpoint.checkpoint_id == "chk-test-001"
        assert sample_checkpoint.workflow_id == "wf-test-001"
        assert sample_checkpoint.created_at == "2025-01-05T10:00:00Z"
        assert sample_checkpoint.specification == "Test specification content"
        assert sample_checkpoint.constraints == "Must be valid Python 3.12+"
        assert sample_checkpoint.rmax == 3
        assert sample_checkpoint.completed_steps == ("g_test",)
        assert sample_checkpoint.artifact_ids == (("g_test", "art-001"),)
        assert sample_checkpoint.failure_type == FailureType.RMAX_EXHAUSTED
        assert sample_checkpoint.failed_step == "g_impl"
        assert sample_checkpoint.failed_artifact_id == "art-002"
        assert sample_checkpoint.failure_feedback == "Syntax error on line 5"
        assert sample_checkpoint.provenance_ids == ("art-002", "art-003")

    def test_checkpoint_immutability(self, sample_checkpoint):
        """Verify frozen dataclass prevents mutation."""
        with pytest.raises(AttributeError):
            sample_checkpoint.checkpoint_id = "changed"

    def test_checkpoint_with_escalation_type(self, sample_checkpoint_escalation):
        """Test checkpoint with ESCALATION failure type."""
        assert sample_checkpoint_escalation.failure_type == FailureType.ESCALATION
        assert "Security vulnerability" in sample_checkpoint_escalation.failure_feedback

    def test_checkpoint_with_empty_completed_steps(
        self, sample_checkpoint_empty_completed
    ):
        """Test checkpoint at first step failure."""
        assert sample_checkpoint_empty_completed.completed_steps == ()
        assert sample_checkpoint_empty_completed.artifact_ids == ()
        assert sample_checkpoint_empty_completed.failed_step == "g_first"

    def test_checkpoint_with_multiple_completed_steps(
        self, sample_checkpoint_escalation
    ):
        """Test checkpoint after partial success."""
        assert len(sample_checkpoint_escalation.completed_steps) == 2
        assert sample_checkpoint_escalation.completed_steps == ("g_config", "g_test")
        assert len(sample_checkpoint_escalation.artifact_ids) == 2


class TestHumanAmendment:
    """Tests for HumanAmendment dataclass."""

    def test_amendment_creation_artifact_type(self, sample_amendment):
        """Test creating amendment with ARTIFACT type."""
        assert sample_amendment.amendment_id == "amd-test-001"
        assert sample_amendment.checkpoint_id == "chk-test-001"
        assert sample_amendment.amendment_type == AmendmentType.ARTIFACT
        assert sample_amendment.created_at == "2025-01-05T10:05:00Z"
        assert sample_amendment.created_by == "test-user"
        assert "def fixed_function" in sample_amendment.content
        assert sample_amendment.context == "Fixed the syntax error"
        assert sample_amendment.parent_artifact_id == "art-002"
        assert sample_amendment.additional_rmax == 0

    def test_amendment_creation_feedback_type(self, sample_amendment_feedback):
        """Test creating amendment with FEEDBACK type."""
        assert sample_amendment_feedback.amendment_type == AmendmentType.FEEDBACK
        assert "standard library" in sample_amendment_feedback.content
        assert sample_amendment_feedback.additional_rmax == 2

    def test_amendment_creation_skip_type(self, sample_amendment_skip):
        """Test creating amendment with SKIP type."""
        assert sample_amendment_skip.amendment_type == AmendmentType.SKIP
        assert sample_amendment_skip.content == ""
        assert "optional for MVP" in sample_amendment_skip.context
        assert sample_amendment_skip.parent_artifact_id is None

    def test_amendment_immutability(self, sample_amendment):
        """Verify frozen dataclass prevents mutation."""
        with pytest.raises(AttributeError):
            sample_amendment.content = "changed"

    def test_amendment_with_additional_rmax(self, sample_amendment_feedback):
        """Verify additional_rmax field works correctly."""
        assert sample_amendment_feedback.additional_rmax == 2

    def test_amendment_default_values(self):
        """Test amendment with minimal fields uses defaults."""
        amendment = HumanAmendment(
            amendment_id="amd-minimal",
            checkpoint_id="chk-001",
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T00:00:00Z",
            created_by="cli",
            content="minimal content",
        )
        assert amendment.context == ""
        assert amendment.parent_artifact_id is None
        assert amendment.additional_rmax == 0
