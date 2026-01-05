"""Tests for ResumableWorkflow - checkpoint and resume support."""

import pytest

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.workflow import ResumableWorkflow
from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import (
    AmendmentType,
    Artifact,
    ArtifactSource,
    FailureType,
    GuardResult,
    HumanAmendment,
    WorkflowStatus,
)
from atomicguard.guards import SyntaxGuard
from atomicguard.infrastructure.llm.mock import MockGenerator
from atomicguard.infrastructure.persistence.checkpoint import InMemoryCheckpointDAG
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


class AlwaysPassGuard(GuardInterface):
    """Guard that always passes."""

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=True, feedback="")


class AlwaysFailGuard(GuardInterface):
    """Guard that always fails."""

    def __init__(self, feedback: str = "Always fails"):
        self._feedback = feedback

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=False, feedback=self._feedback)


class FatalGuard(GuardInterface):
    """Guard that returns fatal failure."""

    def __init__(self, feedback: str = "Fatal: cannot recover") -> None:
        self._feedback = feedback

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=False, feedback=self._feedback, fatal=True)


class FailUntilHumanGuard(GuardInterface):
    """Guard that fails LLM artifacts but passes human artifacts."""

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        if artifact.source == ArtifactSource.HUMAN:
            return GuardResult(passed=True, feedback="Human artifact accepted")
        return GuardResult(passed=False, feedback="LLM artifact rejected")


class CountingGuard(GuardInterface):
    """Guard that fails until N calls, then passes."""

    def __init__(self, fail_count: int = 2):
        self._fail_count = fail_count
        self._calls = 0

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        self._calls += 1
        if self._calls <= self._fail_count:
            return GuardResult(
                passed=False, feedback=f"Fail {self._calls}/{self._fail_count}"
            )
        return GuardResult(passed=True, feedback="Passed")


# =============================================================================
# ResumableWorkflow Initialization Tests
# =============================================================================


class TestResumableWorkflowInit:
    """Tests for ResumableWorkflow initialization."""

    def test_init_creates_default_checkpoint_dag(self) -> None:
        """ResumableWorkflow creates InMemoryCheckpointDAG if none provided."""
        workflow = ResumableWorkflow()

        assert workflow._checkpoint_dag is not None
        assert isinstance(workflow._checkpoint_dag, InMemoryCheckpointDAG)

    def test_init_uses_provided_checkpoint_dag(self) -> None:
        """ResumableWorkflow uses provided checkpoint DAG."""
        checkpoint_dag = InMemoryCheckpointDAG()
        workflow = ResumableWorkflow(checkpoint_dag=checkpoint_dag)

        assert workflow._checkpoint_dag is checkpoint_dag

    def test_init_default_auto_checkpoint_true(self) -> None:
        """auto_checkpoint defaults to True."""
        workflow = ResumableWorkflow()

        assert workflow._auto_checkpoint is True

    def test_init_auto_checkpoint_false(self) -> None:
        """auto_checkpoint can be set to False."""
        workflow = ResumableWorkflow(auto_checkpoint=False)

        assert workflow._auto_checkpoint is False


# =============================================================================
# Checkpoint Creation Tests
# =============================================================================


class TestResumableWorkflowCheckpointCreation:
    """Tests for checkpoint creation on failure."""

    def test_checkpoint_created_on_rmax_exhausted(self) -> None:
        """Checkpoint created when retry budget exhausted."""
        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=AlwaysFailGuard())
        workflow = ResumableWorkflow(rmax=2)
        workflow.add_step("g_test", pair)

        result = workflow.execute("Write something")

        assert result.status == WorkflowStatus.CHECKPOINT
        assert result.checkpoint is not None
        assert result.checkpoint.failure_type == FailureType.RMAX_EXHAUSTED
        assert result.checkpoint.failed_step == "g_test"

    def test_checkpoint_created_on_escalation(self) -> None:
        """Checkpoint created on fatal guard failure."""
        generator = MockGenerator(responses=["code"])
        pair = ActionPair(generator=generator, guard=FatalGuard())
        workflow = ResumableWorkflow()
        workflow.add_step("g_test", pair)

        result = workflow.execute("Write something")

        assert result.status == WorkflowStatus.CHECKPOINT
        assert result.checkpoint is not None
        assert result.checkpoint.failure_type == FailureType.ESCALATION
        assert result.checkpoint.failed_step == "g_test"

    def test_no_checkpoint_when_auto_checkpoint_false(self) -> None:
        """No checkpoint created when auto_checkpoint=False."""
        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=AlwaysFailGuard())
        workflow = ResumableWorkflow(auto_checkpoint=False, rmax=2)
        workflow.add_step("g_test", pair)

        result = workflow.execute("Write something")

        assert result.status == WorkflowStatus.FAILED
        assert result.checkpoint is None

    def test_checkpoint_captures_completed_steps(self) -> None:
        """Checkpoint captures partial success."""
        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["bad"] * 5)
        pair1 = ActionPair(generator=gen1, guard=SyntaxGuard())
        pair2 = ActionPair(generator=gen2, guard=AlwaysFailGuard())

        workflow = ResumableWorkflow(rmax=2)
        workflow.add_step("g_config", pair1)
        workflow.add_step("g_impl", pair2, requires=("g_config",))

        result = workflow.execute("Write something")

        assert result.status == WorkflowStatus.CHECKPOINT
        assert result.checkpoint is not None
        assert "g_config" in result.checkpoint.completed_steps
        assert result.checkpoint.failed_step == "g_impl"

    def test_checkpoint_captures_artifact_ids(self) -> None:
        """Checkpoint captures all artifact references."""
        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["bad"] * 5)
        pair1 = ActionPair(generator=gen1, guard=SyntaxGuard())
        pair2 = ActionPair(generator=gen2, guard=AlwaysFailGuard())

        workflow = ResumableWorkflow(rmax=2)
        workflow.add_step("g_config", pair1)
        workflow.add_step("g_impl", pair2, requires=("g_config",))

        result = workflow.execute("Write something")

        assert result.checkpoint is not None
        artifact_dict = dict(result.checkpoint.artifact_ids)
        assert "g_config" in artifact_dict

    def test_checkpoint_captures_specification(self) -> None:
        """Checkpoint preserves original specification."""
        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=AlwaysFailGuard())
        workflow = ResumableWorkflow(rmax=1)
        workflow.add_step("g_test", pair)

        result = workflow.execute("My specific task")

        assert result.checkpoint is not None
        assert result.checkpoint.specification == "My specific task"

    def test_checkpoint_captures_constraints(self) -> None:
        """Checkpoint preserves original constraints."""
        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=AlwaysFailGuard())
        workflow = ResumableWorkflow(rmax=1, constraints="No imports")
        workflow.add_step("g_test", pair)

        result = workflow.execute("Write code")

        assert result.checkpoint is not None
        assert result.checkpoint.constraints == "No imports"

    def test_checkpoint_persisted_to_dag(self) -> None:
        """Checkpoint is persisted to checkpoint DAG."""
        checkpoint_dag = InMemoryCheckpointDAG()
        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=AlwaysFailGuard())
        workflow = ResumableWorkflow(checkpoint_dag=checkpoint_dag, rmax=1)
        workflow.add_step("g_test", pair)

        result = workflow.execute("Write code")

        assert result.checkpoint is not None
        # Verify checkpoint was stored
        stored = checkpoint_dag.get_checkpoint(result.checkpoint.checkpoint_id)
        assert stored.checkpoint_id == result.checkpoint.checkpoint_id


# =============================================================================
# Resume with Artifact Tests
# =============================================================================


class TestResumableWorkflowResumeWithArtifact:
    """Tests for resuming workflow with human-provided artifact."""

    def test_resume_with_passing_artifact(self) -> None:
        """Human artifact that passes guard allows workflow to continue."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        # First, create a workflow that fails
        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=FailUntilHumanGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_test", pair)

        initial_result = workflow.execute("Write something")
        assert initial_result.status == WorkflowStatus.CHECKPOINT
        checkpoint_id = initial_result.checkpoint.checkpoint_id

        # Now resume with human artifact
        amendment = HumanAmendment(
            amendment_id="amd-001",
            checkpoint_id=checkpoint_id,
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="def good_function(): pass",
            parent_artifact_id=initial_result.checkpoint.failed_artifact_id,
        )

        # Need to recreate workflow with same steps
        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step("g_test", pair)

        resume_result = workflow2.resume(checkpoint_id, amendment)

        assert resume_result.status == WorkflowStatus.SUCCESS
        assert "g_test" in resume_result.artifacts

    def test_resume_with_failing_artifact_creates_new_checkpoint(self) -> None:
        """Human artifact that fails guard creates new checkpoint."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        # Create failing workflow
        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=AlwaysFailGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_test", pair)

        initial_result = workflow.execute("Write something")
        checkpoint_id = initial_result.checkpoint.checkpoint_id

        # Resume with bad human artifact
        amendment = HumanAmendment(
            amendment_id="amd-002",
            checkpoint_id=checkpoint_id,
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="still bad",
        )

        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step("g_test", pair)

        resume_result = workflow2.resume(checkpoint_id, amendment)

        assert resume_result.status == WorkflowStatus.CHECKPOINT
        assert resume_result.checkpoint is not None
        assert resume_result.checkpoint.checkpoint_id != checkpoint_id

    def test_resume_restores_completed_steps(self) -> None:
        """Resume restores completed steps from checkpoint."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        # Create workflow that succeeds on step 1, fails on step 2
        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["bad"] * 5)
        pair1 = ActionPair(generator=gen1, guard=SyntaxGuard())
        pair2 = ActionPair(generator=gen2, guard=FailUntilHumanGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_config", pair1)
        workflow.add_step("g_impl", pair2, requires=("g_config",))

        initial_result = workflow.execute("Write something")
        assert initial_result.status == WorkflowStatus.CHECKPOINT
        checkpoint_id = initial_result.checkpoint.checkpoint_id

        # Resume with passing human artifact
        amendment = HumanAmendment(
            amendment_id="amd-003",
            checkpoint_id=checkpoint_id,
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="fixed code",
        )

        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step("g_config", pair1)
        workflow2.add_step("g_impl", pair2, requires=("g_config",))

        resume_result = workflow2.resume(checkpoint_id, amendment)

        # Should have both artifacts
        assert "g_config" in resume_result.artifacts
        assert "g_impl" in resume_result.artifacts

    def test_human_artifact_has_human_source(self) -> None:
        """Human-provided artifact has source=HUMAN."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=FailUntilHumanGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_test", pair)

        initial_result = workflow.execute("Write something")
        checkpoint_id = initial_result.checkpoint.checkpoint_id

        amendment = HumanAmendment(
            amendment_id="amd-004",
            checkpoint_id=checkpoint_id,
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="human code",
        )

        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step("g_test", pair)

        resume_result = workflow2.resume(checkpoint_id, amendment)

        assert resume_result.status == WorkflowStatus.SUCCESS
        artifact = resume_result.artifacts["g_test"]
        assert artifact.source == ArtifactSource.HUMAN

    def test_human_artifact_links_to_failed(self) -> None:
        """Human artifact has previous_attempt_id linking to failed artifact."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=FailUntilHumanGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_test", pair)

        initial_result = workflow.execute("Write something")
        checkpoint = initial_result.checkpoint

        amendment = HumanAmendment(
            amendment_id="amd-005",
            checkpoint_id=checkpoint.checkpoint_id,
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="human code",
            parent_artifact_id=checkpoint.failed_artifact_id,
        )

        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step("g_test", pair)

        resume_result = workflow2.resume(checkpoint.checkpoint_id, amendment)

        artifact = resume_result.artifacts["g_test"]
        assert artifact.previous_attempt_id == checkpoint.failed_artifact_id

    def test_amendment_stored_in_checkpoint_dag(self) -> None:
        """Amendment is persisted to checkpoint DAG."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=FailUntilHumanGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_test", pair)

        initial_result = workflow.execute("Write something")
        checkpoint_id = initial_result.checkpoint.checkpoint_id

        amendment = HumanAmendment(
            amendment_id="amd-006",
            checkpoint_id=checkpoint_id,
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="human code",
        )

        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step("g_test", pair)

        workflow2.resume(checkpoint_id, amendment)

        # Verify amendment was stored
        stored = checkpoint_dag.get_amendment("amd-006")
        assert stored.content == "human code"


# =============================================================================
# Resume with Feedback Tests
# =============================================================================


class TestResumableWorkflowResumeWithFeedback:
    """Tests for resuming workflow with human feedback for retry."""

    def test_resume_with_feedback_triggers_retry(self) -> None:
        """Feedback amendment triggers LLM retry with guidance."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        # Use AlwaysFailGuard for first workflow (to guarantee failure)
        # Then use a guard that passes for the resumed workflow
        generator = MockGenerator(responses=["bad", "good code"])
        fail_pair = ActionPair(generator=generator, guard=AlwaysFailGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=0,  # No retries - fail immediately
        )
        workflow.add_step("g_test", fail_pair)

        # First execution fails
        initial_result = workflow.execute("Write something")
        assert initial_result.status == WorkflowStatus.CHECKPOINT
        checkpoint_id = initial_result.checkpoint.checkpoint_id

        # Resume with feedback
        amendment = HumanAmendment(
            amendment_id="amd-fb-001",
            checkpoint_id=checkpoint_id,
            amendment_type=AmendmentType.FEEDBACK,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="Use standard library only",
            additional_rmax=1,
        )

        # Create second generator and guard that passes
        gen2 = MockGenerator(responses=["good code"])
        pass_pair = ActionPair(generator=gen2, guard=AlwaysPassGuard())

        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step("g_test", pass_pair)

        resume_result = workflow2.resume(checkpoint_id, amendment)

        assert resume_result.status == WorkflowStatus.SUCCESS

    def test_resume_uses_additional_rmax(self) -> None:
        """Resume applies additional_rmax from amendment."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        # Guard that passes after 3 calls total
        counting_guard = CountingGuard(fail_count=3)
        generator = MockGenerator(responses=["a", "b", "c", "d", "e"])
        pair = ActionPair(generator=generator, guard=counting_guard)

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_test", pair)

        # First execution fails (1 attempt)
        initial_result = workflow.execute("Write something")
        assert initial_result.status == WorkflowStatus.CHECKPOINT

        # Resume with feedback and extra retries
        amendment = HumanAmendment(
            amendment_id="amd-fb-002",
            checkpoint_id=initial_result.checkpoint.checkpoint_id,
            amendment_type=AmendmentType.FEEDBACK,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="Try harder",
            additional_rmax=3,  # 1 (base) + 3 = 4 retries total
        )

        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step(
            "g_test", ActionPair(generator=generator, guard=counting_guard)
        )

        resume_result = workflow2.resume(
            initial_result.checkpoint.checkpoint_id, amendment
        )

        # Should succeed - guard passes on 4th call, we have enough retries
        assert resume_result.status == WorkflowStatus.SUCCESS


# =============================================================================
# Multi-Step Resume Tests
# =============================================================================


class TestResumableWorkflowMultiStepResume:
    """Tests for resuming workflows with multiple steps."""

    def test_resume_continues_to_end(self) -> None:
        """Resume executes remaining steps after failed step passes."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["bad"] * 5)
        gen3 = MockGenerator(responses=["z = 3"])

        pair1 = ActionPair(generator=gen1, guard=SyntaxGuard())
        pair2 = ActionPair(generator=gen2, guard=FailUntilHumanGuard())
        pair3 = ActionPair(generator=gen3, guard=SyntaxGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_config", pair1)
        workflow.add_step("g_impl", pair2, requires=("g_config",))
        workflow.add_step("g_deploy", pair3, requires=("g_impl",))

        # Execute - fails at g_impl
        initial_result = workflow.execute("Write something")
        assert initial_result.status == WorkflowStatus.CHECKPOINT
        assert initial_result.checkpoint.failed_step == "g_impl"

        # Resume with fix
        amendment = HumanAmendment(
            amendment_id="amd-multi-001",
            checkpoint_id=initial_result.checkpoint.checkpoint_id,
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="y = 2",
        )

        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step("g_config", pair1)
        workflow2.add_step("g_impl", pair2, requires=("g_config",))
        workflow2.add_step("g_deploy", pair3, requires=("g_impl",))

        resume_result = workflow2.resume(
            initial_result.checkpoint.checkpoint_id, amendment
        )

        # All three steps should be complete
        assert resume_result.status == WorkflowStatus.SUCCESS
        assert "g_config" in resume_result.artifacts
        assert "g_impl" in resume_result.artifacts
        assert "g_deploy" in resume_result.artifacts

    def test_dependencies_available_after_resume(self) -> None:
        """Prior artifacts are accessible after resume."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["bad"] * 5)

        pair1 = ActionPair(generator=gen1, guard=SyntaxGuard())
        pair2 = ActionPair(generator=gen2, guard=FailUntilHumanGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_config", pair1)
        workflow.add_step("g_impl", pair2, requires=("g_config",), deps=("g_config",))

        initial_result = workflow.execute("Write something")
        config_artifact_id = dict(initial_result.checkpoint.artifact_ids)["g_config"]

        amendment = HumanAmendment(
            amendment_id="amd-multi-002",
            checkpoint_id=initial_result.checkpoint.checkpoint_id,
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="y = 2",
        )

        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step("g_config", pair1)
        workflow2.add_step("g_impl", pair2, requires=("g_config",), deps=("g_config",))

        resume_result = workflow2.resume(
            initial_result.checkpoint.checkpoint_id, amendment
        )

        # Verify g_config artifact is same as before
        assert resume_result.artifacts["g_config"].artifact_id == config_artifact_id


# =============================================================================
# Provenance Tests
# =============================================================================


class TestResumableWorkflowProvenance:
    """Tests for provenance tracking through amendments."""

    def test_human_artifact_in_provenance_chain(self) -> None:
        """Human artifact is tracked in DAG with correct provenance."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=FailUntilHumanGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_test", pair)

        initial_result = workflow.execute("Write something")
        failed_id = initial_result.checkpoint.failed_artifact_id

        amendment = HumanAmendment(
            amendment_id="amd-prov-001",
            checkpoint_id=initial_result.checkpoint.checkpoint_id,
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="human code",
            parent_artifact_id=failed_id,
        )

        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow2.add_step("g_test", pair)

        resume_result = workflow2.resume(
            initial_result.checkpoint.checkpoint_id, amendment
        )

        # Get provenance chain
        final_artifact = resume_result.artifacts["g_test"]
        provenance = artifact_dag.get_provenance(final_artifact.artifact_id)

        # Should have at least 2 artifacts in chain
        assert len(provenance) >= 2
        # Last one should be human
        assert provenance[-1].source == ArtifactSource.HUMAN


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestResumableWorkflowErrors:
    """Tests for error handling in resume."""

    def test_resume_invalid_checkpoint_raises(self) -> None:
        """Resume with invalid checkpoint ID raises KeyError."""
        workflow = ResumableWorkflow()
        workflow.add_step(
            "g_test",
            ActionPair(
                generator=MockGenerator(responses=["x"]), guard=AlwaysPassGuard()
            ),
        )

        amendment = HumanAmendment(
            amendment_id="amd-err-001",
            checkpoint_id="nonexistent",
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="code",
        )

        with pytest.raises(KeyError, match="Checkpoint not found"):
            workflow.resume("nonexistent", amendment)

    def test_resume_missing_step_raises(self) -> None:
        """Resume with checkpoint referencing unknown step raises ValueError."""
        artifact_dag = InMemoryArtifactDAG()
        checkpoint_dag = InMemoryCheckpointDAG()

        generator = MockGenerator(responses=["bad"] * 5)
        pair = ActionPair(generator=generator, guard=AlwaysFailGuard())

        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        workflow.add_step("g_test", pair)

        initial_result = workflow.execute("Write something")
        checkpoint_id = initial_result.checkpoint.checkpoint_id

        amendment = HumanAmendment(
            amendment_id="amd-err-002",
            checkpoint_id=checkpoint_id,
            amendment_type=AmendmentType.ARTIFACT,
            created_at="2025-01-05T10:00:00Z",
            created_by="test-user",
            content="code",
        )

        # Create workflow WITHOUT the failed step
        workflow2 = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=1,
        )
        # Don't add g_test step

        with pytest.raises(ValueError, match="Step not found"):
            workflow2.resume(checkpoint_id, amendment)
