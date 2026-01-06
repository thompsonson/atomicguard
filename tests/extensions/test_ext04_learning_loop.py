"""
Extension 04: Learning Loop Tests.

TDD tests for implementing:
- Refinement predicate Phi_refinement (Def 21)
- Training trace extraction tau (Def 22)
- Sparse reward R_sparse (Def 23)
- Theorem 9: Training Trace Completeness
- Theorem 10: Learning preserves system dynamics
"""

import pytest

from atomicguard.domain.models import Artifact, ArtifactStatus


class TestRefinementPredicate:
    """Tests for Phi_refinement (Def 21)."""

    def test_matches_accepted_with_rejected_prior(self, retry_chain_artifacts):
        """Phi_refinement matches ACCEPTED artifact with REJECTED ancestor."""
        from atomicguard.domain.learning import RefinementPredicate

        predicate = RefinementPredicate()

        # retry_chain_artifacts[-1] is ACCEPTED with REJECTED ancestors
        final_artifact = retry_chain_artifacts[-1]
        assert final_artifact.status == ArtifactStatus.ACCEPTED
        assert final_artifact.previous_attempt_id is not None

        assert predicate.matches(final_artifact) is True

    def test_rejects_first_attempt_success(self, sample_artifact):
        """Phi_refinement rejects artifacts without rejection history."""
        from atomicguard.domain.learning import RefinementPredicate

        predicate = RefinementPredicate()

        # sample_artifact has no previous_attempt_id (first attempt)
        assert sample_artifact.previous_attempt_id is None

        # Not a refinement - no prior rejections
        assert predicate.matches(sample_artifact) is False

    def test_rejects_all_rejected_chain(self, memory_dag, sample_context_snapshot):
        """Phi_refinement rejects if no ACCEPTED in chain."""
        from atomicguard.domain.learning import RefinementPredicate

        # Create chain of all REJECTED artifacts
        art1 = Artifact(
            artifact_id="fail-001",
            workflow_id="wf-fail",
            content="bad code 1",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-fail",
            created_at="2025-01-01T10:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=None,
            feedback="Error 1",
            context=sample_context_snapshot,
        )
        art2 = Artifact(
            artifact_id="fail-002",
            workflow_id="wf-fail",
            content="bad code 2",
            previous_attempt_id="fail-001",
            parent_action_pair_id=None,
            action_pair_id="ap-fail",
            created_at="2025-01-01T10:05:00Z",
            attempt_number=2,
            status=ArtifactStatus.REJECTED,
            guard_result=None,
            feedback="Error 2",
            context=sample_context_snapshot,
        )
        memory_dag.store(art1)
        memory_dag.store(art2)

        predicate = RefinementPredicate(dag=memory_dag)

        # No ACCEPTED artifact in chain - not a successful refinement
        assert predicate.matches(art2) is False

    def test_uses_previous_attempt_id(self, retry_chain_artifacts, memory_dag):
        """Phi_refinement follows previous_attempt_id chain."""
        from atomicguard.domain.learning import RefinementPredicate

        predicate = RefinementPredicate(dag=memory_dag)

        # Final artifact
        final = retry_chain_artifacts[-1]  # ACCEPTED

        # Predicate should trace back through previous_attempt_id
        chain = predicate.get_retry_chain(final)

        assert len(chain) == 3
        assert chain[0].artifact_id == "retry-001"  # First attempt
        assert chain[-1].artifact_id == "retry-003"  # Final ACCEPTED


class TestTrainingTrace:
    """Tests for training trace extraction tau (Def 22)."""

    def test_training_trace_uses_extraction(self, populated_dag):
        """tau = E(R, Phi_training) uses extraction function."""
        from atomicguard.domain.extraction import StatusPredicate
        from atomicguard.domain.learning import extract_training_trace

        # Training trace is just extraction with specific predicate
        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)
        trace = extract_training_trace(populated_dag, predicate)

        assert isinstance(trace, list)
        assert all(isinstance(a, Artifact) for a in trace)

    def test_training_trace_configurable_filter(self, populated_dag):
        """Different Phi_training filters produce different traces."""
        from atomicguard.domain.extraction import StatusPredicate
        from atomicguard.domain.learning import extract_training_trace

        # Accepted only
        accepted_trace = extract_training_trace(
            populated_dag, StatusPredicate(ArtifactStatus.ACCEPTED)
        )

        # Rejected only
        rejected_trace = extract_training_trace(
            populated_dag, StatusPredicate(ArtifactStatus.REJECTED)
        )

        # Different filters -> different traces
        assert len(accepted_trace) != len(rejected_trace)
        assert all(a.status == ArtifactStatus.ACCEPTED for a in accepted_trace)
        assert all(a.status == ArtifactStatus.REJECTED for a in rejected_trace)

    def test_refinement_only_trace(self, memory_dag, retry_chain_artifacts):
        """Phi_refinement filter yields retry->success chains only."""
        from atomicguard.domain.learning import (
            RefinementPredicate,
            extract_training_trace,
        )

        # retry_chain_artifacts fixture populates memory_dag with the chain
        _ = retry_chain_artifacts  # Ensure fixture runs to populate dag
        predicate = RefinementPredicate(dag=memory_dag)
        trace = extract_training_trace(memory_dag, predicate)

        # Should only include the final ACCEPTED artifact (has prior rejections)
        assert len(trace) == 1
        assert trace[0].status == ArtifactStatus.ACCEPTED
        assert trace[0].previous_attempt_id is not None

    def test_all_accepted_trace(self, populated_dag):
        """Phi_status(ACCEPTED) yields all successes."""
        from atomicguard.domain.extraction import StatusPredicate
        from atomicguard.domain.learning import extract_training_trace

        trace = extract_training_trace(
            populated_dag, StatusPredicate(ArtifactStatus.ACCEPTED)
        )

        # All ACCEPTED artifacts (populated_dag has 5)
        assert len(trace) == 5
        assert all(a.status == ArtifactStatus.ACCEPTED for a in trace)


class TestSparseReward:
    """Tests for R_sparse (Def 23)."""

    def test_accepted_yields_positive_one(self, sample_context_snapshot):
        """R_sparse(accepted_artifact) == +1."""
        from atomicguard.domain.learning import sparse_reward

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="good code",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )

        assert sparse_reward(artifact) == 1

    def test_rejected_yields_negative_one(self, sample_context_snapshot):
        """R_sparse(rejected_artifact) == -1."""
        from atomicguard.domain.learning import sparse_reward

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="bad code",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=None,
            feedback="Error",
            context=sample_context_snapshot,
        )

        assert sparse_reward(artifact) == -1

    def test_pending_raises_error(self, sample_artifact):
        """R_sparse(pending_artifact) raises ValueError."""
        from atomicguard.domain.learning import sparse_reward

        # sample_artifact has PENDING status
        assert sample_artifact.status == ArtifactStatus.PENDING

        with pytest.raises(ValueError, match="PENDING"):
            sparse_reward(sample_artifact)

    def test_reward_from_status(self, sample_context_snapshot):
        """Reward derived from artifact.status, not guard_result."""
        from atomicguard.domain.learning import sparse_reward
        from atomicguard.domain.models import GuardResult

        # Artifact with guard_result but status determines reward
        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="code",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,  # Status is ACCEPTED
            guard_result=GuardResult(passed=True, feedback=""),
            feedback="",
            context=sample_context_snapshot,
        )

        # Reward based on status, not guard_result
        assert sparse_reward(artifact) == 1


class TestTrainingTraceCompleteness:
    """Tests for Theorem 9: Training Trace Completeness."""

    def test_trace_contains_prompt(self, retry_chain_artifacts):
        """Each item has r.Psi for prompt construction."""
        # Each artifact in trace should have context.specification
        for artifact in retry_chain_artifacts:
            assert artifact.context is not None
            assert artifact.context.specification is not None
            assert len(artifact.context.specification) > 0

    def test_trace_contains_completion(self, retry_chain_artifacts):
        """Each item has r.a for target completion."""
        # Each artifact has content (the generated code)
        for artifact in retry_chain_artifacts:
            assert artifact.content is not None
            assert len(artifact.content) > 0

    def test_trace_contains_history(self, retry_chain_artifacts):
        """Each item has r.H for reasoning context."""
        # Artifacts with previous_attempt_id have feedback history
        for artifact in retry_chain_artifacts:
            if artifact.previous_attempt_id is not None:
                # Feedback from prior attempt should be available
                assert artifact.context.feedback_history is not None

    def test_trace_contains_provenance(self, retry_chain_artifacts, memory_dag):
        """Provenance links available via metadata."""
        from atomicguard.domain.learning import RefinementPredicate

        predicate = RefinementPredicate(dag=memory_dag)

        # Final ACCEPTED artifact
        final = retry_chain_artifacts[-1]

        # Can trace provenance via previous_attempt_id
        chain = predicate.get_retry_chain(final)

        # Chain provides full provenance
        assert len(chain) > 1
        assert chain[0].artifact_id == "retry-001"
        assert chain[-1].artifact_id == "retry-003"


class TestLearningLoopPreservesDynamics:
    """Tests for Theorem 10: Learning preserves system dynamics."""

    def test_extraction_does_not_modify_workflow(self, populated_dag):
        """Training trace extraction is read-only."""
        from atomicguard.domain.extraction import StatusPredicate
        from atomicguard.domain.learning import extract_training_trace

        # Count before extraction
        count_before = len(list(populated_dag._artifacts.values()))

        # Extract training trace
        _ = extract_training_trace(
            populated_dag, StatusPredicate(ArtifactStatus.ACCEPTED)
        )

        # Count after extraction
        count_after = len(list(populated_dag._artifacts.values()))

        # No modification to DAG
        assert count_before == count_after

    def test_training_external_to_execution(self):
        """Training occurs after workflow execution, not during."""
        from atomicguard.domain.learning import TrainingLoop

        # TrainingLoop is separate from Workflow execution
        # It only reads from repository, doesn't participate in execution
        loop = TrainingLoop()

        # No methods that affect workflow execution
        assert not hasattr(loop, "execute_step")
        assert not hasattr(loop, "run_guard")

        # Only has extraction/training methods
        assert hasattr(loop, "extract_traces")
        assert hasattr(loop, "prepare_dataset")
