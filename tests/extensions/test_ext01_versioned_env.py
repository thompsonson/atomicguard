"""
Extension 01: Versioned Environment Tests.

TDD tests for implementing:
- W_ref content-addressed workflow hashing (Def 11)
- workflow_ref field on Artifact (Def 10)
- Structured metadata on Artifact (Def 10)
- Configuration amendment ⊕ operator (Def 12)
- Resume with workflow verification (Def 15)
- Theorem 4: Resume Preserves System Dynamics
- Theorem 5: Human-in-the-Loop Preserves System Dynamics
"""

import pytest


class TestWorkflowRef:
    """Tests for W_ref content-addressed workflow hashing (Def 11)."""

    def test_workflow_hash_deterministic(self, sample_workflow_definition):
        """Same workflow definition produces same hash."""
        # Import will fail until implemented
        from atomicguard.domain.workflow import compute_workflow_ref

        hash1 = compute_workflow_ref(sample_workflow_definition)
        hash2 = compute_workflow_ref(sample_workflow_definition)
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_workflow_hash_different_for_different_workflows(
        self, sample_workflow_definition
    ):
        """Different workflows produce different hashes."""
        from atomicguard.domain.workflow import compute_workflow_ref

        different_workflow = {
            "steps": [
                {"id": "g_other", "guard": "OtherGuard", "deps": []},
            ]
        }
        hash1 = compute_workflow_ref(sample_workflow_definition)
        hash2 = compute_workflow_ref(different_workflow)
        assert hash1 != hash2

    def test_workflow_hash_integrity_axiom(self, sample_workflow_definition):
        """hash(resolve(W_ref)) == W_ref (round-trip integrity)."""
        from atomicguard.domain.workflow import (
            compute_workflow_ref,
            resolve_workflow_ref,
        )

        w_ref = compute_workflow_ref(sample_workflow_definition)
        # Store workflow (resolve should retrieve it)
        resolved = resolve_workflow_ref(w_ref)
        re_hashed = compute_workflow_ref(resolved)
        assert re_hashed == w_ref

    def test_workflow_hash_ignores_whitespace(self):
        """Hash is stable across formatting differences."""
        from atomicguard.domain.workflow import compute_workflow_ref

        workflow1 = {"steps": [{"id": "g_test", "guard": "TestGuard", "deps": []}]}
        workflow2 = {
            "steps": [
                {
                    "id": "g_test",
                    "guard": "TestGuard",
                    "deps": [],
                }
            ]
        }
        # Same logical content should produce same hash
        assert compute_workflow_ref(workflow1) == compute_workflow_ref(workflow2)


class TestArtifactWorkflowRef:
    """Tests for workflow_ref field on Artifact (Def 10)."""

    def test_artifact_has_workflow_ref_field(self, sample_artifact):
        """Artifact includes workflow_ref: str | None."""
        # Test that the field exists (will fail until Artifact model updated)
        assert hasattr(sample_artifact, "workflow_ref")

    def test_artifact_workflow_ref_stored_on_generation(
        self, sample_context, mock_generator
    ):
        """workflow_ref captured when artifact is generated."""
        from atomicguard.domain.workflow import compute_workflow_ref

        workflow_def = {"steps": [{"id": "g_test", "guard": "TestGuard", "deps": []}]}
        expected_ref = compute_workflow_ref(workflow_def)

        # Generator should embed workflow_ref in artifact
        artifact = mock_generator.generate(
            context=sample_context,
            workflow_ref=expected_ref,
            action_pair_id="ap-001",
            workflow_id="wf-001",
        )
        assert artifact.workflow_ref == expected_ref

    def test_artifact_workflow_ref_immutable(self, sample_context_snapshot):
        """workflow_ref cannot be modified after creation."""
        from atomicguard.domain.models import Artifact, ArtifactStatus

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=sample_context_snapshot,
            workflow_ref="abc123",
        )
        # Artifact is frozen dataclass - assignment should raise
        with pytest.raises((AttributeError, TypeError)):
            artifact.workflow_ref = "different"


class TestArtifactMetadata:
    """Tests for structured metadata on Artifact (Def 10)."""

    def test_artifact_has_metadata_field(self, sample_artifact):
        """Artifact includes metadata: dict field."""
        assert hasattr(sample_artifact, "metadata")

    def test_metadata_accepts_arbitrary_keys(self, sample_context_snapshot):
        """metadata dict can store custom keys."""
        from atomicguard.domain.models import Artifact, ArtifactStatus

        metadata = {
            "generator_model": "gpt-4",
            "temperature": 0.7,
            "custom_tag": "experiment-1",
        }
        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=sample_context_snapshot,
            metadata=metadata,
        )
        assert artifact.metadata["generator_model"] == "gpt-4"
        assert artifact.metadata["temperature"] == 0.7
        assert artifact.metadata["custom_tag"] == "experiment-1"

    def test_metadata_immutable_after_creation(self, sample_context_snapshot):
        """metadata cannot be modified (frozen)."""
        from atomicguard.domain.models import Artifact, ArtifactStatus

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=sample_context_snapshot,
            metadata={"key": "value"},
        )
        # Should not be able to modify metadata
        with pytest.raises((TypeError, AttributeError)):
            artifact.metadata["new_key"] = "new_value"


class TestConfigurationAmendment:
    """Tests for ⊕ operator (Def 12)."""

    def test_context_amend_specification(self, sample_context):
        """Context.amend(delta_spec=...) creates new context."""
        new_context = sample_context.amend(
            delta_spec="Additional requirement: must handle negative numbers"
        )
        assert "negative numbers" in new_context.specification
        assert "negative numbers" not in sample_context.specification

    def test_context_amend_constraints(self, sample_context):
        """Context.amend(delta_constraints=...) creates new context."""
        new_context = sample_context.amend(delta_constraints="Max 100 lines of code")
        assert "100 lines" in new_context.ambient.constraints
        # Original unchanged
        assert "100 lines" not in sample_context.ambient.constraints

    def test_amendment_is_monotonic(self, sample_context):
        """Amendments only add, never remove information."""
        original_spec = sample_context.specification
        new_context = sample_context.amend(delta_spec="Extra requirement")
        # Original content preserved
        assert original_spec in new_context.specification
        # New content added
        assert "Extra requirement" in new_context.specification

    def test_amendment_returns_new_context(self, sample_context):
        """amend() returns new Context, original unchanged."""
        from atomicguard.domain.models import Context

        original_id = id(sample_context)
        new_context = sample_context.amend(delta_spec="New requirement")

        assert isinstance(new_context, Context)
        assert id(new_context) != original_id
        # Original specification unchanged
        assert "New requirement" not in sample_context.specification


class TestResumeWithWorkflowVerification:
    """Tests for W_ref verification on resume (Def 15)."""

    def test_resume_verifies_workflow_ref(
        self, sample_checkpoint, memory_checkpoint_dag
    ):
        """Resume checks W_ref integrity before continuing."""
        from atomicguard.domain.workflow import WorkflowResumer

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)

        # Resumer should verify W_ref matches before continuing
        resumer = WorkflowResumer(checkpoint_dag=memory_checkpoint_dag)
        # This should not raise if W_ref matches
        resumer.verify_workflow_integrity(sample_checkpoint.checkpoint_id)

    def test_resume_fails_on_workflow_mismatch(
        self, sample_checkpoint, memory_checkpoint_dag
    ):
        """Resume raises error if workflow changed."""
        from atomicguard.domain.workflow import (
            WorkflowIntegrityError,
            WorkflowResumer,
        )

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)

        resumer = WorkflowResumer(checkpoint_dag=memory_checkpoint_dag)

        # Simulate workflow change by providing different W_ref
        with pytest.raises(WorkflowIntegrityError):
            resumer.resume(
                checkpoint_id=sample_checkpoint.checkpoint_id,
                current_workflow_ref="different-hash-abc123",
            )

    def test_resume_succeeds_on_matching_workflow(
        self, sample_checkpoint, memory_checkpoint_dag
    ):
        """Resume proceeds when W_ref matches."""
        from atomicguard.domain.workflow import WorkflowResumer

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)

        resumer = WorkflowResumer(checkpoint_dag=memory_checkpoint_dag)

        # Resume with matching W_ref should succeed
        result = resumer.resume(
            checkpoint_id=sample_checkpoint.checkpoint_id,
            current_workflow_ref=sample_checkpoint.workflow_ref,
        )
        assert result is not None


class TestTheorem4ResumePreservesDynamics:
    """Tests for Theorem 4: Resume Preserves System Dynamics."""

    def test_resumed_workflow_equivalent_to_continuous(
        self, sample_checkpoint, memory_checkpoint_dag, memory_dag
    ):
        """Resuming produces same result as uninterrupted execution."""
        from atomicguard.domain.workflow import WorkflowResumer

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)

        resumer = WorkflowResumer(
            checkpoint_dag=memory_checkpoint_dag, artifact_dag=memory_dag
        )

        # Resumed state should be semantically equivalent to
        # having executed continuously from the beginning
        resumed_state = resumer.restore_state(sample_checkpoint.checkpoint_id)

        # Verify completed steps match checkpoint
        assert resumed_state.completed_steps == sample_checkpoint.completed_steps

    def test_resume_reconstructs_context_from_repository(
        self, sample_checkpoint, memory_checkpoint_dag, memory_dag
    ):
        """Context is reconstructed from stored items, not checkpoint."""
        from atomicguard.domain.workflow import WorkflowResumer

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)

        resumer = WorkflowResumer(
            checkpoint_dag=memory_checkpoint_dag, artifact_dag=memory_dag
        )

        # Context should be derived from repository artifacts
        context = resumer.reconstruct_context(sample_checkpoint.checkpoint_id)
        assert context is not None
        # Context should include feedback from provenance chain
        assert context.feedback_history is not None

    def test_resume_continues_refinement_loop(
        self, sample_checkpoint, memory_checkpoint_dag
    ):
        """Resumed workflow continues retry loop with correct attempt_number."""
        from atomicguard.domain.workflow import WorkflowResumer

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)

        resumer = WorkflowResumer(checkpoint_dag=memory_checkpoint_dag)

        # Get the next attempt number for failed step
        next_attempt = resumer.get_next_attempt_number(sample_checkpoint.checkpoint_id)

        # Should continue from where it left off
        assert next_attempt > 0

    def test_resume_preserves_feedback_history(
        self, sample_checkpoint, memory_checkpoint_dag, memory_dag
    ):
        """H_feedback is reconstructed from provenance chain."""
        from atomicguard.domain.workflow import WorkflowResumer

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)

        resumer = WorkflowResumer(
            checkpoint_dag=memory_checkpoint_dag, artifact_dag=memory_dag
        )

        # Feedback history should be reconstructed from artifact provenance
        feedback_history = resumer.reconstruct_feedback_history(
            sample_checkpoint.checkpoint_id
        )

        # Should include failure feedback from checkpoint
        assert sample_checkpoint.failure_feedback in str(feedback_history)


class TestTheorem5HumanInLoopPreservesDynamics:
    """Tests for Theorem 5: Human-in-the-Loop Preserves System Dynamics."""

    def test_human_artifact_flows_through_guard(
        self, sample_amendment, sample_checkpoint, memory_checkpoint_dag
    ):
        """Human-provided artifact is validated by same guard as generated."""
        from atomicguard.domain.workflow import HumanAmendmentProcessor

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)
        memory_checkpoint_dag.store_amendment(sample_amendment)

        processor = HumanAmendmentProcessor(checkpoint_dag=memory_checkpoint_dag)

        # Human artifact should be subject to guard validation
        result = processor.process_amendment(sample_amendment.amendment_id)

        # Result should include guard verdict
        assert hasattr(result, "guard_result")

    def test_human_artifact_stored_in_repository(
        self, sample_amendment, sample_checkpoint, memory_checkpoint_dag, memory_dag
    ):
        """Human amendment stored as normal repository item."""
        from atomicguard.domain.workflow import HumanAmendmentProcessor

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)
        memory_checkpoint_dag.store_amendment(sample_amendment)

        processor = HumanAmendmentProcessor(
            checkpoint_dag=memory_checkpoint_dag, artifact_dag=memory_dag
        )

        # Process amendment - should create artifact in DAG
        artifact = processor.create_artifact_from_amendment(
            sample_amendment.amendment_id
        )

        # Artifact should be stored in repository
        stored = memory_dag.get_artifact(artifact.artifact_id)
        assert stored is not None
        assert stored.content == sample_amendment.content

    def test_human_source_distinguished_from_generated(
        self, sample_amendment, sample_checkpoint, memory_checkpoint_dag, memory_dag
    ):
        """ArtifactSource.HUMAN distinguishes human contributions."""
        from atomicguard.domain.models import ArtifactSource
        from atomicguard.domain.workflow import HumanAmendmentProcessor

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)
        memory_checkpoint_dag.store_amendment(sample_amendment)

        processor = HumanAmendmentProcessor(
            checkpoint_dag=memory_checkpoint_dag, artifact_dag=memory_dag
        )

        artifact = processor.create_artifact_from_amendment(
            sample_amendment.amendment_id
        )

        # Source should be marked as HUMAN
        assert artifact.source == ArtifactSource.HUMAN

    def test_human_artifact_rejection_triggers_retry(
        self, sample_amendment, sample_checkpoint, memory_checkpoint_dag
    ):
        """If guard rejects human artifact, retry loop continues."""
        from atomicguard.domain.workflow import HumanAmendmentProcessor

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)
        memory_checkpoint_dag.store_amendment(sample_amendment)

        processor = HumanAmendmentProcessor(checkpoint_dag=memory_checkpoint_dag)

        # Simulate guard rejection of human artifact
        result = processor.process_amendment_with_guard(
            sample_amendment.amendment_id, guard_passes=False
        )

        # Should return indication to continue retry loop
        assert result.should_retry is True

    def test_human_amendment_monotonic(
        self, sample_amendment, sample_checkpoint, memory_checkpoint_dag
    ):
        """Human amendments only add information (Def 12 ⊕ operator)."""
        from atomicguard.domain.workflow import HumanAmendmentProcessor

        memory_checkpoint_dag.store_checkpoint(sample_checkpoint)
        memory_checkpoint_dag.store_amendment(sample_amendment)

        processor = HumanAmendmentProcessor(checkpoint_dag=memory_checkpoint_dag)

        # Get context before and after amendment
        original_context = processor.get_checkpoint_context(
            sample_checkpoint.checkpoint_id
        )
        amended_context = processor.apply_amendment_to_context(
            sample_amendment.amendment_id
        )

        # Original information should be preserved
        assert original_context.specification in amended_context.specification
