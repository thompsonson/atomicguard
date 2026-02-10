"""
Extension 01: Versioned Environment Tests.

TDD tests for implementing:
- W_ref content-addressed workflow hashing (Def 11)
- workflow_ref field on Artifact (Def 10)
- Structured metadata on Artifact (Def 10)
- Configuration amendment ⊕ operator (Def 12)
"""

import pytest

from atomicguard.domain.prompts import PromptTemplate


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
        template = PromptTemplate(role="test", constraints="", task="test")
        artifact = mock_generator.generate(
            context=sample_context,
            template=template,
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
