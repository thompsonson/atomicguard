"""
Extension 02: Artifact Extraction Tests.

TDD tests for implementing:
- Predicate base class (Def 18)
- StatusPredicate, ActionPairPredicate, WorkflowPredicate, SourcePredicate
- Compound predicates (AND, OR, NOT)
- Extraction function E: R x Phi -> 2^R (Def 17)
- Extraction pagination (limit, offset, order_by)
- Theorem 3: Extraction Invariance
"""

from atomicguard.domain.models import Artifact, ArtifactStatus


class TestPredicateInterface:
    """Tests for Predicate base class (Def 18)."""

    def test_predicate_has_matches_method(self):
        """Predicate defines matches(artifact) -> bool."""
        from atomicguard.domain.extraction import Predicate

        # Predicate should be an ABC with matches method
        assert hasattr(Predicate, "matches")

    def test_predicate_is_callable(self, sample_artifact):
        """Predicate instances are callable as Phi(r)."""
        from atomicguard.domain.extraction import StatusPredicate

        predicate = StatusPredicate(ArtifactStatus.PENDING)
        # Should be callable: predicate(artifact) == predicate.matches(artifact)
        result = predicate(sample_artifact)
        assert isinstance(result, bool)


class TestStatusPredicate:
    """Tests for Phi_status filter."""

    def test_matches_accepted_status(self, sample_context_snapshot):
        """StatusPredicate(ACCEPTED) matches accepted artifacts."""
        from atomicguard.domain.extraction import StatusPredicate

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )

        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)
        assert predicate.matches(artifact) is True

    def test_matches_rejected_status(self, sample_context_snapshot):
        """StatusPredicate(REJECTED) matches rejected artifacts."""
        from atomicguard.domain.extraction import StatusPredicate
        from atomicguard.domain.models import GuardResult

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=GuardResult(passed=False, feedback="Error"),
            context=sample_context_snapshot,
        )

        predicate = StatusPredicate(ArtifactStatus.REJECTED)
        assert predicate.matches(artifact) is True

    def test_does_not_match_different_status(self, sample_artifact):
        """StatusPredicate(ACCEPTED) rejects PENDING artifacts."""
        from atomicguard.domain.extraction import StatusPredicate

        # sample_artifact has PENDING status
        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)
        assert predicate.matches(sample_artifact) is False

    def test_matches_multiple_statuses(self, sample_context_snapshot):
        """StatusPredicate(ACCEPTED, SUPERSEDED) matches either."""
        from atomicguard.domain.extraction import StatusPredicate

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )

        predicate = StatusPredicate(ArtifactStatus.ACCEPTED, ArtifactStatus.SUPERSEDED)
        assert predicate.matches(artifact) is True


class TestActionPairPredicate:
    """Tests for Phi_action_pair filter."""

    def test_matches_action_pair_id(self, sample_artifact):
        """ActionPairPredicate('ap-001') matches artifacts from ap-001."""
        from atomicguard.domain.extraction import ActionPairPredicate

        # sample_artifact has action_pair_id="ap-001"
        predicate = ActionPairPredicate("ap-001")
        assert predicate.matches(sample_artifact) is True

    def test_does_not_match_different_action_pair(self, sample_artifact):
        """ActionPairPredicate('ap-002') rejects ap-001 artifacts."""
        from atomicguard.domain.extraction import ActionPairPredicate

        predicate = ActionPairPredicate("ap-002")
        assert predicate.matches(sample_artifact) is False


class TestWorkflowPredicate:
    """Tests for Phi_workflow filter."""

    def test_matches_workflow_id(self, sample_artifact):
        """WorkflowPredicate('test-workflow-001') matches workflow artifacts."""
        from atomicguard.domain.extraction import WorkflowPredicate

        # sample_artifact has workflow_id="test-workflow-001"
        predicate = WorkflowPredicate("test-workflow-001")
        assert predicate.matches(sample_artifact) is True

    def test_does_not_match_different_workflow(self, sample_artifact):
        """WorkflowPredicate('wf-002') rejects test-workflow-001 artifacts."""
        from atomicguard.domain.extraction import WorkflowPredicate

        predicate = WorkflowPredicate("wf-002")
        assert predicate.matches(sample_artifact) is False


class TestSourcePredicate:
    """Tests for Phi_source filter."""

    def test_matches_generator_source(self, sample_context_snapshot):
        """SourcePredicate(GENERATED) matches LLM artifacts."""
        from atomicguard.domain.extraction import SourcePredicate
        from atomicguard.domain.models import ArtifactSource

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
            source=ArtifactSource.GENERATED,
        )

        predicate = SourcePredicate(ArtifactSource.GENERATED)
        assert predicate.matches(artifact) is True

    def test_matches_human_source(self, sample_context_snapshot):
        """SourcePredicate(HUMAN) matches human amendments."""
        from atomicguard.domain.extraction import SourcePredicate
        from atomicguard.domain.models import ArtifactSource

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="human provided code",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=sample_context_snapshot,
            source=ArtifactSource.HUMAN,
        )

        predicate = SourcePredicate(ArtifactSource.HUMAN)
        assert predicate.matches(artifact) is True


class TestCompoundPredicates:
    """Tests for Phi_1 AND Phi_2 composition."""

    def test_and_predicate_requires_both(self, sample_context_snapshot):
        """AndPredicate(p1, p2) requires both to match."""
        from atomicguard.domain.extraction import (
            AndPredicate,
            StatusPredicate,
            WorkflowPredicate,
        )

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )

        p1 = StatusPredicate(ArtifactStatus.ACCEPTED)
        p2 = WorkflowPredicate("wf-001")

        combined = AndPredicate(p1, p2)
        assert combined.matches(artifact) is True

    def test_and_predicate_fails_if_one_fails(self, sample_context_snapshot):
        """AndPredicate fails if either predicate fails."""
        from atomicguard.domain.extraction import (
            AndPredicate,
            StatusPredicate,
            WorkflowPredicate,
        )

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,  # Different status
            guard_result=None,
            context=sample_context_snapshot,
        )

        p1 = StatusPredicate(ArtifactStatus.ACCEPTED)  # Won't match
        p2 = WorkflowPredicate("wf-001")  # Will match

        combined = AndPredicate(p1, p2)
        assert combined.matches(artifact) is False

    def test_or_predicate_matches_either(self, sample_context_snapshot):
        """OrPredicate(p1, p2) matches if either matches."""
        from atomicguard.domain.extraction import (
            OrPredicate,
            StatusPredicate,
            WorkflowPredicate,
        )

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=None,
            context=sample_context_snapshot,
        )

        p1 = StatusPredicate(ArtifactStatus.ACCEPTED)  # Won't match
        p2 = WorkflowPredicate("wf-001")  # Will match

        combined = OrPredicate(p1, p2)
        assert combined.matches(artifact) is True

    def test_nested_compound_predicates(self, sample_context_snapshot):
        """(p1 AND p2) OR p3 works correctly."""
        from atomicguard.domain.extraction import (
            ActionPairPredicate,
            AndPredicate,
            OrPredicate,
            StatusPredicate,
            WorkflowPredicate,
        )

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="wf-001",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-special",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=None,
            context=sample_context_snapshot,
        )

        p1 = StatusPredicate(ArtifactStatus.ACCEPTED)
        p2 = WorkflowPredicate("wf-001")
        p3 = ActionPairPredicate("ap-special")

        # (ACCEPTED AND wf-001) OR ap-special
        # p1 AND p2 = False (status is REJECTED)
        # p3 = True (action_pair matches)
        # Result: False OR True = True
        combined = OrPredicate(AndPredicate(p1, p2), p3)
        assert combined.matches(artifact) is True

    def test_not_predicate_inverts(self, sample_artifact):
        """NotPredicate(p) inverts the result."""
        from atomicguard.domain.extraction import NotPredicate, StatusPredicate

        # sample_artifact has PENDING status
        p = StatusPredicate(ArtifactStatus.ACCEPTED)
        assert p.matches(sample_artifact) is False

        not_p = NotPredicate(p)
        assert not_p.matches(sample_artifact) is True


class TestExtractionFunction:
    """Tests for E: R x Phi -> 2^R (Def 17)."""

    def test_extract_returns_list(self, populated_dag):
        """extract() returns list of matching artifacts."""
        from atomicguard.domain.extraction import StatusPredicate, extract

        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)
        result = extract(populated_dag, predicate)

        assert isinstance(result, list)
        assert all(isinstance(a, Artifact) for a in result)

    def test_extract_empty_when_no_matches(self, memory_dag):
        """extract() returns empty list when nothing matches."""
        from atomicguard.domain.extraction import StatusPredicate, extract

        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)
        result = extract(memory_dag, predicate)

        assert result == []

    def test_extract_all_matching(self, populated_dag):
        """extract() returns ALL matching artifacts."""
        from atomicguard.domain.extraction import StatusPredicate, extract

        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)
        result = extract(populated_dag, predicate)

        # Verify all results match predicate
        for artifact in result:
            assert artifact.status == ArtifactStatus.ACCEPTED

        # Verify we got all matching (populated_dag has 5 ACCEPTED)
        assert len(result) == 5

    def test_extract_preserves_artifact_integrity(self, populated_dag):
        """Extracted artifacts are identical to stored."""
        from atomicguard.domain.extraction import StatusPredicate, extract

        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)
        result = extract(populated_dag, predicate)

        # Each extracted artifact should match stored version
        for artifact in result:
            stored = populated_dag.get_artifact(artifact.artifact_id)
            assert artifact == stored

    def test_extract_does_not_modify_dag(self, populated_dag):
        """Extraction is read-only (Theorem 3)."""
        from atomicguard.domain.extraction import StatusPredicate, extract

        # Count artifacts before
        all_before = list(populated_dag._artifacts.values())
        count_before = len(all_before)

        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)
        _ = extract(populated_dag, predicate)

        # Count artifacts after - should be unchanged
        all_after = list(populated_dag._artifacts.values())
        count_after = len(all_after)

        assert count_before == count_after


class TestExtractionPagination:
    """Tests for limit, offset, order_by."""

    def test_limit_restricts_results(self, populated_dag):
        """extract(limit=5) returns at most 5 artifacts."""
        from atomicguard.domain.extraction import extract

        result = extract(populated_dag, predicate=None, limit=5)
        assert len(result) <= 5

    def test_offset_skips_results(self, populated_dag):
        """extract(offset=10) skips first 10 matches."""
        from atomicguard.domain.extraction import extract

        all_results = extract(populated_dag, predicate=None)
        offset_results = extract(populated_dag, predicate=None, offset=10)

        # Offset results should be subset of all results minus first 10
        if len(all_results) > 10:
            assert offset_results == all_results[10:]
        else:
            assert offset_results == []

    def test_order_by_created_at(self, populated_dag):
        """extract(order_by='created_at') orders chronologically."""
        from atomicguard.domain.extraction import extract

        result = extract(populated_dag, predicate=None, order_by="created_at")

        # Verify ordering
        for i in range(len(result) - 1):
            assert result[i].created_at <= result[i + 1].created_at

    def test_order_by_descending(self, populated_dag):
        """extract(order_by='-created_at') orders reverse chronologically."""
        from atomicguard.domain.extraction import extract

        result = extract(populated_dag, predicate=None, order_by="-created_at")

        # Verify descending order
        for i in range(len(result) - 1):
            assert result[i].created_at >= result[i + 1].created_at

    def test_pagination_with_predicate(self, populated_dag):
        """Pagination works with predicate filtering."""
        from atomicguard.domain.extraction import StatusPredicate, extract

        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)

        # Get all matching
        all_matching = extract(populated_dag, predicate)

        # Get paginated
        paginated = extract(populated_dag, predicate, limit=2, offset=1)

        # Paginated should be subset
        assert len(paginated) <= 2
        assert paginated == all_matching[1:3]


class TestExtractionInvariance:
    """Tests for Theorem 3: Extraction Invariance."""

    def test_extraction_idempotent(self, populated_dag):
        """E(E(R, Phi), Phi) == E(R, Phi) (Corollary 3.1)."""
        from atomicguard.domain.extraction import StatusPredicate, extract

        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)

        # First extraction
        first_result = extract(populated_dag, predicate)

        # Create a new DAG with only extracted artifacts
        from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG

        subset_dag = InMemoryArtifactDAG()
        for artifact in first_result:
            subset_dag.store(artifact)

        # Second extraction from subset
        second_result = extract(subset_dag, predicate)

        # Results should be identical (idempotent)
        assert len(first_result) == len(second_result)
        first_ids = {a.artifact_id for a in first_result}
        second_ids = {a.artifact_id for a in second_result}
        assert first_ids == second_ids

    def test_extraction_referentially_transparent(self, populated_dag):
        """Same R and Phi yield identical results (Corollary 3.2)."""
        from atomicguard.domain.extraction import StatusPredicate, extract

        predicate = StatusPredicate(ArtifactStatus.ACCEPTED)

        # Multiple extractions with same inputs
        result1 = extract(populated_dag, predicate)
        result2 = extract(populated_dag, predicate)
        result3 = extract(populated_dag, predicate)

        # All should be identical
        ids1 = [a.artifact_id for a in result1]
        ids2 = [a.artifact_id for a in result2]
        ids3 = [a.artifact_id for a in result3]

        assert ids1 == ids2 == ids3
