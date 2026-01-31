#!/usr/bin/env python3
"""
Extension 02: Artifact Extraction (Definitions 17-18).

Demonstrates repository query capabilities:
- Predicate classes: StatusPredicate, WorkflowPredicate, ActionPairPredicate, SourcePredicate
- Compound predicates: AndPredicate, OrPredicate, NotPredicate
- extract(): E: ℛ × Φ → 2^ℛ (read-only query function)
- Pagination: limit, offset, order_by

The extraction function is:
- Read-only: Never modifies repository state
- Idempotent: Same predicate → same results (Theorem 3)
- Composable: Predicates can be combined with AND/OR/NOT

Run: python -m examples.basics.06_extraction
"""

from atomicguard.domain.extraction import (
    ActionPairPredicate,
    AndPredicate,
    NotPredicate,
    OrPredicate,
    SourcePredicate,
    StatusPredicate,
    WorkflowPredicate,
    extract,
)
from atomicguard.domain.models import (
    Artifact,
    ArtifactSource,
    ArtifactStatus,
    ContextSnapshot,
)
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


def create_context_snapshot(workflow_id: str = "wf-demo") -> ContextSnapshot:
    """Create a minimal context snapshot for test artifacts."""
    return ContextSnapshot(
        workflow_id=workflow_id,
        specification="Demo specification",
        constraints="",
        feedback_history=(),
        dependency_artifacts=(),
    )


def create_artifact(
    artifact_id: str,
    workflow_id: str,
    action_pair_id: str,
    status: ArtifactStatus,
    created_at: str,
    source: ArtifactSource = ArtifactSource.GENERATED,
) -> Artifact:
    """Create a test artifact with specified properties."""
    return Artifact(
        artifact_id=artifact_id,
        workflow_id=workflow_id,
        content=f"# Artifact {artifact_id}",
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id=action_pair_id,
        created_at=created_at,
        attempt_number=1,
        status=status,
        guard_result=None,
        context=create_context_snapshot(workflow_id),
        source=source,
    )


def populate_repository(dag: InMemoryArtifactDAG) -> None:
    """
    Populate repository with sample artifacts for demonstration.

    Creates artifacts across:
    - 2 workflows (workflow-1, workflow-2)
    - 3 action pairs (g_test, g_impl, g_review)
    - 3 statuses (PENDING, ACCEPTED, REJECTED)
    - 2 sources (GENERATED, HUMAN)
    """
    artifacts = [
        # Workflow 1 - g_test
        create_artifact(
            "art-001",
            "workflow-1",
            "g_test",
            ArtifactStatus.REJECTED,
            "2025-01-01T10:00:00Z",
        ),
        create_artifact(
            "art-002",
            "workflow-1",
            "g_test",
            ArtifactStatus.ACCEPTED,
            "2025-01-01T10:05:00Z",
        ),
        # Workflow 1 - g_impl
        create_artifact(
            "art-003",
            "workflow-1",
            "g_impl",
            ArtifactStatus.REJECTED,
            "2025-01-01T10:10:00Z",
        ),
        create_artifact(
            "art-004",
            "workflow-1",
            "g_impl",
            ArtifactStatus.REJECTED,
            "2025-01-01T10:15:00Z",
        ),
        create_artifact(
            "art-005",
            "workflow-1",
            "g_impl",
            ArtifactStatus.ACCEPTED,
            "2025-01-01T10:20:00Z",
        ),
        # Workflow 1 - g_review (human artifact)
        create_artifact(
            "art-006",
            "workflow-1",
            "g_review",
            ArtifactStatus.ACCEPTED,
            "2025-01-01T10:25:00Z",
            ArtifactSource.HUMAN,
        ),
        # Workflow 2 - g_test
        create_artifact(
            "art-007",
            "workflow-2",
            "g_test",
            ArtifactStatus.ACCEPTED,
            "2025-01-02T09:00:00Z",
        ),
        # Workflow 2 - g_impl
        create_artifact(
            "art-008",
            "workflow-2",
            "g_impl",
            ArtifactStatus.PENDING,
            "2025-01-02T09:05:00Z",
        ),
        create_artifact(
            "art-009",
            "workflow-2",
            "g_impl",
            ArtifactStatus.REJECTED,
            "2025-01-02T09:10:00Z",
        ),
        # Workflow 2 - human intervention
        create_artifact(
            "art-010",
            "workflow-2",
            "g_impl",
            ArtifactStatus.ACCEPTED,
            "2025-01-02T09:15:00Z",
            ArtifactSource.HUMAN,
        ),
    ]

    for artifact in artifacts:
        dag.store(artifact)

    print(f"Populated repository with {len(artifacts)} artifacts")
    print("  Workflows: workflow-1, workflow-2")
    print("  Action pairs: g_test, g_impl, g_review")
    print(f"  Statuses: {[s.value for s in ArtifactStatus]}")


def demo_simple_predicates(dag: InMemoryArtifactDAG) -> None:
    """
    Demonstrate single-field filtering with simple predicates.

    Available predicates:
    - StatusPredicate: Filter by ArtifactStatus
    - WorkflowPredicate: Filter by workflow_id
    - ActionPairPredicate: Filter by action_pair_id
    - SourcePredicate: Filter by ArtifactSource
    """
    print("\n" + "=" * 60)
    print("SIMPLE PREDICATES - Single Field Filtering")
    print("=" * 60)

    # StatusPredicate: Filter by status
    print("\n--- StatusPredicate ---")
    accepted = extract(dag, StatusPredicate(ArtifactStatus.ACCEPTED))
    print(f"ACCEPTED artifacts: {len(accepted)}")
    for a in accepted:
        print(f"  {a.artifact_id}: {a.workflow_id}/{a.action_pair_id}")

    rejected = extract(dag, StatusPredicate(ArtifactStatus.REJECTED))
    print(f"\nREJECTED artifacts: {len(rejected)}")

    # Multiple statuses in one predicate
    not_pending = extract(
        dag, StatusPredicate(ArtifactStatus.ACCEPTED, ArtifactStatus.REJECTED)
    )
    print(f"ACCEPTED or REJECTED: {len(not_pending)}")

    # WorkflowPredicate: Filter by workflow
    print("\n--- WorkflowPredicate ---")
    wf1 = extract(dag, WorkflowPredicate("workflow-1"))
    print(f"workflow-1 artifacts: {len(wf1)}")

    wf2 = extract(dag, WorkflowPredicate("workflow-2"))
    print(f"workflow-2 artifacts: {len(wf2)}")

    # ActionPairPredicate: Filter by step/guard
    print("\n--- ActionPairPredicate ---")
    g_test = extract(dag, ActionPairPredicate("g_test"))
    print(f"g_test artifacts: {len(g_test)}")

    g_impl = extract(dag, ActionPairPredicate("g_impl"))
    print(f"g_impl artifacts: {len(g_impl)}")

    # SourcePredicate: Filter by source
    print("\n--- SourcePredicate ---")
    human = extract(dag, SourcePredicate(ArtifactSource.HUMAN))
    print(f"HUMAN artifacts: {len(human)}")
    for a in human:
        print(f"  {a.artifact_id}: {a.workflow_id}/{a.action_pair_id}")

    generated = extract(dag, SourcePredicate(ArtifactSource.GENERATED))
    print(f"GENERATED artifacts: {len(generated)}")


def demo_compound_predicates(dag: InMemoryArtifactDAG) -> None:
    """
    Demonstrate combining predicates with logical operators.

    Compound predicates:
    - AndPredicate(p1, p2): Φ₁ ∧ Φ₂ - Both must match
    - OrPredicate(p1, p2): Φ₁ ∨ Φ₂ - Either matches
    - NotPredicate(p): ¬Φ - Inverts the result
    """
    print("\n" + "=" * 60)
    print("COMPOUND PREDICATES - Logical Combinations")
    print("=" * 60)

    # AND: ACCEPTED artifacts from workflow-1
    print("\n--- AndPredicate (Φ₁ ∧ Φ₂) ---")
    wf1_accepted = extract(
        dag,
        AndPredicate(
            WorkflowPredicate("workflow-1"),
            StatusPredicate(ArtifactStatus.ACCEPTED),
        ),
    )
    print(f"workflow-1 AND ACCEPTED: {len(wf1_accepted)}")
    for a in wf1_accepted:
        print(f"  {a.artifact_id}: {a.action_pair_id}")

    # Triple AND: workflow-1 AND g_impl AND REJECTED
    wf1_impl_rejected = extract(
        dag,
        AndPredicate(
            WorkflowPredicate("workflow-1"),
            AndPredicate(
                ActionPairPredicate("g_impl"),
                StatusPredicate(ArtifactStatus.REJECTED),
            ),
        ),
    )
    print(f"\nworkflow-1 AND g_impl AND REJECTED: {len(wf1_impl_rejected)}")

    # OR: g_test OR g_review
    print("\n--- OrPredicate (Φ₁ ∨ Φ₂) ---")
    test_or_review = extract(
        dag,
        OrPredicate(
            ActionPairPredicate("g_test"),
            ActionPairPredicate("g_review"),
        ),
    )
    print(f"g_test OR g_review: {len(test_or_review)}")

    # NOT: Everything except REJECTED
    print("\n--- NotPredicate (¬Φ) ---")
    not_rejected = extract(dag, NotPredicate(StatusPredicate(ArtifactStatus.REJECTED)))
    print(f"NOT REJECTED: {len(not_rejected)}")

    # Complex: (workflow-1 OR workflow-2) AND NOT PENDING
    print("\n--- Complex: (wf-1 OR wf-2) AND NOT PENDING ---")
    complex_pred = AndPredicate(
        OrPredicate(
            WorkflowPredicate("workflow-1"),
            WorkflowPredicate("workflow-2"),
        ),
        NotPredicate(StatusPredicate(ArtifactStatus.PENDING)),
    )
    results = extract(dag, complex_pred)
    print(f"Result count: {len(results)}")


def demo_pagination(dag: InMemoryArtifactDAG) -> None:
    """
    Demonstrate pagination and ordering.

    Options:
    - limit: Maximum number of results
    - offset: Skip first N results
    - order_by: Field name (prefix with '-' for descending)
    """
    print("\n" + "=" * 60)
    print("PAGINATION AND ORDERING")
    print("=" * 60)

    # Get all artifacts for reference
    all_artifacts = extract(dag)
    print(f"\nTotal artifacts: {len(all_artifacts)}")

    # Limit: First 3 artifacts
    print("\n--- Limit ---")
    first_3 = extract(dag, limit=3)
    print("First 3 artifacts:")
    for a in first_3:
        print(f"  {a.artifact_id}: {a.created_at}")

    # Offset + Limit: Skip 3, take 3
    print("\n--- Offset + Limit ---")
    page_2 = extract(dag, limit=3, offset=3)
    print("Skip 3, take 3:")
    for a in page_2:
        print(f"  {a.artifact_id}: {a.created_at}")

    # Order by created_at ascending (default)
    print("\n--- Order By (ascending) ---")
    oldest_first = extract(dag, order_by="created_at", limit=3)
    print("Oldest 3:")
    for a in oldest_first:
        print(f"  {a.artifact_id}: {a.created_at}")

    # Order by created_at descending
    print("\n--- Order By (descending) ---")
    newest_first = extract(dag, order_by="-created_at", limit=3)
    print("Newest 3:")
    for a in newest_first:
        print(f"  {a.artifact_id}: {a.created_at}")

    # Combine predicate with pagination
    print("\n--- Predicate + Pagination ---")
    oldest_rejected = extract(
        dag,
        StatusPredicate(ArtifactStatus.REJECTED),
        limit=2,
        order_by="created_at",
    )
    print("Oldest 2 REJECTED:")
    for a in oldest_rejected:
        print(f"  {a.artifact_id}: {a.workflow_id}/{a.action_pair_id}")


def demo_extraction_invariance(dag: InMemoryArtifactDAG) -> None:
    """
    Demonstrate Theorem 3: Extraction Invariance.

    The extract() function is:
    - Idempotent: extract(R, Φ) == extract(R, Φ) (same results)
    - Read-only: Does not modify repository state
    - Referentially transparent: Result depends only on inputs
    """
    print("\n" + "=" * 60)
    print("THEOREM 3: EXTRACTION INVARIANCE")
    print("=" * 60)

    pred = AndPredicate(
        WorkflowPredicate("workflow-1"),
        StatusPredicate(ArtifactStatus.ACCEPTED),
    )

    # Multiple extractions with same predicate
    result_1 = extract(dag, pred)
    result_2 = extract(dag, pred)
    result_3 = extract(dag, pred)

    # Results should be identical
    ids_1 = [a.artifact_id for a in result_1]
    ids_2 = [a.artifact_id for a in result_2]
    ids_3 = [a.artifact_id for a in result_3]

    print(f"\nExtraction 1: {ids_1}")
    print(f"Extraction 2: {ids_2}")
    print(f"Extraction 3: {ids_3}")
    print(f"\nIdempotent (same results): {ids_1 == ids_2 == ids_3}")

    # Repository unchanged
    total_before = len(extract(dag))
    _ = extract(dag, StatusPredicate(ArtifactStatus.REJECTED))
    total_after = len(extract(dag))
    print(f"Read-only (repo unchanged): {total_before == total_after}")


def main() -> None:
    """Run all extraction demonstrations."""
    print("\n" + "=" * 60)
    print("EXTENSION 02: ARTIFACT EXTRACTION")
    print("Definitions 17-18 from the formal specification")
    print("=" * 60)

    # Create and populate repository
    dag = InMemoryArtifactDAG()
    populate_repository(dag)

    # Run demos
    demo_simple_predicates(dag)
    demo_compound_predicates(dag)
    demo_pagination(dag)
    demo_extraction_invariance(dag)

    print("\n" + "=" * 60)
    print("SUCCESS: All extraction demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
