#!/usr/bin/env python3
"""
Extension 07: Incremental Execution (Definitions 33-37).

Demonstrates configuration-based change detection for selective re-execution:
- compute_config_ref(): Content-addressable fingerprint (Ψ_ref)
- Dependency graph and Merkle propagation
- unchanged() predicate for skip/execute decision
- Invalidation cascade algorithm

Like incremental builds in CI/CD - only changed action pairs re-execute.

Run: python -m examples.basics.08_incremental
"""

from dataclasses import dataclass

from atomicguard.domain.models import (
    Artifact,
    ArtifactSource,
    ArtifactStatus,
    ContextSnapshot,
)
from atomicguard.domain.workflow import compute_config_ref, compute_workflow_ref

# =============================================================================
# MOCK ARTIFACT FACTORY
# =============================================================================


def make_mock_artifact(
    action_pair_id: str,
    content: str,
    config_ref: str | None = None,
    status: ArtifactStatus = ArtifactStatus.ACCEPTED,
) -> Artifact:
    """Create a mock artifact for demonstration."""
    return Artifact(
        artifact_id=f"art-{action_pair_id}",
        workflow_id="wf-demo",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id=action_pair_id,
        created_at="2024-01-01T00:00:00Z",
        attempt_number=1,
        status=status,
        guard_result=status == ArtifactStatus.ACCEPTED,
        feedback="",
        context=ContextSnapshot(
            workflow_id="wf-demo",
            specification="Demo specification",
            constraints="",
            feedback_history=(),
        ),
        source=ArtifactSource.GENERATED,
        config_ref=config_ref,
    )


# =============================================================================
# MOCK REPOSITORY FOR DEMONSTRATION
# =============================================================================


@dataclass
class MockRepository:
    """Simple repository storing artifacts by config_ref."""

    artifacts: dict[str, Artifact]

    def lookup(self, action_pair_id: str, config_ref: str) -> Artifact | None:
        """Look up accepted artifact matching config_ref (Definition 36)."""
        for artifact in self.artifacts.values():
            if (
                artifact.action_pair_id == action_pair_id
                and artifact.config_ref == config_ref
                and artifact.status == ArtifactStatus.ACCEPTED
            ):
                return artifact
        return None


# =============================================================================
# DEMO 1: CONFIGURATION REFERENCE (Definition 33)
# =============================================================================


def demo_config_ref() -> None:
    """
    Demonstrate Ψ_ref computation (Definition 33).

    Ψ_ref = hash(prompt, model, guard_config, upstream_refs, artifact_hashes)

    Key properties:
    - Deterministic: Same config → same hash
    - Content-addressed: Changes in any input → different hash
    - Scoped to action pair (unlike W_ref which covers entire workflow)
    """
    print("=" * 60)
    print("CONFIGURATION REFERENCE (Ψ_ref) - Definition 33")
    print("=" * 60)

    workflow_config = {
        "name": "TDD Workflow",
        "model": "qwen2.5-coder:7b",
        "action_pairs": {
            "g_test": {
                "guard": "syntax",
                "guard_config": {"strict": True},
            },
            "g_impl": {
                "guard": "dynamic_test",
                "requires": ["g_test"],
            },
        },
    }

    prompt_config = {
        "g_test": {
            "role": "Test writer",
            "task": "Write pytest tests for a Stack class",
        },
        "g_impl": {
            "role": "Implementer",
            "task": "Implement the Stack class to pass tests",
        },
    }

    # Compute Ψ_ref for root action pair (no dependencies)
    psi_test = compute_config_ref("g_test", workflow_config, prompt_config)
    print(f"\nΨ_ref(g_test) = {psi_test[:32]}...")
    print("  (Root action pair - computed from own config only)")

    # Same config → same hash (deterministic)
    psi_test_2 = compute_config_ref("g_test", workflow_config, prompt_config)
    print(f"\nDeterministic check: same hash on recompute = {psi_test == psi_test_2}")

    # Compare W_ref vs Ψ_ref scope
    w_ref = compute_workflow_ref(workflow_config, store=False)
    print("\nW_ref vs Ψ_ref scope:")
    print(f"  W_ref (workflow):     {w_ref[:32]}...")
    print(f"  Ψ_ref (action pair):  {psi_test[:32]}...")
    print("  W_ref covers entire workflow; Ψ_ref is per-action-pair")


# =============================================================================
# DEMO 2: MERKLE PROPAGATION (Definition 35)
# =============================================================================


def demo_merkle_propagation() -> None:
    """
    Demonstrate Merkle propagation through dependency chain (Definition 35).

    Property: Upstream changes cascade to all downstream action pairs.

    g_test → g_impl → g_review
       ↓         ↓          ↓
    Change g_test config → all Ψ_ref values change
    """
    print("\n" + "=" * 60)
    print("MERKLE PROPAGATION - Definition 35")
    print("=" * 60)

    workflow_config = {
        "action_pairs": {
            "g_test": {"guard": "syntax"},
            "g_impl": {"guard": "dynamic_test", "requires": ["g_test"]},
            "g_review": {"guard": "human", "requires": ["g_impl"]},
        },
    }

    # Version 1: Original prompts
    prompt_v1 = {
        "g_test": {"role": "Test writer"},
        "g_impl": {"role": "Implementer"},
        "g_review": {"role": "Reviewer"},
    }

    print("\n--- Version 1: Original Configuration ---")

    # Compute in topological order
    psi_test_v1 = compute_config_ref("g_test", workflow_config, prompt_v1)
    art_test_v1 = make_mock_artifact("g_test", "tests v1", config_ref=psi_test_v1)

    psi_impl_v1 = compute_config_ref(
        "g_impl", workflow_config, prompt_v1, {"g_test": art_test_v1}
    )
    art_impl_v1 = make_mock_artifact("g_impl", "impl v1", config_ref=psi_impl_v1)

    psi_review_v1 = compute_config_ref(
        "g_review", workflow_config, prompt_v1, {"g_impl": art_impl_v1}
    )

    print(f"  Ψ_ref(g_test):   {psi_test_v1[:16]}...")
    print(f"  Ψ_ref(g_impl):   {psi_impl_v1[:16]}...")
    print(f"  Ψ_ref(g_review): {psi_review_v1[:16]}...")

    # Version 2: Change g_test prompt (root change)
    prompt_v2 = {
        "g_test": {"role": "Test writer UPDATED"},  # Changed!
        "g_impl": {"role": "Implementer"},
        "g_review": {"role": "Reviewer"},
    }

    print("\n--- Version 2: g_test prompt changed ---")

    psi_test_v2 = compute_config_ref("g_test", workflow_config, prompt_v2)
    art_test_v2 = make_mock_artifact("g_test", "tests v2", config_ref=psi_test_v2)

    psi_impl_v2 = compute_config_ref(
        "g_impl", workflow_config, prompt_v2, {"g_test": art_test_v2}
    )
    art_impl_v2 = make_mock_artifact("g_impl", "impl v2", config_ref=psi_impl_v2)

    psi_review_v2 = compute_config_ref(
        "g_review", workflow_config, prompt_v2, {"g_impl": art_impl_v2}
    )

    print(
        f"  Ψ_ref(g_test):   {psi_test_v2[:16]}... {'CHANGED' if psi_test_v1 != psi_test_v2 else 'same'}"
    )
    print(
        f"  Ψ_ref(g_impl):   {psi_impl_v2[:16]}... {'CHANGED' if psi_impl_v1 != psi_impl_v2 else 'same'}"
    )
    print(
        f"  Ψ_ref(g_review): {psi_review_v2[:16]}... {'CHANGED' if psi_review_v1 != psi_review_v2 else 'same'}"
    )

    print("\nMerkle Propagation: Root change cascades to ALL downstream!")


# =============================================================================
# DEMO 3: CHANGE DETECTION (Definition 36)
# =============================================================================


def demo_change_detection() -> None:
    """
    Demonstrate unchanged() predicate (Definition 36).

    unchanged(ap) ⟺ ∃a ∈ ℛ : a.config_ref = Ψ_ref_current(ap) ∧ a.status = ACCEPTED

    Decision:
    - unchanged → skip execution
    - changed → execute action pair
    """
    print("\n" + "=" * 60)
    print("CHANGE DETECTION - Definition 36")
    print("=" * 60)

    workflow_config = {
        "action_pairs": {
            "g_test": {"guard": "syntax"},
            "g_impl": {"guard": "dynamic_test", "requires": ["g_test"]},
        },
    }
    prompt_config = {
        "g_test": {"role": "Test writer"},
        "g_impl": {"role": "Implementer"},
    }

    # Compute current refs
    psi_test = compute_config_ref("g_test", workflow_config, prompt_config)

    # Simulate repository with one accepted artifact
    repo = MockRepository(
        artifacts={
            "art-g_test": make_mock_artifact("g_test", "tests", config_ref=psi_test),
        }
    )

    print(f"\nCurrent Ψ_ref(g_test): {psi_test[:32]}...")
    print("Repository contains accepted artifact with matching config_ref: True")

    # Check unchanged predicate
    match = repo.lookup("g_test", psi_test)
    is_unchanged = match is not None
    print(f"\nunchanged(g_test) = {is_unchanged}")
    print(f"Decision: {'SKIP' if is_unchanged else 'EXECUTE'}")

    # Now compute for g_impl (no matching artifact in repo)
    art_test = make_mock_artifact("g_test", "tests", config_ref=psi_test)
    psi_impl = compute_config_ref(
        "g_impl", workflow_config, prompt_config, {"g_test": art_test}
    )
    match_impl = repo.lookup("g_impl", psi_impl)
    is_unchanged_impl = match_impl is not None

    print(f"\nCurrent Ψ_ref(g_impl): {psi_impl[:32]}...")
    print("Repository contains matching artifact: False")
    print(f"\nunchanged(g_impl) = {is_unchanged_impl}")
    print(f"Decision: {'SKIP' if is_unchanged_impl else 'EXECUTE'}")


# =============================================================================
# DEMO 4: INVALIDATION CASCADE (Algorithm 1)
# =============================================================================


def invalidated_action_pairs(
    changed_aps: set[str],
    dependency_graph: dict[str, list[str]],
) -> set[str]:
    """
    Compute all invalidated action pairs (Algorithm 1).

    Given set of directly changed action pairs, return all that need re-execution
    including those invalidated through dependency cascade.
    """
    invalidated = set(changed_aps)

    # Topological order (simple BFS-based)
    all_aps = set(dependency_graph.keys())
    for ap in dependency_graph:
        all_aps.update(dependency_graph[ap])

    # Process in order, propagating invalidation
    for ap in sorted(all_aps):
        deps = dependency_graph.get(ap, [])
        if any(dep in invalidated for dep in deps):
            invalidated.add(ap)

    return invalidated


def demo_invalidation_cascade() -> None:
    """
    Demonstrate invalidation cascade (Algorithm 1).

    Given directly changed action pairs, compute full set of invalidated APs.
    """
    print("\n" + "=" * 60)
    print("INVALIDATION CASCADE - Algorithm 1")
    print("=" * 60)

    # Dependency graph: g_config → g_add → g_coder
    #                          → g_bdd ↗
    dependency_graph = {
        "g_config": [],
        "g_add": ["g_config"],
        "g_bdd": ["g_config"],
        "g_coder": ["g_add", "g_bdd"],
    }

    print("\nDependency Graph:")
    print("  g_config → g_add  ↘")
    print("          → g_bdd  → g_coder")

    # Case 1: Root changed
    print("\n--- Case 1: changed_aps = {g_config} ---")
    invalidated = invalidated_action_pairs({"g_config"}, dependency_graph)
    print(f"  Invalidated: {sorted(invalidated)}")
    print("  (Root change → all downstream invalidated)")

    # Case 2: Middle changed
    print("\n--- Case 2: changed_aps = {g_add} ---")
    invalidated = invalidated_action_pairs({"g_add"}, dependency_graph)
    print(f"  Invalidated: {sorted(invalidated)}")
    print("  (g_config and g_bdd unchanged)")

    # Case 3: Leaf changed
    print("\n--- Case 3: changed_aps = {g_coder} ---")
    invalidated = invalidated_action_pairs({"g_coder"}, dependency_graph)
    print(f"  Invalidated: {sorted(invalidated)}")
    print("  (Leaf change → only leaf invalidated)")


# =============================================================================
# DEMO 5: EXECUTION SCENARIOS (Section 8 from Extension doc)
# =============================================================================


def demo_execution_scenarios() -> None:
    """
    Demonstrate concrete execution scenarios from Extension 07 Section 8.

    Run 1: Full execution (all new)
    Run 2: No changes → all skip
    Run 3: Config change → partial re-execution
    Run 4: Artifact amendment → downstream re-execution
    """
    print("\n" + "=" * 60)
    print("EXECUTION SCENARIOS - Section 8")
    print("=" * 60)

    workflow_config = {
        "action_pairs": {
            "g_test": {"guard": "syntax"},
            "g_impl": {"guard": "dynamic_test", "requires": ["g_test"]},
            "g_review": {"guard": "human", "requires": ["g_impl"]},
        },
    }

    prompt_config = {
        "g_test": {"role": "Writer"},
        "g_impl": {"role": "Implementer"},
        "g_review": {"role": "Reviewer"},
    }

    # Start with empty repository
    repo = MockRepository(artifacts={})

    def execute_workflow(
        workflow: dict,
        prompts: dict,
        repo: MockRepository,
        run_label: str,
    ) -> dict[str, str]:
        """Simulate workflow execution with change detection."""
        print(f"\n--- {run_label} ---")
        print(f"{'Step':<12} {'Ψ_ref':<20} {'Changed?':<10} {'Action'}")
        print("-" * 50)

        artifacts: dict[str, Artifact] = {}
        refs: dict[str, str] = {}

        for ap_id in ["g_test", "g_impl", "g_review"]:
            # Get upstream artifacts for this step
            ap_config = workflow["action_pairs"][ap_id]
            upstream = {}
            for dep in ap_config.get("requires", []):
                if dep in artifacts:
                    upstream[dep] = artifacts[dep]

            # Compute current config ref
            psi_ref = compute_config_ref(ap_id, workflow, prompts, upstream or None)
            refs[ap_id] = psi_ref

            # Check if unchanged
            existing = repo.lookup(ap_id, psi_ref)
            is_changed = existing is None

            if is_changed:
                action = "Execute"
                # Create new artifact
                artifact = make_mock_artifact(
                    ap_id, f"content for {ap_id}", config_ref=psi_ref
                )
                repo.artifacts[artifact.artifact_id] = artifact
                artifacts[ap_id] = artifact
            else:
                action = "Skip"
                artifacts[ap_id] = existing

            changed_str = "yes" if is_changed else "no"
            print(f"{ap_id:<12} {psi_ref[:16]}... {changed_str:<10} {action}")

        return refs

    # Run 1: Full execution (empty repo)
    refs_v1 = execute_workflow(
        workflow_config, prompt_config, repo, "Run 1: Full Execution (empty repo)"
    )

    # Run 2: No changes (all skip)
    execute_workflow(
        workflow_config, prompt_config, repo, "Run 2: No Changes (all cached)"
    )

    # Run 3: Change g_impl prompt
    prompt_v3 = {
        "g_test": {"role": "Writer"},
        "g_impl": {"role": "Senior Implementer"},  # Changed!
        "g_review": {"role": "Reviewer"},
    }
    execute_workflow(workflow_config, prompt_v3, repo, "Run 3: g_impl Prompt Changed")

    # Run 4: Simulate human amendment to g_test artifact
    # (Create new artifact with same config but different content hash)
    print("\n--- Run 4: Human Amends g_test Artifact ---")
    print("(Same config, but human edited the artifact content)")

    # The key insight: even with same config, if upstream artifact content changes,
    # downstream Ψ_ref changes due to artifact_hash inclusion in Ψ_ref computation
    amended_test = make_mock_artifact(
        "g_test",
        "AMENDED test content by human",  # Different content!
        config_ref=refs_v1["g_test"],  # Same config_ref
    )
    repo.artifacts["art-g_test-amended"] = amended_test

    # Recompute downstream with amended artifact
    psi_impl_amended = compute_config_ref(
        "g_impl", workflow_config, prompt_config, {"g_test": amended_test}
    )
    print("  g_test content hash changed → g_impl Ψ_ref changed")
    print(f"  New Ψ_ref(g_impl): {psi_impl_amended[:16]}...")
    print("  Decision: EXECUTE g_impl (and downstream)")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run all incremental execution demonstrations."""
    print("\n" + "=" * 60)
    print("EXTENSION 07: INCREMENTAL EXECUTION")
    print("Definitions 33-37 from the formal specification")
    print("=" * 60)

    demo_config_ref()
    demo_merkle_propagation()
    demo_change_detection()
    demo_invalidation_cascade()
    demo_execution_scenarios()

    print("\n" + "=" * 60)
    print("SUCCESS: All incremental execution demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
