#!/usr/bin/env python3
"""
Extension 01: Versioned Environment (Definitions 10-16).

Demonstrates the core versioned environment features:
- compute_workflow_ref(): Content-addressed workflow hashing (W_ref)
- resolve_workflow_ref(): Retrieve workflow definition by hash
- Context.amend(): Monotonic configuration amendment (⊕ operator)
- WorkflowRegistry: Singleton for storing/resolving workflow definitions

These features enable:
- Workflow integrity verification on resume
- Human-in-the-loop with specification amendments
- Content-addressed workflow references for reproducibility

Run: python -m examples.basics.05_versioned_env
"""

from atomicguard.domain.models import AmbientEnvironment, Context
from atomicguard.domain.workflow import (
    WorkflowRegistry,
    compute_workflow_ref,
    resolve_workflow_ref,
)


def demo_workflow_ref() -> None:
    """
    Demonstrate content-addressed workflow hashing (Definition 11).

    W_ref = SHA256(canonical_json(workflow))

    Key properties:
    - Deterministic: Same workflow → same hash
    - Content-addressed: Hash changes if workflow changes
    - Resolvable: Can retrieve original workflow from hash
    """
    print("=" * 60)
    print("WORKFLOW REFERENCE (W_ref) - Content-Addressed Hashing")
    print("=" * 60)

    # Define a TDD workflow
    workflow = {
        "id": "tdd-workflow",
        "version": "1.0",
        "steps": [
            {"id": "g_test", "guard": "SyntaxGuard", "deps": []},
            {"id": "g_impl", "guard": "TestGuard", "deps": ["g_test"]},
            {"id": "g_review", "guard": "QualityGuard", "deps": ["g_impl"]},
        ],
    }

    # Compute deterministic hash
    w_ref = compute_workflow_ref(workflow)
    print("\nWorkflow Definition:")
    print(f"  ID: {workflow['id']}")
    print(f"  Steps: {[s['id'] for s in workflow['steps']]}")
    print(f"\nW_ref (SHA256): {w_ref[:32]}...")

    # Resolve back to original workflow
    resolved = resolve_workflow_ref(w_ref)
    print(f"\nResolved workflow matches original: {resolved == workflow}")

    # Same workflow always produces same hash (deterministic)
    w_ref_2 = compute_workflow_ref(workflow, store=False)
    print(f"Same hash on recompute: {w_ref == w_ref_2}")

    # Different workflow produces different hash
    modified_workflow = {
        "id": "tdd-workflow",
        "version": "1.1",  # Changed version
        "steps": workflow["steps"],
    }
    w_ref_modified = compute_workflow_ref(modified_workflow, store=False)
    print(f"Different hash for modified workflow: {w_ref != w_ref_modified}")


def demo_workflow_registry() -> None:
    """
    Demonstrate WorkflowRegistry singleton pattern.

    The registry stores workflow definitions indexed by W_ref,
    enabling resolve_workflow_ref() to retrieve them.
    """
    print("\n" + "=" * 60)
    print("WORKFLOW REGISTRY - Store and Resolve")
    print("=" * 60)

    # Clear registry for clean demo
    registry = WorkflowRegistry()
    registry.clear()

    # Define workflows
    workflow_a = {"id": "workflow-a", "steps": [{"id": "step-1"}]}
    workflow_b = {"id": "workflow-b", "steps": [{"id": "step-2"}]}

    # Store workflows and get their refs
    ref_a = registry.store(workflow_a)
    ref_b = registry.store(workflow_b)

    print(f"\nStored workflow-a: {ref_a[:16]}...")
    print(f"Stored workflow-b: {ref_b[:16]}...")

    # Resolve workflows by ref
    resolved_a = registry.resolve(ref_a)
    resolved_b = registry.resolve(ref_b)

    print(f"\nResolved workflow-a: {resolved_a['id']}")
    print(f"Resolved workflow-b: {resolved_b['id']}")

    # Demonstrate integrity axiom: hash(resolve(W_ref)) == W_ref
    recomputed_ref = compute_workflow_ref(resolved_a, store=False)
    print(f"\nIntegrity check (hash(resolve(ref)) == ref): {recomputed_ref == ref_a}")


def demo_context_amend() -> None:
    """
    Demonstrate monotonic context amendment (Definition 12).

    The ⊕ operator appends to specification/constraints without removing.
    Original context is unchanged (immutability preserved).

    C ⊕ Δ = C' where:
    - C'.specification = C.specification + Δ.specification
    - C'.constraints = C.constraints + Δ.constraints
    """
    print("\n" + "=" * 60)
    print("CONTEXT AMENDMENT (⊕) - Monotonic Updates")
    print("=" * 60)

    # Create initial context
    ctx = Context(
        ambient=AmbientEnvironment(repository=None, constraints="Python 3.12+"),
        specification="Write a function that adds two numbers",
    )

    print("\nOriginal Context:")
    print(f"  Specification: {ctx.specification}")
    print(f"  Constraints: {ctx.ambient.constraints if ctx.ambient else 'None'}")

    # Amend with additional requirements
    amended = ctx.amend(delta_spec="\nMust handle negative numbers correctly")

    print("\nAmended Context (delta_spec added):")
    print(f"  Specification: {amended.specification}")

    # Original unchanged (immutability)
    print(f"\nOriginal specification unchanged: {ctx.specification}")
    print(f"Immutability preserved: {ctx.specification != amended.specification}")

    # Monotonic: original content preserved in amended
    print(
        f"Monotonic (original in amended): {ctx.specification in amended.specification}"
    )

    # Chain multiple amendments
    twice_amended = amended.amend(delta_spec="\nMust return an integer, not float")

    print("\nChained amendments:")
    print(
        f"  Final specification:\n    {twice_amended.specification.replace(chr(10), chr(10) + '    ')}"
    )


def demo_amendment_with_constraints() -> None:
    """
    Demonstrate amending constraints separately from specification.
    """
    print("\n" + "=" * 60)
    print("CONSTRAINT AMENDMENT")
    print("=" * 60)

    ctx = Context(
        ambient=AmbientEnvironment(repository=None, constraints="No external imports"),
        specification="Implement a sorting algorithm",
    )

    print(
        f"\nOriginal constraints: {ctx.ambient.constraints if ctx.ambient else 'None'}"
    )

    # Amend constraints
    amended = ctx.amend(delta_constraints="\nMust have O(n log n) complexity")

    print(
        f"Amended constraints: {amended.ambient.constraints if amended.ambient else 'None'}"
    )


def main() -> None:
    """Run all versioned environment demonstrations."""
    print("\n" + "=" * 60)
    print("EXTENSION 01: VERSIONED ENVIRONMENT")
    print("Definitions 10-16 from the formal specification")
    print("=" * 60)

    demo_workflow_ref()
    demo_workflow_registry()
    demo_context_amend()
    demo_amendment_with_constraints()

    print("\n" + "=" * 60)
    print("SUCCESS: All versioned environment demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
