#!/usr/bin/env python3
"""
Extension 03: Multi-Agent System (Definitions 19-20).

Demonstrates multi-agent coordination through shared repository:
- MultiAgentSystem: MAS = ⟨{Ag₁, ..., Agₙ}, ℛ, G⟩
- AgentState: σᵢ: G → {⊥, ⊤} - Agent's belief about workflow progress
- Shared repository: All agents read/write to same ℛ
- Guard determinism: G(r) same for all agents (Theorem 6)

Key invariants:
- Agents communicate only through ℛ (no direct messages)
- Agent state σᵢ is derived from ℛ, not stored separately
- Guards are deterministic: same artifact → same result

Run: python -m examples.basics.07_multiagent
"""

from atomicguard.domain.extraction import WorkflowPredicate, extract
from atomicguard.domain.models import (
    Artifact,
    ArtifactSource,
    ArtifactStatus,
    ContextSnapshot,
    GuardResult,
)
from atomicguard.domain.multiagent import MultiAgentSystem
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


def create_context_snapshot(workflow_id: str) -> ContextSnapshot:
    """Create a minimal context snapshot."""
    return ContextSnapshot(
        workflow_id=workflow_id,
        specification="Multi-agent demo",
        constraints="",
        feedback_history=(),
        dependency_artifacts=(),
    )


def create_artifact(
    artifact_id: str,
    workflow_id: str,
    action_pair_id: str,
    status: ArtifactStatus,
    content: str = "# Generated content",
) -> Artifact:
    """Create a test artifact."""
    return Artifact(
        artifact_id=artifact_id,
        workflow_id=workflow_id,
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id=action_pair_id,
        created_at="2025-01-01T10:00:00Z",
        attempt_number=1,
        status=status,
        guard_result=None,
        context=create_context_snapshot(workflow_id),
        source=ArtifactSource.GENERATED,
    )


def demo_mas_setup() -> None:
    """
    Demonstrate Multi-Agent System initialization.

    MAS = ⟨{Ag₁, ..., Agₙ}, ℛ, G⟩ where:
    - {Ag₁, ..., Agₙ}: Collection of agent workflows
    - ℛ: Shared artifact repository
    - G: Shared guard library
    """
    print("=" * 60)
    print("MULTI-AGENT SYSTEM SETUP")
    print("=" * 60)

    # Create shared repository
    repo = InMemoryArtifactDAG()

    # Define agent workflows
    agent_1_workflow = {
        "id": "agent-1",
        "role": "test-writer",
        "steps": [
            {"id": "g_spec", "guard": "SpecGuard"},
            {"id": "g_test", "guard": "SyntaxGuard"},
        ],
    }

    agent_2_workflow = {
        "id": "agent-2",
        "role": "implementer",
        "steps": [
            {"id": "g_impl", "guard": "TestGuard", "deps": ["agent-1:g_test"]},
            {"id": "g_refactor", "guard": "QualityGuard"},
        ],
    }

    # Create MAS with shared repository and agents
    mas = MultiAgentSystem(
        repository=repo,
        guards={},  # Guard library (empty for demo)
        agents=[agent_1_workflow, agent_2_workflow],
    )

    print("\nMAS initialized:")
    print(f"  Repository: {type(mas.repository).__name__}")
    print(f"  Agents: {len(mas.agents)}")
    for agent in mas.agents:
        print(f"    - {agent['id']} ({agent['role']})")

    # All agents share the same repository instance
    print(f"\nShared repository: {mas.repository is repo}")

    # Register additional agent dynamically
    agent_3_workflow = {
        "id": "agent-3",
        "role": "reviewer",
        "steps": [{"id": "g_review", "guard": "ReviewGuard"}],
    }
    mas.register_agent(agent_3_workflow)
    print(f"Registered new agent: {agent_3_workflow['id']}")
    print(f"Total agents: {len(mas.agents)}")


def demo_agent_state() -> None:
    """
    Demonstrate AgentState derived from repository (Definition 20).

    σᵢ: G → {⊥, ⊤}

    Agent state is not stored - it's always computed from the
    current state of the repository. This ensures consistency.
    """
    print("\n" + "=" * 60)
    print("AGENT STATE - Derived from Repository")
    print("=" * 60)

    repo = InMemoryArtifactDAG()
    mas = MultiAgentSystem(repository=repo)
    mas.register_agent({"id": "agent-1", "steps": []})

    # Get agent state (computed from repo, not stored)
    state = mas.get_agent_state("agent-1")

    print("\nInitial state for agent-1:")
    print(f"  g_test complete: {state.is_step_complete('g_test')}")
    print(f"  g_impl complete: {state.is_step_complete('g_impl')}")

    # Store a REJECTED artifact - step still not complete
    rejected = create_artifact("art-001", "agent-1", "g_test", ArtifactStatus.REJECTED)
    repo.store(rejected)

    print("\nAfter storing REJECTED artifact for g_test:")
    print(f"  g_test complete: {state.is_step_complete('g_test')}")

    # Store an ACCEPTED artifact - now step is complete
    accepted = create_artifact("art-002", "agent-1", "g_test", ArtifactStatus.ACCEPTED)
    repo.store(accepted)

    print("\nAfter storing ACCEPTED artifact for g_test:")
    print(f"  g_test complete: {state.is_step_complete('g_test')}")
    print(f"  g_impl complete: {state.is_step_complete('g_impl')}")

    # State reflects current repo - no need to "refresh"
    print("\nState is always current (derived, not cached)")


def demo_cross_workflow_extraction() -> None:
    """
    Demonstrate agents reading artifacts from other workflows.

    Agents can use extract() to query the shared repository
    for artifacts from other agents' workflows.
    """
    print("\n" + "=" * 60)
    print("CROSS-WORKFLOW EXTRACTION")
    print("=" * 60)

    mas = MultiAgentSystem(repository=InMemoryArtifactDAG())

    # Agent-1 produces a design artifact
    design_artifact = create_artifact(
        "design-001",
        "agent-1",
        "g_design",
        ArtifactStatus.ACCEPTED,
        content="# System Design\n\nComponent A -> Component B",
    )
    mas.repository.store(design_artifact)
    print(f"\nAgent-1 stored design artifact: {design_artifact.artifact_id}")

    # Agent-2 produces test artifacts
    test_artifact = create_artifact(
        "test-001",
        "agent-2",
        "g_test",
        ArtifactStatus.ACCEPTED,
        content="def test_add():\n    assert add(2, 3) == 5",
    )
    mas.repository.store(test_artifact)
    print(f"Agent-2 stored test artifact: {test_artifact.artifact_id}")

    # Agent-3 extracts from other agents' workflows
    print("\nAgent-3 extracts from shared repository:")

    # Extract all artifacts from agent-1
    agent_1_artifacts = extract(mas.repository, WorkflowPredicate("agent-1"))
    print(f"  From agent-1: {len(agent_1_artifacts)} artifact(s)")
    for a in agent_1_artifacts:
        print(f"    - {a.artifact_id}: {a.action_pair_id}")

    # Extract all artifacts from agent-2
    agent_2_artifacts = extract(mas.repository, WorkflowPredicate("agent-2"))
    print(f"  From agent-2: {len(agent_2_artifacts)} artifact(s)")
    for a in agent_2_artifacts:
        print(f"    - {a.artifact_id}: {a.action_pair_id}")

    # This enables cross-workflow dependencies
    print("\nCross-workflow dependencies work via extraction:")
    print("  Agent-3 can wait for Agent-1's design before implementing")


def demo_shared_repository_coordination() -> None:
    """
    Demonstrate coordination through shared repository.

    Agents don't communicate directly - they:
    1. Write artifacts to repository
    2. Read artifacts from repository via extract()
    3. Derive state from repository

    This is the "blackboard" pattern without explicit Blackboard class.
    """
    print("\n" + "=" * 60)
    print("COORDINATION VIA SHARED REPOSITORY")
    print("=" * 60)

    repo = InMemoryArtifactDAG()
    mas = MultiAgentSystem(repository=repo)
    mas.register_agent({"id": "test-writer"})
    mas.register_agent({"id": "implementer"})

    # Test-writer produces tests
    print("\nStep 1: test-writer produces tests")
    test_artifact = create_artifact(
        "test-001",
        "test-writer",
        "g_test",
        ArtifactStatus.ACCEPTED,
        content="def test_add():\n    assert add(2, 3) == 5",
    )
    repo.store(test_artifact)
    print(f"  Stored: {test_artifact.artifact_id}")

    # Implementer checks if tests are ready
    print("\nStep 2: implementer checks for tests")
    test_writer_state = mas.get_agent_state("test-writer")
    tests_ready = test_writer_state.is_step_complete("g_test")
    print(f"  Tests ready: {tests_ready}")

    # Implementer reads the tests
    if tests_ready:
        tests = extract(repo, WorkflowPredicate("test-writer"))
        print(f"  Retrieved tests: {[a.artifact_id for a in tests]}")

    # Implementer produces implementation
    print("\nStep 3: implementer produces implementation")
    impl_artifact = create_artifact(
        "impl-001",
        "implementer",
        "g_impl",
        ArtifactStatus.ACCEPTED,
        content="def add(a, b):\n    return a + b",
    )
    repo.store(impl_artifact)
    print(f"  Stored: {impl_artifact.artifact_id}")

    # Final state
    print("\nFinal repository state:")
    all_artifacts = extract(repo)
    for a in all_artifacts:
        print(
            f"  {a.artifact_id}: {a.workflow_id}/{a.action_pair_id} ({a.status.value})"
        )


def demo_guard_determinism() -> None:
    """
    Demonstrate Theorem 6: Guard Determinism (Belief Convergence).

    Guards are deterministic: G(r) produces the same result
    regardless of which agent evaluates it.

    This ensures all agents converge to the same beliefs about
    which artifacts are valid.
    """
    print("\n" + "=" * 60)
    print("THEOREM 6: GUARD DETERMINISM")
    print("=" * 60)

    # Simple deterministic guard
    class DeterministicSyntaxGuard:
        """Guard that always produces same result for same artifact."""

        @property
        def guard_id(self) -> str:
            return "syntax_guard"

        def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
            # Deterministic check based only on artifact content
            is_valid = "def " in artifact.content and ":" in artifact.content
            return GuardResult(
                passed=is_valid,
                feedback="" if is_valid else "Invalid syntax",
            )

    repo = InMemoryArtifactDAG()
    guard = DeterministicSyntaxGuard()

    # Create test artifact
    artifact = create_artifact(
        "code-001",
        "any-agent",
        "g_impl",
        ArtifactStatus.PENDING,
        content="def add(a, b):\n    return a + b",
    )
    repo.store(artifact)

    # Any agent evaluating the same artifact gets the same result
    print(f"\nArtifact: {artifact.artifact_id}")
    print(f"Content: {artifact.content[:30]}...")

    result_1 = guard.validate(artifact)
    result_2 = guard.validate(artifact)
    result_3 = guard.validate(artifact)

    print("\nGuard evaluations:")
    print(f"  Agent-1 result: passed={result_1.passed}")
    print(f"  Agent-2 result: passed={result_2.passed}")
    print(f"  Agent-3 result: passed={result_3.passed}")

    print(
        f"\nDeterministic (all same): {result_1.passed == result_2.passed == result_3.passed}"
    )
    print("\nThis ensures all agents converge to same beliefs about artifact validity")


def main() -> None:
    """Run all multi-agent system demonstrations."""
    print("\n" + "=" * 60)
    print("EXTENSION 03: MULTI-AGENT SYSTEM")
    print("Definitions 19-20 from the formal specification")
    print("=" * 60)

    demo_mas_setup()
    demo_agent_state()
    demo_cross_workflow_extraction()
    demo_shared_repository_coordination()
    demo_guard_determinism()

    print("\n" + "=" * 60)
    print("SUCCESS: All multi-agent demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
