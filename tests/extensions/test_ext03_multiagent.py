"""
Extension 03: Multi-Agent Tests.

TDD tests for implementing:
- MAS initialization (Def 19)
- Agent-local state sigma_i (Def 20)
- Cross-workflow artifact sharing via extraction
- Concurrency guarantees
- Theorem 6: Belief Convergence
- Theorem 7: System Dynamics Preservation
- Theorem 8: Cross-Workflow Dependency Resolution
"""

from atomicguard.domain.models import Artifact, ArtifactStatus


class TestMultiAgentSystemInit:
    """Tests for MAS initialization (Def 19)."""

    def test_mas_requires_shared_repository(self, memory_dag):
        """MAS requires ArtifactDAGInterface."""
        from atomicguard.domain.multiagent import MultiAgentSystem

        # Should accept ArtifactDAGInterface
        mas = MultiAgentSystem(repository=memory_dag)
        assert mas.repository is memory_dag

    def test_mas_requires_guard_library(self, memory_dag):
        """MAS requires dict of guards."""
        from atomicguard.domain.multiagent import MultiAgentSystem
        from atomicguard.guards.static.syntax import SyntaxGuard

        guards = {"syntax": SyntaxGuard()}
        mas = MultiAgentSystem(repository=memory_dag, guards=guards)
        assert "syntax" in mas.guards

    def test_mas_accepts_agent_set(self, memory_dag):
        """MAS accepts collection of Workflow instances."""
        from atomicguard.domain.multiagent import MultiAgentSystem

        # Create mock workflows
        workflows = [
            {"id": "agent-1", "steps": []},
            {"id": "agent-2", "steps": []},
        ]

        mas = MultiAgentSystem(repository=memory_dag, agents=workflows)
        assert len(mas.agents) == 2


class TestAgentLocalState:
    """Tests for agent-local state sigma_i (Def 20)."""

    def test_agent_state_derived_from_repository(
        self, memory_dag, sample_context_snapshot
    ):
        """Agent's sigma_i is derived from R, not stored separately."""
        from atomicguard.domain.multiagent import AgentState, MultiAgentSystem

        # Store an artifact
        artifact = Artifact(
            artifact_id="art-001",
            workflow_id="agent-1",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_test",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )
        memory_dag.store(artifact)

        mas = MultiAgentSystem(repository=memory_dag)
        state = mas.get_agent_state("agent-1")

        # State should be derived from repository
        assert isinstance(state, AgentState)
        assert state.is_step_complete("g_test") is True

    def test_agent_state_reflects_own_artifacts(
        self, memory_dag, sample_context_snapshot
    ):
        """sigma_i(g) = T when agent's artifact accepted for g."""
        from atomicguard.domain.multiagent import MultiAgentSystem

        # Store accepted artifact for agent-1
        artifact = Artifact(
            artifact_id="art-001",
            workflow_id="agent-1",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_impl",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )
        memory_dag.store(artifact)

        mas = MultiAgentSystem(repository=memory_dag)
        state = mas.get_agent_state("agent-1")

        # sigma_i(g_impl) should be True
        assert state.is_step_complete("g_impl") is True
        # Other steps should be False
        assert state.is_step_complete("g_test") is False

    def test_agent_state_independent(self, memory_dag, sample_context_snapshot):
        """Different agents have independent sigma_i."""
        from atomicguard.domain.multiagent import MultiAgentSystem

        # Store artifact for agent-1
        artifact1 = Artifact(
            artifact_id="art-001",
            workflow_id="agent-1",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_impl",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )
        memory_dag.store(artifact1)

        mas = MultiAgentSystem(repository=memory_dag)

        state1 = mas.get_agent_state("agent-1")
        state2 = mas.get_agent_state("agent-2")

        # agent-1 has g_impl complete
        assert state1.is_step_complete("g_impl") is True
        # agent-2 does not
        assert state2.is_step_complete("g_impl") is False


class TestCrossWorkflowDependencies:
    """Tests for cross-workflow artifact sharing."""

    def test_agent_can_extract_other_workflow_artifacts(
        self, memory_dag, sample_context_snapshot
    ):
        """Agent A can read artifacts from Agent B's workflow."""
        from atomicguard.domain.extraction import WorkflowPredicate, extract
        from atomicguard.domain.multiagent import MultiAgentSystem

        # Store artifact for agent-1
        artifact = Artifact(
            artifact_id="art-001",
            workflow_id="agent-1",
            content="shared result",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_shared",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )
        memory_dag.store(artifact)

        mas = MultiAgentSystem(repository=memory_dag)

        # agent-2 extracts from agent-1's workflow
        predicate = WorkflowPredicate("agent-1")
        results = extract(mas.repository, predicate)

        assert len(results) == 1
        assert results[0].workflow_id == "agent-1"

    def test_extraction_respects_predicates(self, memory_dag, sample_context_snapshot):
        """Cross-workflow extraction uses filter predicates."""
        from atomicguard.domain.extraction import (
            AndPredicate,
            StatusPredicate,
            WorkflowPredicate,
            extract,
        )
        from atomicguard.domain.multiagent import MultiAgentSystem

        # Store multiple artifacts
        for i, (wf, status) in enumerate(
            [
                ("agent-1", ArtifactStatus.ACCEPTED),
                ("agent-1", ArtifactStatus.REJECTED),
                ("agent-2", ArtifactStatus.ACCEPTED),
            ]
        ):
            artifact = Artifact(
                artifact_id=f"art-{i:03d}",
                workflow_id=wf,
                content=f"content-{i}",
                previous_attempt_id=None,
                parent_action_pair_id=None,
                action_pair_id="g_test",
                created_at=f"2025-01-0{i + 1}T00:00:00Z",
                attempt_number=1,
                status=status,
                guard_result=None,
                context=sample_context_snapshot,
            )
            memory_dag.store(artifact)

        mas = MultiAgentSystem(repository=memory_dag)

        # Extract only ACCEPTED from agent-1
        predicate = AndPredicate(
            WorkflowPredicate("agent-1"), StatusPredicate(ArtifactStatus.ACCEPTED)
        )
        results = extract(mas.repository, predicate)

        assert len(results) == 1
        assert results[0].workflow_id == "agent-1"
        assert results[0].status == ArtifactStatus.ACCEPTED


class TestConcurrencyGuarantees:
    """Tests for concurrent access safety."""

    def test_concurrent_writes_atomic(self, tmp_path):
        """Concurrent artifact writes don't corrupt DAG."""
        import threading

        from atomicguard.domain.models import ContextSnapshot
        from atomicguard.infrastructure.persistence.filesystem import (
            FilesystemArtifactDAG,
        )

        dag = FilesystemArtifactDAG(str(tmp_path / "dag"))

        context = ContextSnapshot(
            workflow_id="test",
            specification="test",
            constraints="",
            feedback_history=(),
            dependency_artifacts=(),
        )

        errors = []

        def write_artifact(n):
            try:
                artifact = Artifact(
                    artifact_id=f"art-{n:03d}",
                    workflow_id="concurrent",
                    content=f"content-{n}",
                    previous_attempt_id=None,
                    parent_action_pair_id=None,
                    action_pair_id="g_test",
                    created_at=f"2025-01-01T00:{n:02d}:00Z",
                    attempt_number=1,
                    status=ArtifactStatus.PENDING,
                    guard_result=None,
                    context=context,
                )
                dag.store(artifact)
            except Exception as e:
                errors.append(e)

        # Launch concurrent writes
        threads = [
            threading.Thread(target=write_artifact, args=(i,)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0

        # All artifacts should be stored
        for i in range(10):
            artifact = dag.get_artifact(f"art-{i:03d}")
            assert artifact is not None

    def test_monotonic_reads(self, memory_dag, sample_context_snapshot):
        """Once artifact visible, always visible."""
        # Store artifact
        artifact = Artifact(
            artifact_id="mono-001",
            workflow_id="test",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_test",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )
        memory_dag.store(artifact)

        # Multiple reads should always succeed
        for _ in range(100):
            result = memory_dag.get_artifact("mono-001")
            assert result is not None

    def test_read_after_write_consistency(self, memory_dag, sample_context_snapshot):
        """Artifact written at t0 visible at t1 > t0."""
        # Write
        artifact = Artifact(
            artifact_id="raw-001",
            workflow_id="test",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_test",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )
        memory_dag.store(artifact)

        # Immediately readable
        result = memory_dag.get_artifact("raw-001")
        assert result is not None
        assert result.content == "test"


class TestTheorem6BeliefConvergence:
    """Tests for Theorem 6: Belief Convergence (Shared Truth via Guards)."""

    def test_same_guard_same_artifact_same_verdict(self, sample_context_snapshot):
        """G(r) evaluated by Ag1 == G(r) evaluated by Ag2."""
        from atomicguard.guards.static.syntax import SyntaxGuard

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="any",
            content="def foo(): return 42",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_test",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=sample_context_snapshot,
        )

        guard = SyntaxGuard()

        # Same guard, same artifact = same verdict
        result1 = guard.validate(artifact)
        result2 = guard.validate(artifact)

        assert result1.passed == result2.passed
        assert result1.feedback == result2.feedback

    def test_guard_determinism_across_agents(self, memory_dag, sample_context_snapshot):
        """Two agents running same guard on same item get identical result."""
        from atomicguard.domain.multiagent import MultiAgentSystem
        from atomicguard.guards.static.syntax import SyntaxGuard

        guard = SyntaxGuard()
        mas = MultiAgentSystem(repository=memory_dag, guards={"syntax": guard})

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="shared",
            content="def foo(): return 42",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_test",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=sample_context_snapshot,
        )
        memory_dag.store(artifact)

        # Two agents evaluate same guard
        result1 = mas.evaluate_guard("agent-1", "syntax", "test-001")
        result2 = mas.evaluate_guard("agent-2", "syntax", "test-001")

        assert result1.passed == result2.passed

    def test_belief_derivable_from_repository(
        self, memory_dag, sample_context_snapshot
    ):
        """Agent beliefs are functions of repository state, not agent state."""
        from atomicguard.domain.multiagent import MultiAgentSystem

        mas = MultiAgentSystem(repository=memory_dag)

        # Store artifact
        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="agent-1",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_complete",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )
        memory_dag.store(artifact)

        # Get agent state multiple times - should be consistent
        state1 = mas.get_agent_state("agent-1")
        state2 = mas.get_agent_state("agent-1")

        assert state1.is_step_complete("g_complete") == state2.is_step_complete(
            "g_complete"
        )

    def test_no_hidden_agent_state_affects_guards(self, sample_context_snapshot):
        """Guards cannot access agent-private state."""
        from atomicguard.guards.static.syntax import SyntaxGuard

        guard = SyntaxGuard()

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="any",
            content="def foo(): return 42",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_test",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=sample_context_snapshot,
        )

        # Guard signature only accepts artifact and dependencies
        # No agent_id or agent_state parameter
        result = guard.validate(artifact)

        assert result is not None


class TestTheorem7SystemDynamicsPreservation:
    """Tests for Theorem 7: System Dynamics Preservation in MAS."""

    def test_each_agent_follows_definition7(self, memory_dag):
        """Each agent executes standard single-agent dynamics."""
        from atomicguard.domain.multiagent import MultiAgentSystem

        mas = MultiAgentSystem(repository=memory_dag)

        # Define agent workflows
        agent1_workflow = {
            "id": "agent-1",
            "steps": [
                {"id": "g_test", "guard": "TestGuard"},
                {"id": "g_impl", "guard": "SyntaxGuard"},
            ],
        }

        mas.register_agent(agent1_workflow)

        # Agent follows standard dynamics: precondition -> generate -> guard
        # This is tested by verifying the agent runner exists and follows protocol
        runner = mas.get_agent_runner("agent-1")
        assert runner is not None
        assert hasattr(runner, "execute_step")

    def test_mas_does_not_modify_core_dynamics(self, memory_dag):
        """Multi-agent extension is additive, not modifying Definition 7."""
        from atomicguard.domain.multiagent import MultiAgentSystem

        mas = MultiAgentSystem(repository=memory_dag)

        # Core dynamics (Definition 7) remain unchanged
        # MAS only adds coordination layer on top
        assert hasattr(mas, "repository")  # Shared repository
        assert hasattr(mas, "guards")  # Shared guards

        # Individual agent execution uses same interfaces
        # (GeneratorInterface, GuardInterface, ArtifactDAGInterface)

    def test_agent_isolation_during_execution(self, memory_dag):
        """Agent execution is independent until repository interaction."""
        from atomicguard.domain.multiagent import MultiAgentSystem

        mas = MultiAgentSystem(repository=memory_dag)

        # Two agents can execute independently
        # Only interact through repository
        agent1_context = mas.create_agent_context("agent-1")
        agent2_context = mas.create_agent_context("agent-2")

        # Contexts are independent
        assert agent1_context.workflow_id != agent2_context.workflow_id

    def test_coordination_via_repository_only(self, memory_dag):
        """Agents communicate only through shared R, no direct messages."""
        from atomicguard.domain.multiagent import MultiAgentSystem

        mas = MultiAgentSystem(repository=memory_dag)

        # No direct communication methods exist
        assert not hasattr(mas, "send_message")
        assert not hasattr(mas, "receive_message")

        # Communication happens through repository artifacts
        assert hasattr(mas, "repository")


class TestTheorem8CrossWorkflowDependencies:
    """Tests for Theorem 8: Cross-Workflow Dependency Resolution."""

    def test_consumer_extracts_producer_artifacts(
        self, memory_dag, sample_context_snapshot
    ):
        """Agj can consume items from Agi via extraction (Def 17)."""
        from atomicguard.domain.extraction import WorkflowPredicate, extract
        from atomicguard.domain.multiagent import MultiAgentSystem

        mas = MultiAgentSystem(repository=memory_dag)

        # Agi (agent-1) produces artifact
        artifact = Artifact(
            artifact_id="prod-001",
            workflow_id="agent-1",
            content="produced by agent-1",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_output",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )
        memory_dag.store(artifact)

        # Agj (agent-2) extracts via predicate
        predicate = WorkflowPredicate("agent-1")
        results = extract(mas.repository, predicate)

        assert len(results) == 1
        assert results[0].workflow_id == "agent-1"

    def test_no_direct_communication_required(
        self, memory_dag, sample_context_snapshot
    ):
        """Cross-workflow access happens through R, not messages."""
        from atomicguard.domain.multiagent import MultiAgentSystem

        mas = MultiAgentSystem(repository=memory_dag)

        # Store artifact from agent-1
        artifact = Artifact(
            artifact_id="shared-001",
            workflow_id="agent-1",
            content="shared",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="g_shared",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=sample_context_snapshot,
        )
        memory_dag.store(artifact)

        # agent-2 accesses via repository, not direct call
        result = mas.repository.get_artifact("shared-001")
        assert result is not None

    def test_extraction_uses_workflow_predicate(
        self, memory_dag, sample_context_snapshot
    ):
        """Consumer uses WorkflowPredicate to filter by source workflow."""
        from atomicguard.domain.extraction import WorkflowPredicate, extract

        # Store artifacts from multiple workflows
        for wf in ["agent-1", "agent-2", "agent-3"]:
            artifact = Artifact(
                artifact_id=f"art-{wf}",
                workflow_id=wf,
                content=f"from {wf}",
                previous_attempt_id=None,
                parent_action_pair_id=None,
                action_pair_id="g_test",
                created_at="2025-01-01T00:00:00Z",
                attempt_number=1,
                status=ArtifactStatus.ACCEPTED,
                guard_result=None,
                context=sample_context_snapshot,
            )
            memory_dag.store(artifact)

        # Extract only from agent-2
        predicate = WorkflowPredicate("agent-2")
        results = extract(memory_dag, predicate)

        assert len(results) == 1
        assert results[0].workflow_id == "agent-2"
