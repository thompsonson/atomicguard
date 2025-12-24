"""Tests for DualStateAgent - stateless executor with retry logic."""

import pytest

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.agent import DualStateAgent
from atomicguard.domain.exceptions import EscalationRequired, RmaxExhausted
from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    ContextSnapshot,
    GuardResult,
)
from atomicguard.guards import SyntaxGuard
from atomicguard.infrastructure.llm.mock import MockGenerator
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


class AlwaysPassGuard(GuardInterface):
    """Guard that always passes."""

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=True, feedback="")


class AlwaysFailGuard(GuardInterface):
    """Guard that always fails with configurable feedback."""

    def __init__(self, feedback: str = "Always fails") -> None:
        self._feedback = feedback

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=False, feedback=self._feedback)


class FailThenPassGuard(GuardInterface):
    """Guard that fails n times then passes."""

    def __init__(self, fail_count: int = 1) -> None:
        self._fail_count = fail_count
        self._call_count = 0

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            return GuardResult(passed=False, feedback=f"Fail #{self._call_count}")
        return GuardResult(passed=True, feedback="")


class FatalGuard(GuardInterface):
    """Guard that returns fatal failure."""

    def __init__(self, feedback: str = "Fatal: cannot recover") -> None:
        self._feedback = feedback

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=False, feedback=self._feedback, fatal=True)


class TestDualStateAgentInit:
    """Tests for DualStateAgent initialization."""

    def test_init_stores_action_pair(self, memory_dag: InMemoryArtifactDAG) -> None:
        """DualStateAgent stores action pair reference."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag)

        assert agent._action_pair is pair

    def test_init_default_rmax(self, memory_dag: InMemoryArtifactDAG) -> None:
        """Default rmax is 3."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag)

        assert agent._rmax == 3

    def test_init_custom_rmax(self, memory_dag: InMemoryArtifactDAG) -> None:
        """Custom rmax is stored."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag, rmax=5)

        assert agent._rmax == 5

    def test_init_with_constraints(self, memory_dag: InMemoryArtifactDAG) -> None:
        """Constraints are stored."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        agent = DualStateAgent(
            action_pair=pair, artifact_dag=memory_dag, constraints="No imports"
        )

        assert agent._constraints == "No imports"


class TestDualStateAgentExecute:
    """Tests for DualStateAgent.execute() method."""

    def test_execute_success_on_first_try(
        self, memory_dag: InMemoryArtifactDAG
    ) -> None:
        """execute() returns artifact when guard passes on first try."""
        generator = MockGenerator(responses=["def foo(): pass"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag)

        artifact = agent.execute("Write a function")

        assert artifact.content == "def foo(): pass"
        assert generator.call_count == 1

    def test_execute_retries_on_failure(self, memory_dag: InMemoryArtifactDAG) -> None:
        """execute() retries when guard fails."""
        generator = MockGenerator(
            responses=[
                "def foo( pass",  # Invalid syntax
                "def foo(): pass",  # Valid syntax
            ]
        )
        pair = ActionPair(generator=generator, guard=SyntaxGuard())
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag)

        artifact = agent.execute("Write a function")

        assert artifact.content == "def foo(): pass"
        assert generator.call_count == 2

    def test_execute_raises_rmax_exhausted(
        self, memory_dag: InMemoryArtifactDAG
    ) -> None:
        """execute() raises RmaxExhausted after max retries."""
        generator = MockGenerator(
            responses=[
                "invalid 1",
                "invalid 2",
                "invalid 3",
                "invalid 4",
                "invalid 5",
            ]
        )
        pair = ActionPair(generator=generator, guard=AlwaysFailGuard("Bad code"))
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag, rmax=2)

        with pytest.raises(RmaxExhausted) as exc_info:
            agent.execute("Write something")

        assert "Failed after 2 retries" in str(exc_info.value)
        # rmax=2 means 0, 1, 2 = 3 attempts total
        assert generator.call_count == 3

    def test_execute_stores_artifacts_in_dag(
        self, memory_dag: InMemoryArtifactDAG
    ) -> None:
        """execute() stores all artifacts in the DAG."""
        generator = MockGenerator(
            responses=[
                "bad",
                "def foo(): pass",
            ]
        )
        guard = FailThenPassGuard(fail_count=1)
        pair = ActionPair(generator=generator, guard=guard)
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag)

        agent.execute("Write a function")

        # Both artifacts should be stored
        assert len(memory_dag._artifacts) == 2

    def test_execute_with_dependencies(self, memory_dag: InMemoryArtifactDAG) -> None:
        """execute() passes dependencies through."""

        class DependencyTrackingGuard(GuardInterface):
            def __init__(self) -> None:
                self.seen_deps: list[dict[str, Artifact]] = []

            def validate(self, _artifact: Artifact, **deps: Artifact) -> GuardResult:
                self.seen_deps.append(deps)
                return GuardResult(passed=True, feedback="")

        generator = MockGenerator(responses=["x = 1"])
        guard = DependencyTrackingGuard()
        pair = ActionPair(generator=generator, guard=guard)
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag)

        dep = Artifact(
            artifact_id="dep-001",
            workflow_id="test-workflow-001",
            content="# dep",
            previous_attempt_id=None,
            action_pair_id="ap-dep",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            feedback="",
            context=ContextSnapshot(
                workflow_id="test-workflow-001",
                specification="",
                constraints="",
                feedback_history=(),
                dependency_artifacts=(),
            ),
        )

        agent.execute("Write something", dependencies={"impl": dep})

        assert len(guard.seen_deps) == 1
        assert "impl" in guard.seen_deps[0]

    def test_execute_rmax_exhausted_includes_provenance(
        self, memory_dag: InMemoryArtifactDAG
    ) -> None:
        """RmaxExhausted exception includes provenance chain."""
        generator = MockGenerator(responses=["bad1", "bad2"])
        pair = ActionPair(
            generator=generator, guard=AlwaysFailGuard("Validation failed")
        )
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag, rmax=1)

        with pytest.raises(RmaxExhausted) as exc_info:
            agent.execute("Write something")

        # rmax=1 means 2 attempts: 0 and 1
        assert len(exc_info.value.provenance) == 2
        # Each provenance entry is (artifact, feedback)
        assert exc_info.value.provenance[0][1] == "Validation failed"

    def test_execute_raises_escalation_on_fatal(
        self, memory_dag: InMemoryArtifactDAG
    ) -> None:
        """execute() raises EscalationRequired on fatal guard result."""
        generator = MockGenerator(responses=["some code"])
        pair = ActionPair(generator=generator, guard=FatalGuard())
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag)

        with pytest.raises(EscalationRequired) as exc_info:
            agent.execute("Write something")

        assert exc_info.value.feedback == "Fatal: cannot recover"
        assert exc_info.value.artifact.content == "some code"
        # Verify no retries occurred
        assert generator.call_count == 1

    def test_execute_fatal_stores_artifact_before_raising(
        self, memory_dag: InMemoryArtifactDAG
    ) -> None:
        """execute() stores artifact in DAG before raising EscalationRequired."""
        generator = MockGenerator(responses=["fatal code"])
        pair = ActionPair(generator=generator, guard=FatalGuard())
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag)

        with pytest.raises(EscalationRequired):
            agent.execute("Write something")

        # Artifact should be stored even though escalation was raised
        assert len(memory_dag._artifacts) == 1


class TestDualStateAgentContextComposition:
    """Tests for context composition methods."""

    def test_compose_context_includes_specification(
        self, memory_dag: InMemoryArtifactDAG
    ) -> None:
        """_compose_context includes specification."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag)

        context = agent._compose_context("Write a function")

        assert context.specification == "Write a function"

    def test_compose_context_includes_constraints(
        self, memory_dag: InMemoryArtifactDAG
    ) -> None:
        """_compose_context includes constraints in ambient."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        agent = DualStateAgent(
            action_pair=pair, artifact_dag=memory_dag, constraints="No imports allowed"
        )

        context = agent._compose_context("Write a function")

        assert context.ambient.constraints == "No imports allowed"

    def test_refine_context_includes_feedback_history(
        self, memory_dag: InMemoryArtifactDAG
    ) -> None:
        """_refine_context includes feedback history."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        agent = DualStateAgent(action_pair=pair, artifact_dag=memory_dag)

        artifact = Artifact(
            artifact_id="test-001",
            workflow_id="test-workflow-001",
            content="bad code",
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=None,
            feedback="",
            context=ContextSnapshot(
                workflow_id="test-workflow-001",
                specification="",
                constraints="",
                feedback_history=(),
                dependency_artifacts=(),
            ),
        )

        feedback_history = [(artifact, "Syntax error")]
        context = agent._refine_context("Write a function", artifact, feedback_history)

        assert len(context.feedback_history) == 1
        assert context.feedback_history[0][1] == "Syntax error"
