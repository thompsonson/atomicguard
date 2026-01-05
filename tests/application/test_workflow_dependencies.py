"""Tests for workflow dependency key handling.

These tests verify that:
1. Dependencies are passed to guards with original action_pair_id as key
2. Non-standard action pair names (without g_ prefix) work correctly
3. Multiple dependencies are passed with correct keys
4. Explicit deps parameter overrides requires
"""

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.workflow import Workflow
from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult
from atomicguard.infrastructure.llm.mock import MockGenerator
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


class DependencyCapturingGuard(GuardInterface):
    """Guard that captures the dependency keys it receives."""

    def __init__(self) -> None:
        self.received_deps: dict[str, Artifact] = {}

    def validate(self, _artifact: Artifact, **deps: Artifact) -> GuardResult:
        self.received_deps = deps.copy()
        return GuardResult(passed=True, feedback="")


class AlwaysPassGuard(GuardInterface):
    """Simple guard that always passes."""

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=True, feedback="")


class TestDependencyKeyNames:
    """Tests for verifying dependency keys match action_pair_ids."""

    def test_dependency_keys_match_guard_ids_with_g_prefix(self) -> None:
        """Verify guard receives deps keyed by original action_pair_id (g_prefix)."""
        dag = InMemoryArtifactDAG()
        workflow = Workflow(artifact_dag=dag, rmax=1)

        # First step produces artifact
        gen1 = MockGenerator(responses=["# config"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())

        # Second step should receive dependency keyed by first step's guard_id
        capturing_guard = DependencyCapturingGuard()
        gen2 = MockGenerator(responses=["# impl"])
        pair2 = ActionPair(generator=gen2, guard=capturing_guard)

        workflow.add_step("g_config", pair1)
        workflow.add_step("g_impl", pair2, requires=("g_config",))

        result = workflow.execute("Test spec")

        assert result.status.name == "SUCCESS"
        # Key point: dependency key should be "g_config", NOT "config"
        assert "g_config" in capturing_guard.received_deps
        assert "config" not in capturing_guard.received_deps

    def test_dependency_keys_match_guard_ids_without_prefix(self) -> None:
        """Verify guard receives deps with non-prefixed action_pair_ids."""
        dag = InMemoryArtifactDAG()
        workflow = Workflow(artifact_dag=dag, rmax=1)

        gen1 = MockGenerator(responses=["# tests"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())

        capturing_guard = DependencyCapturingGuard()
        gen2 = MockGenerator(responses=["# impl"])
        pair2 = ActionPair(generator=gen2, guard=capturing_guard)

        # Use arbitrary names without g_ prefix
        workflow.add_step("unit_tests", pair1)
        workflow.add_step("implementation", pair2, requires=("unit_tests",))

        result = workflow.execute("Test spec")

        assert result.status.name == "SUCCESS"
        assert "unit_tests" in capturing_guard.received_deps


class TestNonStandardActionPairNames:
    """Tests for workflows with arbitrary action pair naming."""

    def test_workflow_with_custom_names_executes_successfully(self) -> None:
        """Workflow executes with non-standard action pair names."""
        dag = InMemoryArtifactDAG()
        workflow = Workflow(artifact_dag=dag, rmax=1)

        gen1 = MockGenerator(responses=["# step one"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())

        gen2 = MockGenerator(responses=["# step two"])
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard())

        gen3 = MockGenerator(responses=["# step three"])
        pair3 = ActionPair(generator=gen3, guard=AlwaysPassGuard())

        workflow.add_step("first_step", pair1)
        workflow.add_step("second_step", pair2, requires=("first_step",))
        workflow.add_step("third_step", pair3, requires=("second_step",))

        result = workflow.execute("Test spec")

        assert result.status.name == "SUCCESS"
        assert "first_step" in result.artifacts
        assert "second_step" in result.artifacts
        assert "third_step" in result.artifacts

    def test_workflow_with_mixed_naming_conventions(self) -> None:
        """Workflow works with mixed naming conventions."""
        dag = InMemoryArtifactDAG()
        workflow = Workflow(artifact_dag=dag, rmax=1)

        gen1 = MockGenerator(responses=["# g_style"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())

        gen2 = MockGenerator(responses=["# snake_case"])
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard())

        gen3 = MockGenerator(responses=["# camelCase"])
        pair3 = ActionPair(generator=gen3, guard=AlwaysPassGuard())

        workflow.add_step("g_config", pair1)
        workflow.add_step("test_generator", pair2, requires=("g_config",))
        workflow.add_step("implOutput", pair3, requires=("test_generator",))

        result = workflow.execute("Test spec")

        assert result.status.name == "SUCCESS"


class TestMultipleDependencies:
    """Tests for guards receiving multiple dependencies."""

    def test_multiple_dependencies_passed_with_correct_keys(self) -> None:
        """Guard receives all dependencies with their original keys."""
        dag = InMemoryArtifactDAG()
        workflow = Workflow(artifact_dag=dag, rmax=1)

        gen1 = MockGenerator(responses=["# config"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())

        gen2 = MockGenerator(responses=["# tests"])
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard())

        capturing_guard = DependencyCapturingGuard()
        gen3 = MockGenerator(responses=["# impl"])
        pair3 = ActionPair(generator=gen3, guard=capturing_guard)

        workflow.add_step("config", pair1)
        workflow.add_step("tests", pair2, requires=("config",))
        workflow.add_step("impl", pair3, requires=("config", "tests"))

        result = workflow.execute("Test spec")

        assert result.status.name == "SUCCESS"
        assert "config" in capturing_guard.received_deps
        assert "tests" in capturing_guard.received_deps
        assert len(capturing_guard.received_deps) == 2

    def test_dependencies_contain_correct_artifacts(self) -> None:
        """Verify dependency artifacts have correct content."""
        dag = InMemoryArtifactDAG()
        workflow = Workflow(artifact_dag=dag, rmax=1)

        gen1 = MockGenerator(responses=["CONFIG_CONTENT"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())

        gen2 = MockGenerator(responses=["TEST_CONTENT"])
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard())

        capturing_guard = DependencyCapturingGuard()
        gen3 = MockGenerator(responses=["# impl"])
        pair3 = ActionPair(generator=gen3, guard=capturing_guard)

        workflow.add_step("config", pair1)
        workflow.add_step("tests", pair2)
        workflow.add_step("impl", pair3, requires=("config", "tests"))

        result = workflow.execute("Test spec")

        assert result.status.name == "SUCCESS"
        assert capturing_guard.received_deps["config"].content == "CONFIG_CONTENT"
        assert capturing_guard.received_deps["tests"].content == "TEST_CONTENT"


class TestExplicitDepsParameter:
    """Tests for explicit deps parameter overriding requires."""

    def test_explicit_deps_overrides_requires(self) -> None:
        """Explicit deps parameter determines what's passed to guard."""
        dag = InMemoryArtifactDAG()
        workflow = Workflow(artifact_dag=dag, rmax=1)

        gen1 = MockGenerator(responses=["# config"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())

        gen2 = MockGenerator(responses=["# tests"])
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard())

        capturing_guard = DependencyCapturingGuard()
        gen3 = MockGenerator(responses=["# impl"])
        pair3 = ActionPair(generator=gen3, guard=capturing_guard)

        workflow.add_step("config", pair1)
        workflow.add_step("tests", pair2, requires=("config",))
        # requires tests, but deps only includes config
        workflow.add_step("impl", pair3, requires=("tests",), deps=("config",))

        result = workflow.execute("Test spec")

        assert result.status.name == "SUCCESS"
        # Guard should receive config, not tests
        assert "config" in capturing_guard.received_deps
        assert "tests" not in capturing_guard.received_deps

    def test_empty_deps_passes_no_dependencies(self) -> None:
        """Empty deps tuple passes no dependencies to guard."""
        dag = InMemoryArtifactDAG()
        workflow = Workflow(artifact_dag=dag, rmax=1)

        gen1 = MockGenerator(responses=["# config"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())

        capturing_guard = DependencyCapturingGuard()
        gen2 = MockGenerator(responses=["# impl"])
        pair2 = ActionPair(generator=gen2, guard=capturing_guard)

        workflow.add_step("config", pair1)
        # requires config for ordering, but deps=() passes nothing
        workflow.add_step("impl", pair2, requires=("config",), deps=())

        result = workflow.execute("Test spec")

        assert result.status.name == "SUCCESS"
        assert len(capturing_guard.received_deps) == 0
