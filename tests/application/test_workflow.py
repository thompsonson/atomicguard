"""Tests for Workflow - orchestrates ActionPair execution across steps."""

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.workflow import Workflow
from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult
from atomicguard.guards.syntax import SyntaxGuard
from atomicguard.infrastructure.llm.mock import MockGenerator
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


class AlwaysPassGuard(GuardInterface):
    """Guard that always passes."""

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=True, feedback="")


class AlwaysFailGuard(GuardInterface):
    """Guard that always fails."""

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=False, feedback="Always fails")


class TestWorkflowInit:
    """Tests for Workflow initialization."""

    def test_init_creates_default_dag(self) -> None:
        """Workflow creates InMemoryArtifactDAG if none provided."""
        workflow = Workflow()

        assert workflow._dag is not None

    def test_init_uses_provided_dag(self, memory_dag: InMemoryArtifactDAG) -> None:
        """Workflow uses provided artifact DAG."""
        workflow = Workflow(artifact_dag=memory_dag)

        assert workflow._dag is memory_dag

    def test_init_default_rmax(self) -> None:
        """Default rmax is 3."""
        workflow = Workflow()

        assert workflow._rmax == 3

    def test_init_custom_rmax(self) -> None:
        """Custom rmax is stored."""
        workflow = Workflow(rmax=5)

        assert workflow._rmax == 5

    def test_init_with_constraints(self) -> None:
        """Constraints are stored."""
        workflow = Workflow(constraints="No external imports")

        assert workflow._constraints == "No external imports"

    def test_init_empty_steps(self) -> None:
        """Workflow starts with no steps."""
        workflow = Workflow()

        assert len(workflow._steps) == 0


class TestWorkflowAddStep:
    """Tests for Workflow.add_step() method."""

    def test_add_step_appends_step(self) -> None:
        """add_step() adds step to workflow."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        workflow = Workflow()

        workflow.add_step("g_test", pair)

        assert len(workflow._steps) == 1
        assert workflow._steps[0].guard_id == "g_test"

    def test_add_step_returns_self_for_chaining(self) -> None:
        """add_step() returns self for fluent chaining."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        workflow = Workflow()

        result = workflow.add_step("g_test", pair)

        assert result is workflow

    def test_add_step_with_requires(self) -> None:
        """add_step() stores requires tuple."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        workflow = Workflow()

        workflow.add_step("g_impl", pair, requires=("g_test",))

        assert workflow._steps[0].requires == ("g_test",)

    def test_add_step_deps_defaults_to_requires(self) -> None:
        """add_step() defaults deps to requires."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        workflow = Workflow()

        workflow.add_step("g_impl", pair, requires=("g_test",))

        assert workflow._steps[0].deps == ("g_test",)

    def test_add_step_with_explicit_deps(self) -> None:
        """add_step() accepts explicit deps."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        workflow = Workflow()

        workflow.add_step("g_impl", pair, requires=("g_test",), deps=("g_other",))

        assert workflow._steps[0].deps == ("g_other",)

    def test_add_multiple_steps_chained(self) -> None:
        """Multiple steps can be chained."""
        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["y = 2"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard())
        workflow = Workflow()

        workflow.add_step("g_test", pair1).add_step(
            "g_impl", pair2, requires=("g_test",)
        )

        assert len(workflow._steps) == 2
        assert workflow._steps[0].guard_id == "g_test"
        assert workflow._steps[1].guard_id == "g_impl"


class TestWorkflowExecute:
    """Tests for Workflow.execute() method."""

    def test_execute_single_step_success(self) -> None:
        """execute() succeeds with single passing step."""
        generator = MockGenerator(responses=["def foo(): pass"])
        pair = ActionPair(generator=generator, guard=SyntaxGuard())
        workflow = Workflow()
        workflow.add_step("g_test", pair)

        result = workflow.execute("Write a function")

        assert result.success is True
        assert "g_test" in result.artifacts

    def test_execute_multiple_steps_success(self) -> None:
        """execute() succeeds with multiple passing steps."""
        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["y = 2"])
        pair1 = ActionPair(generator=gen1, guard=SyntaxGuard())
        pair2 = ActionPair(generator=gen2, guard=SyntaxGuard())
        workflow = Workflow()
        workflow.add_step("g_test", pair1).add_step(
            "g_impl", pair2, requires=("g_test",)
        )

        result = workflow.execute("Write code")

        assert result.success is True
        assert "g_test" in result.artifacts
        assert "g_impl" in result.artifacts

    def test_execute_fails_when_step_exhausts_retries(self) -> None:
        """execute() returns failure when step exhausts retries."""
        generator = MockGenerator(responses=["bad", "bad", "bad", "bad"])
        pair = ActionPair(generator=generator, guard=AlwaysFailGuard())
        workflow = Workflow(rmax=2)
        workflow.add_step("g_test", pair)

        result = workflow.execute("Write something")

        assert result.success is False
        assert result.failed_step == "g_test"

    def test_execute_respects_step_dependencies(self) -> None:
        """execute() runs steps in dependency order."""
        execution_order: list[str] = []

        class TrackingGuard(GuardInterface):
            def __init__(self, step_id: str) -> None:
                self._step_id = step_id

            def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
                execution_order.append(self._step_id)
                return GuardResult(passed=True, feedback="")

        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["y = 2"])
        pair1 = ActionPair(generator=gen1, guard=TrackingGuard("g_test"))
        pair2 = ActionPair(generator=gen2, guard=TrackingGuard("g_impl"))
        workflow = Workflow()
        workflow.add_step("g_test", pair1).add_step(
            "g_impl", pair2, requires=("g_test",)
        )

        workflow.execute("Write code")

        assert execution_order == ["g_test", "g_impl"]

    def test_execute_empty_workflow_success(self) -> None:
        """execute() succeeds immediately with no steps."""
        workflow = Workflow()

        result = workflow.execute("Do nothing")

        assert result.success is True
        assert len(result.artifacts) == 0

    def test_execute_failure_includes_provenance(self) -> None:
        """execute() includes provenance in failure result."""
        generator = MockGenerator(responses=["bad1", "bad2", "bad3"])
        pair = ActionPair(generator=generator, guard=AlwaysFailGuard())
        workflow = Workflow(rmax=2)
        workflow.add_step("g_test", pair)

        result = workflow.execute("Write something")

        assert result.success is False
        assert result.provenance is not None
        assert len(result.provenance) > 0


class TestWorkflowInternalMethods:
    """Tests for Workflow internal helper methods."""

    def test_precondition_met_no_requires(self) -> None:
        """_precondition_met returns True when no requires."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        workflow = Workflow()
        workflow.add_step("g_test", pair)

        step = workflow._steps[0]
        assert workflow._precondition_met(step) is True

    def test_precondition_met_satisfied_requires(self) -> None:
        """_precondition_met returns True when requires satisfied."""
        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["y = 2"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard())
        workflow = Workflow()
        workflow.add_step("g_test", pair1).add_step(
            "g_impl", pair2, requires=("g_test",)
        )

        # Satisfy g_test
        workflow._workflow_state.satisfy("g_test", "artifact-001")

        step = workflow._steps[1]  # g_impl
        assert workflow._precondition_met(step) is True

    def test_precondition_met_unsatisfied_requires(self) -> None:
        """_precondition_met returns False when requires not satisfied."""
        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["y = 2"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard())
        workflow = Workflow()
        workflow.add_step("g_test", pair1).add_step(
            "g_impl", pair2, requires=("g_test",)
        )

        step = workflow._steps[1]  # g_impl (g_test not satisfied)
        assert workflow._precondition_met(step) is False

    def test_find_applicable_returns_first_applicable(self) -> None:
        """_find_applicable returns first step with precondition met."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        workflow = Workflow()
        workflow.add_step("g_test", pair)

        step = workflow._find_applicable()

        assert step is not None
        assert step.guard_id == "g_test"

    def test_find_applicable_skips_satisfied_steps(self) -> None:
        """_find_applicable skips already satisfied steps."""
        gen1 = MockGenerator(responses=["x = 1"])
        gen2 = MockGenerator(responses=["y = 2"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard())
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard())
        workflow = Workflow()
        workflow.add_step("g_test", pair1).add_step(
            "g_impl", pair2, requires=("g_test",)
        )

        # Satisfy g_test
        workflow._workflow_state.satisfy("g_test", "artifact-001")

        step = workflow._find_applicable()

        assert step is not None
        assert step.guard_id == "g_impl"

    def test_find_applicable_returns_none_when_blocked(self) -> None:
        """_find_applicable returns None when no applicable step."""
        gen2 = MockGenerator(responses=["y = 2"])
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard())
        workflow = Workflow()
        # Add g_impl first (depends on g_test which doesn't exist yet)
        workflow.add_step("g_impl", pair2, requires=("g_test",))

        step = workflow._find_applicable()

        assert step is None

    def test_is_goal_state_empty_workflow(self) -> None:
        """_is_goal_state returns True for empty workflow."""
        workflow = Workflow()

        assert workflow._is_goal_state() is True

    def test_is_goal_state_all_satisfied(self) -> None:
        """_is_goal_state returns True when all steps satisfied."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        workflow = Workflow()
        workflow.add_step("g_test", pair)

        workflow._workflow_state.satisfy("g_test", "artifact-001")

        assert workflow._is_goal_state() is True

    def test_is_goal_state_not_all_satisfied(self) -> None:
        """_is_goal_state returns False when not all steps satisfied."""
        generator = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=generator, guard=AlwaysPassGuard())
        workflow = Workflow()
        workflow.add_step("g_test", pair)

        assert workflow._is_goal_state() is False
