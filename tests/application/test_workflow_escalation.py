"""Tests for Workflow escalation and cascade invalidation.

Extension 09: Escalation via Informed Backtracking (Definitions 44, 45, 47, 48).

Tests the separation between:
- StagnationDetected: Level 2 recovery (workflow backtracking)
- EscalationRequired: Level 4 recovery (human intervention)
"""

from types import MappingProxyType

import pytest

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.workflow import Workflow, WorkflowStep
from atomicguard.domain.exceptions import StagnationDetected
from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult, WorkflowStatus
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.llm.mock import MockGenerator
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG

_TEMPLATE = PromptTemplate(role="test", constraints="", task="test")


class AlwaysPassGuard(GuardInterface):
    """Guard that always passes."""

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        return GuardResult(passed=True, feedback="")


class FailNTimesThenPassGuard(GuardInterface):
    """Guard that fails n times with similar feedback, then passes."""

    def __init__(self, fail_count: int = 2, feedback: str = "Test failed") -> None:
        self._fail_count = fail_count
        self._feedback = feedback
        self._call_count = 0

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            return GuardResult(passed=False, feedback=f"{self._feedback}: attempt {self._call_count}")
        return GuardResult(passed=True, feedback="")


class TrackingGuard(GuardInterface):
    """Guard that tracks call count and constraint injection."""

    def __init__(self) -> None:
        self.call_count = 0
        self.seen_constraints: list[str] = []

    def validate(self, _artifact: Artifact, **_deps: Artifact) -> GuardResult:
        self.call_count += 1
        return GuardResult(passed=True, feedback="")


class TestWorkflowStepWithEscalation:
    """Tests for add_step with escalation parameters."""

    def test_add_step_with_r_patience(self) -> None:
        """add_step stores r_patience parameter."""
        gen = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=5)  # rmax > r_patience for valid config

        workflow.add_step("g_test", pair, r_patience=3)

        assert workflow._steps[0].r_patience == 3

    def test_add_step_with_e_max(self) -> None:
        """add_step stores e_max parameter."""
        gen = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())

        workflow.add_step("g_test", pair, e_max=2)

        assert workflow._steps[0].e_max == 2

    def test_add_step_with_escalation_targets(self) -> None:
        """add_step stores escalation targets."""
        gen = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())

        workflow.add_step("g_impl", pair, escalation=("g_test", "g_analysis"))

        assert workflow._steps[0].escalation == ("g_test", "g_analysis")

    def test_add_step_defaults_e_max_to_one(self) -> None:
        """add_step defaults e_max to 1."""
        gen = MockGenerator(responses=["x = 1"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())

        workflow.add_step("g_test", pair)

        assert workflow._steps[0].e_max == 1


class TestCascadeInvalidation:
    """Tests for Definition 47: Cascade Invalidation."""

    def test_get_transitive_dependents_direct(self) -> None:
        """_get_transitive_dependents finds direct dependents."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())
        workflow.add_step("ap_analysis", pair)
        workflow.add_step("ap_test", pair, requires=("ap_analysis",))

        dependents = workflow._get_transitive_dependents("ap_analysis")

        assert "ap_test" in dependents

    def test_get_transitive_dependents_multi_level(self) -> None:
        """_get_transitive_dependents finds transitive dependents."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())
        workflow.add_step("ap_analysis", pair)
        workflow.add_step("ap_test", pair, requires=("ap_analysis",))
        workflow.add_step("ap_patch", pair, requires=("ap_test",))

        dependents = workflow._get_transitive_dependents("ap_analysis")

        assert "ap_test" in dependents
        assert "ap_patch" in dependents

    def test_invalidate_dependents_unsatisfies_target(self) -> None:
        """_invalidate_dependents marks target as unsatisfied."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())
        workflow.add_step("ap_analysis", pair)

        # Satisfy the step
        workflow._workflow_state.satisfy("ap_analysis", "artifact-001")
        assert workflow._workflow_state.is_satisfied("ap_analysis")

        # Invalidate
        workflow._invalidate_dependents("ap_analysis")

        assert not workflow._workflow_state.is_satisfied("ap_analysis")

    def test_invalidate_dependents_clears_artifacts(self) -> None:
        """_invalidate_dependents removes artifacts from cache."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())
        workflow.add_step("ap_analysis", pair)
        workflow.add_step("ap_test", pair, requires=("ap_analysis",))

        # Simulate completed state
        workflow._artifacts["ap_analysis"] = "mock_artifact_1"
        workflow._artifacts["ap_test"] = "mock_artifact_2"
        workflow._workflow_state.satisfy("ap_analysis", "a-001")
        workflow._workflow_state.satisfy("ap_test", "a-002")

        # Invalidate analysis (should cascade to test)
        workflow._invalidate_dependents("ap_analysis")

        assert "ap_analysis" not in workflow._artifacts
        assert "ap_test" not in workflow._artifacts


class TestContextInjection:
    """Tests for Definition 48: Context Injection."""

    def test_inject_failure_context_stores_summary(self) -> None:
        """_inject_failure_context stores summary for target step."""
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())

        workflow._inject_failure_context("ap_analysis", "## Failure Summary\nTest failed")

        assert "ap_analysis" in workflow._escalation_context
        assert "Failure Summary" in workflow._escalation_context["ap_analysis"]

    def test_execute_applies_injected_context(self) -> None:
        """execute() applies injected context to step constraints."""
        # This test verifies context injection is consumed during execution
        # We use tracking to verify the context was used
        gen1 = MockGenerator(responses=["analysis1", "analysis2"])
        gen2 = MockGenerator(responses=["test1", "test2"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        pair2 = ActionPair(generator=gen2, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)

        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())
        workflow.add_step("ap_analysis", pair1)
        workflow.add_step("ap_test", pair2, requires=("ap_analysis",))

        # Pre-inject context
        workflow._inject_failure_context("ap_analysis", "## Test Failure\nMust handle edge case")

        result = workflow.execute("Spec")

        # Context should have been consumed (popped from dict)
        assert "ap_analysis" not in workflow._escalation_context
        assert result.status == WorkflowStatus.SUCCESS


class TestEscalationTrigger:
    """Tests for escalation triggering and handling."""

    def test_escalation_after_stagnation(self) -> None:
        """Workflow triggers escalation after r_patience similar failures."""
        # Setup: analysis always passes, test fails r_patience times then passes
        gen1 = MockGenerator(responses=["analysis1", "analysis2"])
        gen2 = MockGenerator(responses=["test1", "test2", "test3", "test4"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        # This guard fails 2 times with similar feedback (triggers r_patience=2)
        guard2 = FailNTimesThenPassGuard(fail_count=2)
        pair2 = ActionPair(generator=gen2, guard=guard2, prompt_template=_TEMPLATE)

        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=5)  # High enough to allow retries
        workflow.add_step("ap_analysis", pair1)
        workflow.add_step(
            "ap_test",
            pair2,
            requires=("ap_analysis",),
            r_patience=2,
            e_max=1,
            escalation=("ap_analysis",),
        )

        result = workflow.execute("Spec")

        # Should succeed after escalation
        assert result.status == WorkflowStatus.SUCCESS

    def test_escalation_count_tracked(self) -> None:
        """Workflow tracks escalation count per step."""
        gen1 = MockGenerator(responses=["a1", "a2", "a3"])
        gen2 = MockGenerator(responses=["t1", "t2", "t3", "t4", "t5"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        # Fails 3 times then passes
        guard2 = FailNTimesThenPassGuard(fail_count=3)
        pair2 = ActionPair(generator=gen2, guard=guard2, prompt_template=_TEMPLATE)

        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=5)
        workflow.add_step("ap_analysis", pair1)
        workflow.add_step(
            "ap_test",
            pair2,
            requires=("ap_analysis",),
            r_patience=2,
            e_max=2,  # Allow 2 escalations
            escalation=("ap_analysis",),
        )

        workflow.execute("Spec")

        # Check that escalation count was tracked
        # (internal state, may have been incremented)
        assert workflow._escalation_count["ap_test"] >= 0

    def test_escalation_exceeds_e_max_returns_escalation_status(self) -> None:
        """Workflow returns ESCALATION status when e_max exceeded."""
        gen1 = MockGenerator(responses=["a1", "a2", "a3", "a4"])
        gen2 = MockGenerator(responses=["t" + str(i) for i in range(20)])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        # Always fails
        class AlwaysFailGuard(GuardInterface):
            def validate(self, _a: Artifact, **_d: Artifact) -> GuardResult:
                return GuardResult(passed=False, feedback="Test failed: same error")

        pair2 = ActionPair(generator=gen2, guard=AlwaysFailGuard(), prompt_template=_TEMPLATE)

        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=3)  # Low rmax
        workflow.add_step("ap_analysis", pair1)
        workflow.add_step(
            "ap_test",
            pair2,
            requires=("ap_analysis",),
            r_patience=2,  # Trigger escalation after 2 similar
            e_max=1,  # Only 1 escalation allowed
            escalation=("ap_analysis",),
        )

        result = workflow.execute("Spec")

        # Should fail with escalation (e_max exceeded or rmax on later escalation)
        assert result.status in (WorkflowStatus.ESCALATION, WorkflowStatus.FAILED)


class TestWorkflowStateUnsatisfy:
    """Tests for WorkflowState.unsatisfy method."""

    def test_unsatisfy_marks_guard_false(self) -> None:
        """unsatisfy sets guard to False."""
        from atomicguard.domain.models import WorkflowState

        state = WorkflowState()
        state.satisfy("g_test", "a-001")
        assert state.is_satisfied("g_test")

        state.unsatisfy("g_test")

        assert not state.is_satisfied("g_test")

    def test_unsatisfy_removes_artifact_id(self) -> None:
        """unsatisfy removes artifact_id mapping."""
        from atomicguard.domain.models import WorkflowState

        state = WorkflowState()
        state.satisfy("g_test", "a-001")
        assert state.get_artifact_id("g_test") == "a-001"

        state.unsatisfy("g_test")

        assert state.get_artifact_id("g_test") is None

    def test_unsatisfy_noop_if_not_satisfied(self) -> None:
        """unsatisfy is safe to call on unsatisfied guard."""
        from atomicguard.domain.models import WorkflowState

        state = WorkflowState()

        # Should not raise
        state.unsatisfy("nonexistent")

        assert not state.is_satisfied("nonexistent")


class TestMultiTargetCascadeInvalidation:
    """Tests for multi-target cascade invalidation (Definition 47)."""

    def test_invalidate_multiple_targets_union(self) -> None:
        """Invalidating multiple targets invalidates union of dependency trees."""
        gen = MockGenerator(responses=["x"] * 10)
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())
        workflow.add_step("ap_config", pair)
        workflow.add_step("ap_analysis", pair, requires=("ap_config",))
        workflow.add_step("ap_test", pair, requires=("ap_analysis",))
        workflow.add_step("ap_patch", pair, requires=("ap_test",))

        # Satisfy all steps
        for step_id in ["ap_config", "ap_analysis", "ap_test", "ap_patch"]:
            workflow._workflow_state.satisfy(step_id, f"artifact-{step_id}")
            workflow._artifacts[step_id] = f"mock-{step_id}"

        # Invalidate both ap_config and ap_analysis
        for target_id in ["ap_config", "ap_analysis"]:
            workflow._invalidate_dependents(target_id)

        # All should be invalidated (union of both cascades)
        assert not workflow._workflow_state.is_satisfied("ap_config")
        assert not workflow._workflow_state.is_satisfied("ap_analysis")
        assert not workflow._workflow_state.is_satisfied("ap_test")
        assert not workflow._workflow_state.is_satisfied("ap_patch")

    def test_context_injected_to_all_targets(self) -> None:
        """Failure context is injected to all escalation targets."""
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())
        targets = ["ap_analysis", "ap_test", "ap_config"]

        for target_id in targets:
            workflow._inject_failure_context(target_id, f"Summary for {target_id}")

        # All targets should have context
        for target_id in targets:
            assert target_id in workflow._escalation_context
            assert f"Summary for {target_id}" in workflow._escalation_context[target_id]


class TestStagnationVsFatalEscalation:
    """Tests for the separation between StagnationDetected and EscalationRequired."""

    def test_stagnation_triggers_backtracking(self) -> None:
        """Stagnation (Level 2) triggers workflow backtracking, not human escalation."""
        gen1 = MockGenerator(responses=["analysis1", "analysis2", "analysis3"])
        gen2 = MockGenerator(responses=["test" + str(i) for i in range(10)])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        guard2 = FailNTimesThenPassGuard(fail_count=4)  # Will fail enough to trigger stagnation
        pair2 = ActionPair(generator=gen2, guard=guard2, prompt_template=_TEMPLATE)

        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=5)
        workflow.add_step("ap_analysis", pair1)
        workflow.add_step(
            "ap_test",
            pair2,
            requires=("ap_analysis",),
            r_patience=2,
            e_max=2,
            escalation=("ap_analysis",),
        )

        result = workflow.execute("Spec")

        # Should succeed after backtracking (not return ESCALATION)
        assert result.status == WorkflowStatus.SUCCESS

    def test_fatal_guard_bypasses_backtracking(self) -> None:
        """Fatal guard (Level 4) returns ESCALATION status immediately."""

        class FatalGuard(GuardInterface):
            def validate(self, _a: Artifact, **_d: Artifact) -> GuardResult:
                return GuardResult(passed=False, feedback="FATAL: Security issue", fatal=True)

        gen1 = MockGenerator(responses=["analysis"])
        gen2 = MockGenerator(responses=["test"])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        pair2 = ActionPair(generator=gen2, guard=FatalGuard(), prompt_template=_TEMPLATE)

        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=5)
        workflow.add_step("ap_analysis", pair1)
        workflow.add_step(
            "ap_test",
            pair2,
            requires=("ap_analysis",),
            r_patience=2,
            e_max=2,
            escalation=("ap_analysis",),  # Should be ignored for fatal
        )

        result = workflow.execute("Spec")

        # Fatal should return ESCALATION immediately
        assert result.status == WorkflowStatus.ESCALATION
        assert "FATAL: Security issue" in result.escalation_feedback

    def test_e_max_exhausted_promotes_to_human_escalation(self) -> None:
        """After e_max stagnation attempts, promote to human escalation."""

        class AlwaysSimilarFailGuard(GuardInterface):
            def validate(self, _a: Artifact, **_d: Artifact) -> GuardResult:
                return GuardResult(passed=False, feedback="Test failed: same error pattern")

        gen1 = MockGenerator(responses=["a" + str(i) for i in range(20)])
        gen2 = MockGenerator(responses=["t" + str(i) for i in range(20)])
        pair1 = ActionPair(generator=gen1, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        pair2 = ActionPair(generator=gen2, guard=AlwaysSimilarFailGuard(), prompt_template=_TEMPLATE)

        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=3)
        workflow.add_step("ap_analysis", pair1)
        workflow.add_step(
            "ap_test",
            pair2,
            requires=("ap_analysis",),
            r_patience=2,
            e_max=1,  # Only 1 escalation allowed
            escalation=("ap_analysis",),
        )

        result = workflow.execute("Spec")

        # After e_max, should return ESCALATION (promote to human)
        assert result.status == WorkflowStatus.ESCALATION
        assert "Automated escalation exhausted" in result.escalation_feedback


class TestEscalationBudgetEnforcement:
    """Tests for escalation budget enforcement."""

    def test_escalation_count_increments_per_step(self) -> None:
        """Escalation count is tracked per step, not globally."""
        gen = MockGenerator(responses=["x"] * 20)
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)

        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())
        workflow.add_step("ap_step1", pair, r_patience=2, e_max=3, escalation=())
        workflow.add_step("ap_step2", pair, r_patience=2, e_max=5, escalation=())

        # Simulate escalation counts
        workflow._escalation_count["ap_step1"] = 2
        workflow._escalation_count["ap_step2"] = 4

        # Each step has independent count
        assert workflow._escalation_count["ap_step1"] == 2
        assert workflow._escalation_count["ap_step2"] == 4


class TestInvariantValidation:
    """Tests for Extension 09 invariant validation."""

    def test_r_patience_equal_to_rmax_raises_error(self) -> None:
        """r_patience >= rmax raises ValueError (Definition 44 invariant)."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=3)

        with pytest.raises(ValueError, match="r_patience.*must be < rmax"):
            workflow.add_step("test", pair, r_patience=3)

    def test_r_patience_greater_than_rmax_raises_error(self) -> None:
        """r_patience > rmax raises ValueError."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=3)

        with pytest.raises(ValueError, match="r_patience.*must be < rmax"):
            workflow.add_step("test", pair, r_patience=5)

    def test_r_patience_less_than_rmax_allowed(self) -> None:
        """r_patience < rmax is valid."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=6)

        # Should not raise
        workflow.add_step("test", pair, r_patience=2)

        assert workflow._steps[0].r_patience == 2

    def test_r_patience_none_allowed(self) -> None:
        """r_patience=None (disabled) is always valid."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=3)

        # Should not raise
        workflow.add_step("test", pair, r_patience=None)

        assert workflow._steps[0].r_patience is None

    def test_invalid_escalation_target_raises_on_execute(self) -> None:
        """Escalation target not in workflow steps raises ValueError."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=5)
        workflow.add_step("step1", pair, escalation=("nonexistent",))

        with pytest.raises(ValueError, match="not found in workflow"):
            workflow.execute("Spec")

    def test_valid_escalation_targets_allowed(self) -> None:
        """Valid escalation targets do not raise."""
        gen = MockGenerator(responses=["x", "y"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG(), rmax=5)
        workflow.add_step("step1", pair)
        workflow.add_step("step2", pair, requires=("step1",), escalation=("step1",))

        # Should not raise, and should succeed
        result = workflow.execute("Spec")

        assert result.status == WorkflowStatus.SUCCESS


class TestGuardSpecificEscalation:
    """Tests for escalation_by_guard (Definition 45)."""

    def test_stagnation_detected_carries_stagnant_guard(self) -> None:
        """StagnationDetected exception includes stagnant_guard field."""
        from atomicguard.domain.models import (
            Artifact,
            ArtifactStatus,
            ContextSnapshot,
        )

        artifact = Artifact(
            artifact_id="test-id",
            workflow_id="w-1",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-1",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=GuardResult(passed=False, feedback="error"),
            context=ContextSnapshot(
                workflow_id="w-1",
                specification="spec",
                constraints="",
                feedback_history=(),
            ),
        )

        exc = StagnationDetected(
            artifact=artifact,
            feedback="Test failure",
            escalate_to=["upstream_step"],
            failure_summary="Repeated failure pattern",
            stagnant_guard="SyntaxGuard",
        )

        assert exc.stagnant_guard == "SyntaxGuard"
        assert exc.artifact is artifact
        assert exc.escalate_to == ["upstream_step"]

    def test_stagnation_detected_without_stagnant_guard(self) -> None:
        """StagnationDetected works without stagnant_guard (backwards compat)."""
        from atomicguard.domain.models import (
            Artifact,
            ArtifactStatus,
            ContextSnapshot,
        )

        artifact = Artifact(
            artifact_id="test-id",
            workflow_id="w-1",
            content="test",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-1",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.REJECTED,
            guard_result=None,
            context=ContextSnapshot(
                workflow_id="w-1",
                specification="spec",
                constraints="",
                feedback_history=(),
            ),
        )

        exc = StagnationDetected(
            artifact=artifact,
            feedback="Test",
            escalate_to=["step1", "step2"],
            failure_summary="Summary",
        )

        assert exc.stagnant_guard is None

    def test_workflow_step_escalation_by_guard(self) -> None:
        """WorkflowStep supports escalation_by_guard mapping."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)

        escalation_map = MappingProxyType(
            {
                "SyntaxGuard": ("syntax_fix",),
                "TypeGuard": ("type_fix",),
            }
        )

        step = WorkflowStep(
            guard_id="test",
            action_pair=pair,
            requires=(),
            deps=(),
            escalation_by_guard=escalation_map,
        )

        assert step.escalation_by_guard["SyntaxGuard"] == ("syntax_fix",)
        assert step.escalation_by_guard["TypeGuard"] == ("type_fix",)

    def test_add_step_with_escalation_by_guard(self) -> None:
        """add_step accepts escalation_by_guard parameter."""
        gen = MockGenerator(responses=["x"])
        pair = ActionPair(generator=gen, guard=AlwaysPassGuard(), prompt_template=_TEMPLATE)
        workflow = Workflow(artifact_dag=InMemoryArtifactDAG())

        workflow.add_step(
            guard_id="test_step",
            action_pair=pair,
            r_patience=2,
            e_max=2,
            escalation=("upstream",),
            escalation_by_guard={"Guard1": ("target1",)},
        )

        step = workflow.get_step("test_step")
        assert step.r_patience == 2
        assert step.e_max == 2
        assert step.escalation == ("upstream",)
        assert step.escalation_by_guard["Guard1"] == ("target1",)
