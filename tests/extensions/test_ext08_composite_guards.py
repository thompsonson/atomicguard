"""Tests for Extension 08: Composite Guards (Definitions 38-43, Theorem 14)."""

import pytest

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult
from atomicguard.guards.composite import (
    AggregationPolicy,
    ParallelGuard,
    SequentialGuard,
    SubGuardResult,
)


class AlwaysPassGuard(GuardInterface):
    """Test guard that always passes."""

    def validate(self, _artifact: Artifact, **_deps) -> GuardResult:
        return GuardResult(passed=True, feedback="Pass")


class AlwaysFailGuard(GuardInterface):
    """Test guard that always fails."""

    def validate(self, _artifact: Artifact, **_deps) -> GuardResult:
        return GuardResult(passed=False, feedback="Fail")


class OrderTrackingGuard(GuardInterface):
    """Guard that tracks execution order."""

    execution_order: list[str] = []

    def __init__(self, name: str, should_pass: bool = True):
        self.name = name
        self.should_pass = should_pass

    def validate(self, _artifact: Artifact, **_deps) -> GuardResult:
        OrderTrackingGuard.execution_order.append(self.name)
        return GuardResult(passed=self.should_pass, feedback=f"{self.name} executed")


class TestDefinition38CompositeGuard:
    """Tests for Definition 38: Composite Guard."""

    def test_satisfies_guard_interface(self):
        """Composite guard implements GuardInterface."""
        sequential = SequentialGuard([AlwaysPassGuard()])
        parallel = ParallelGuard([AlwaysPassGuard()])

        assert isinstance(sequential, GuardInterface)
        assert isinstance(parallel, GuardInterface)

    def test_returns_verdict_and_feedback(self, sample_artifact):
        """Returns (v, φ) tuple as per Definition 4."""
        guard = SequentialGuard([AlwaysPassGuard()])
        result = guard.validate(sample_artifact)

        assert hasattr(result, "passed")
        assert hasattr(result, "feedback")
        assert isinstance(result.passed, bool)
        assert isinstance(result.feedback, str)

    def test_composite_structure(self):
        """Composite has guards, compose strategy, and policy."""
        guards = [AlwaysPassGuard(), AlwaysFailGuard()]
        sequential = SequentialGuard(guards, AggregationPolicy.ALL_PASS)

        assert len(sequential.guards) == 2
        assert sequential._policy == AggregationPolicy.ALL_PASS


class TestDefinition39SequentialGuard:
    """Tests for Definition 39: Sequential Guard."""

    def test_executes_in_order(self, sample_artifact):
        """Guards execute G₁, G₂, ..., Gₙ in order."""
        OrderTrackingGuard.execution_order = []

        guard = SequentialGuard(
            [
                OrderTrackingGuard("G1"),
                OrderTrackingGuard("G2"),
                OrderTrackingGuard("G3"),
            ]
        )
        guard.validate(sample_artifact)

        assert OrderTrackingGuard.execution_order == ["G1", "G2", "G3"]

    def test_short_circuits_on_failure(self, sample_artifact):
        """Stops on first vᵢ ≠ ⊤."""
        OrderTrackingGuard.execution_order = []

        guard = SequentialGuard(
            [
                OrderTrackingGuard("G1"),
                OrderTrackingGuard("G2", should_pass=False),
                OrderTrackingGuard("G3"),  # Should not execute
            ]
        )
        guard.validate(sample_artifact)

        assert OrderTrackingGuard.execution_order == ["G1", "G2"]
        assert "G3" not in OrderTrackingGuard.execution_order

    def test_feedback_from_failing_guard(self, sample_artifact):
        """Feedback φ comes from the guard that failed."""
        guard = SequentialGuard(
            [
                AlwaysPassGuard(),
                AlwaysFailGuard(),
            ]
        )
        result = guard.validate(sample_artifact)

        assert "AlwaysFailGuard" in result.feedback
        assert "FAIL" in result.feedback


class TestDefinition40ParallelGuard:
    """Tests for Definition 40: Parallel Guard."""

    def test_executes_all_guards(self, sample_artifact):
        """All guards run in parallel (no short-circuit)."""
        OrderTrackingGuard.execution_order = []

        guard = ParallelGuard(
            [
                OrderTrackingGuard("G1", should_pass=False),
                OrderTrackingGuard("G2"),
                OrderTrackingGuard("G3"),
            ]
        )
        guard.validate(sample_artifact)

        # All guards should have executed
        assert "G1" in OrderTrackingGuard.execution_order
        assert "G2" in OrderTrackingGuard.execution_order
        assert "G3" in OrderTrackingGuard.execution_order

    def test_aggregation_applied(self, sample_artifact):
        """Results aggregated per policy."""
        # ALL_PASS - should fail
        guard_all = ParallelGuard(
            [AlwaysPassGuard(), AlwaysFailGuard()],
            AggregationPolicy.ALL_PASS,
        )
        assert guard_all.validate(sample_artifact).passed is False

        # ANY_PASS - should pass
        guard_any = ParallelGuard(
            [AlwaysPassGuard(), AlwaysFailGuard()],
            AggregationPolicy.ANY_PASS,
        )
        assert guard_any.validate(sample_artifact).passed is True


class TestDefinition41AggregationPolicy:
    """Tests for Definition 41: Aggregation Policy."""

    @pytest.mark.parametrize(
        "policy,pass_count,total,expected",
        [
            # ALL_PASS: requires all
            (AggregationPolicy.ALL_PASS, 3, 3, True),
            (AggregationPolicy.ALL_PASS, 2, 3, False),
            (AggregationPolicy.ALL_PASS, 0, 3, False),
            # ANY_PASS: requires at least one
            (AggregationPolicy.ANY_PASS, 1, 3, True),
            (AggregationPolicy.ANY_PASS, 3, 3, True),
            (AggregationPolicy.ANY_PASS, 0, 3, False),
            # MAJORITY_PASS: requires > 50%
            (AggregationPolicy.MAJORITY_PASS, 2, 3, True),
            (AggregationPolicy.MAJORITY_PASS, 3, 3, True),
            (AggregationPolicy.MAJORITY_PASS, 1, 3, False),
            (AggregationPolicy.MAJORITY_PASS, 1, 2, False),  # 50% is not majority
            (AggregationPolicy.MAJORITY_PASS, 2, 2, True),
        ],
    )
    def test_policy_outcomes(
        self, sample_artifact, policy, pass_count, total, expected
    ):
        """Verify policy aggregation logic."""
        guards = []
        for i in range(total):
            if i < pass_count:
                guards.append(AlwaysPassGuard())
            else:
                guards.append(AlwaysFailGuard())

        guard = ParallelGuard(guards, policy)
        result = guard.validate(sample_artifact)
        assert result.passed is expected


class TestDefinition42NestedComposition:
    """Tests for Definition 42: Nested Composition."""

    def test_composition_closure(self, sample_artifact):
        """Composite guards can be sub-guards."""
        inner = SequentialGuard([AlwaysPassGuard()])
        outer = ParallelGuard([inner, AlwaysPassGuard()])

        # Both should be valid guards
        assert isinstance(inner, GuardInterface)
        assert isinstance(outer, GuardInterface)

        result = outer.validate(sample_artifact)
        assert result.passed is True

    def test_deep_nesting(self, sample_artifact):
        """Arbitrary nesting depth works."""
        level3 = SequentialGuard([AlwaysPassGuard()])
        level2 = ParallelGuard([level3, AlwaysPassGuard()])
        level1 = SequentialGuard([level2, AlwaysPassGuard()])
        level0 = ParallelGuard([level1])

        result = level0.validate(sample_artifact)
        assert result.passed is True

    def test_nested_failure_propagates(self, sample_artifact):
        """Failure in nested guard propagates correctly."""
        inner = SequentialGuard([AlwaysFailGuard()])
        outer = ParallelGuard([inner], AggregationPolicy.ALL_PASS)

        result = outer.validate(sample_artifact)
        assert result.passed is False


class TestDefinition43SubGuardResult:
    """Tests for Definition 43: Sub-Guard Result."""

    def test_subguard_result_fields(self):
        """SubGuardResult has required fields."""
        result = SubGuardResult(
            guard_name="TestGuard",
            passed=True,
            feedback="Test feedback",
            execution_time_ms=42.5,
        )

        assert result.guard_name == "TestGuard"
        assert result.passed is True
        assert result.feedback == "Test feedback"
        assert result.execution_time_ms == 42.5

    def test_default_execution_time(self):
        """execution_time_ms defaults to 0."""
        result = SubGuardResult(
            guard_name="TestGuard",
            passed=True,
            feedback="Test",
        )
        assert result.execution_time_ms == 0.0


class TestTheorem14DynamicsPreservation:
    """Tests for Theorem 14: System Dynamics Preservation."""

    def test_output_matches_guard_interface(self, sample_artifact):
        """Output (v, φ) compatible with Definition 7."""
        # Sequential
        seq = SequentialGuard([AlwaysPassGuard()])
        seq_result = seq.validate(sample_artifact)
        assert isinstance(seq_result, GuardResult)
        assert isinstance(seq_result.passed, bool)
        assert isinstance(seq_result.feedback, str)

        # Parallel
        par = ParallelGuard([AlwaysPassGuard()])
        par_result = par.validate(sample_artifact)
        assert isinstance(par_result, GuardResult)
        assert isinstance(par_result.passed, bool)
        assert isinstance(par_result.feedback, str)

    def test_workflow_unaware_of_composition(self, sample_artifact):
        """Workflow executor sees single guard result."""
        # A simple guard and a complex composite should produce same result type
        simple = AlwaysPassGuard()
        composite = SequentialGuard(
            [
                ParallelGuard([AlwaysPassGuard(), AlwaysPassGuard()]),
                AlwaysPassGuard(),
            ]
        )

        simple_result = simple.validate(sample_artifact)
        composite_result = composite.validate(sample_artifact)

        # Both return GuardResult - workflow treats them identically
        assert type(simple_result) is type(composite_result)
        assert hasattr(simple_result, "passed")
        assert hasattr(composite_result, "passed")

    def test_composition_transparent_to_caller(self, sample_artifact):
        """Internal composition details are hidden from caller."""
        # Create a complex nested structure
        guard = SequentialGuard(
            [
                ParallelGuard(
                    [
                        SequentialGuard([AlwaysPassGuard()]),
                        AlwaysPassGuard(),
                    ]
                ),
            ]
        )

        # Caller just sees: guard.validate(artifact) -> GuardResult
        result = guard.validate(sample_artifact)

        # No internal state leaked
        assert isinstance(result, GuardResult)
        assert result.passed is True
