"""Tests for CompositeGuard and Extension 08 composite patterns."""

import time

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult
from atomicguard.guards import (
    AggregationPolicy,
    CompositeGuard,
    ParallelGuard,
    SequentialGuard,
    SyntaxGuard,
)


class AlwaysPassGuard(GuardInterface):
    """Test guard that always passes."""

    def validate(self, artifact: Artifact, **deps) -> GuardResult:  # noqa: ARG002
        return GuardResult(passed=True, feedback="Always passes")


class AlwaysFailGuard(GuardInterface):
    """Test guard that always fails."""

    def validate(self, artifact: Artifact, **deps) -> GuardResult:  # noqa: ARG002
        return GuardResult(passed=False, feedback="Always fails")


class CountingGuard(GuardInterface):
    """Test guard that counts invocations."""

    def __init__(self):
        self.call_count = 0

    def validate(self, artifact: Artifact, **deps) -> GuardResult:  # noqa: ARG002
        self.call_count += 1
        return GuardResult(passed=True, feedback=f"Call {self.call_count}")


class SlowGuard(GuardInterface):
    """Test guard that takes time to execute."""

    def __init__(self, delay_ms: float = 100, should_pass: bool = True):
        self.delay_ms = delay_ms
        self.should_pass = should_pass
        self.executed = False

    def validate(self, artifact: Artifact, **deps) -> GuardResult:  # noqa: ARG002
        time.sleep(self.delay_ms / 1000)
        self.executed = True
        return GuardResult(
            passed=self.should_pass,
            feedback="Slow guard passed" if self.should_pass else "Slow guard failed",
        )


class TestCompositeGuard:
    """Tests for CompositeGuard composition."""

    def test_all_pass(self, sample_artifact):
        """When all guards pass, composite passes."""
        composite = CompositeGuard(AlwaysPassGuard(), AlwaysPassGuard())
        result = composite.validate(sample_artifact)
        assert result.passed is True

    def test_first_fails(self, sample_artifact):
        """When first guard fails, short-circuits."""
        composite = CompositeGuard(AlwaysFailGuard(), AlwaysPassGuard())
        result = composite.validate(sample_artifact)
        assert result.passed is False
        assert "Always fails" in result.feedback

    def test_second_fails(self, sample_artifact):
        """When second guard fails, returns that failure."""
        composite = CompositeGuard(AlwaysPassGuard(), AlwaysFailGuard())
        result = composite.validate(sample_artifact)
        assert result.passed is False
        assert "Always fails" in result.feedback

    def test_empty_composite(self, sample_artifact):
        """Empty composite should pass."""
        composite = CompositeGuard()
        result = composite.validate(sample_artifact)
        assert result.passed is True
        # Empty composite returns "No guards to evaluate"
        assert "guards" in result.feedback.lower()

    def test_single_guard(self, sample_artifact):
        """Single guard in composite works correctly."""
        composite = CompositeGuard(AlwaysPassGuard())
        result = composite.validate(sample_artifact)
        assert result.passed is True

    def test_short_circuit_behavior(self, sample_artifact):
        """Verify short-circuit prevents later guards from running."""
        counter = CountingGuard()
        composite = CompositeGuard(AlwaysFailGuard(), counter)
        composite.validate(sample_artifact)
        # Counter should not be called due to short-circuit
        assert counter.call_count == 0

    def test_all_guards_called_when_passing(self, sample_artifact):
        """All guards should be called when all pass."""
        counter1 = CountingGuard()
        counter2 = CountingGuard()
        composite = CompositeGuard(counter1, counter2)
        composite.validate(sample_artifact)
        assert counter1.call_count == 1
        assert counter2.call_count == 1

    def test_with_syntax_guard_valid(self, sample_artifact):
        """Integration with real SyntaxGuard - valid code."""
        composite = CompositeGuard(SyntaxGuard(), AlwaysPassGuard())
        result = composite.validate(sample_artifact)
        assert result.passed is True

    def test_with_syntax_guard_invalid(self, invalid_syntax_artifact):
        """Integration with real SyntaxGuard - invalid code."""
        composite = CompositeGuard(SyntaxGuard(), AlwaysPassGuard())
        result = composite.validate(invalid_syntax_artifact)
        assert result.passed is False
        assert "syntax error" in result.feedback.lower()

    def test_three_guards(self, sample_artifact):
        """Test with three guards."""
        composite = CompositeGuard(
            AlwaysPassGuard(),
            AlwaysPassGuard(),
            AlwaysPassGuard(),
        )
        result = composite.validate(sample_artifact)
        assert result.passed is True

    def test_middle_guard_fails(self, sample_artifact):
        """Middle guard failure should short-circuit."""
        counter = CountingGuard()
        composite = CompositeGuard(
            AlwaysPassGuard(),
            AlwaysFailGuard(),
            counter,
        )
        result = composite.validate(sample_artifact)
        assert result.passed is False
        assert counter.call_count == 0

    def test_nested_composite(self, sample_artifact):
        """Nested composites should work correctly."""
        inner = CompositeGuard(AlwaysPassGuard(), AlwaysPassGuard())
        outer = CompositeGuard(inner, AlwaysPassGuard())
        result = outer.validate(sample_artifact)
        assert result.passed is True

    def test_nested_composite_inner_fails(self, sample_artifact):
        """Nested composite with inner failure."""
        inner = CompositeGuard(AlwaysPassGuard(), AlwaysFailGuard())
        outer = CompositeGuard(inner, AlwaysPassGuard())
        result = outer.validate(sample_artifact)
        assert result.passed is False


class TestSequentialGuard:
    """Tests for SequentialGuard (Definition 39)."""

    def test_all_pass(self, sample_artifact):
        """When all guards pass, sequential passes."""
        guard = SequentialGuard([AlwaysPassGuard(), AlwaysPassGuard()])
        result = guard.validate(sample_artifact)
        assert result.passed is True

    def test_fail_fast_stops_execution(self, sample_artifact):
        """Verify fail-fast behavior stops at first failure."""
        counter = CountingGuard()
        guard = SequentialGuard([AlwaysFailGuard(), counter])
        guard.validate(sample_artifact)
        # Counter should not be called due to short-circuit
        assert counter.call_count == 0

    def test_all_guards_called_on_success(self, sample_artifact):
        """All guards should be called when all pass."""
        counter1 = CountingGuard()
        counter2 = CountingGuard()
        guard = SequentialGuard([counter1, counter2])
        guard.validate(sample_artifact)
        assert counter1.call_count == 1
        assert counter2.call_count == 1

    def test_feedback_from_failed_guard(self, sample_artifact):
        """Feedback comes from the guard that failed."""
        guard = SequentialGuard([AlwaysPassGuard(), AlwaysFailGuard()])
        result = guard.validate(sample_artifact)
        assert result.passed is False
        assert "AlwaysFailGuard" in result.feedback

    def test_empty_guards(self, sample_artifact):
        """Empty guards list should pass."""
        guard = SequentialGuard([])
        result = guard.validate(sample_artifact)
        assert result.passed is True

    def test_guards_property(self):
        """Guards are accessible via property."""
        guards = [AlwaysPassGuard(), AlwaysFailGuard()]
        sequential = SequentialGuard(guards)
        assert len(sequential.guards) == 2


class TestParallelGuard:
    """Tests for ParallelGuard (Definition 40)."""

    def test_all_guards_execute(self, sample_artifact):
        """Verify all guards run even if some fail."""
        counter1 = CountingGuard()
        counter2 = CountingGuard()
        guard = ParallelGuard([AlwaysFailGuard(), counter1, counter2])
        guard.validate(sample_artifact)
        # Both counters should be called despite failure
        assert counter1.call_count == 1
        assert counter2.call_count == 1

    def test_all_pass_policy(self, sample_artifact):
        """ALL_PASS requires all guards to pass."""
        guard = ParallelGuard(
            [AlwaysPassGuard(), AlwaysFailGuard()],
            policy=AggregationPolicy.ALL_PASS,
        )
        result = guard.validate(sample_artifact)
        assert result.passed is False

    def test_any_pass_policy(self, sample_artifact):
        """ANY_PASS passes if at least one guard passes."""
        guard = ParallelGuard(
            [AlwaysPassGuard(), AlwaysFailGuard()],
            policy=AggregationPolicy.ANY_PASS,
        )
        result = guard.validate(sample_artifact)
        assert result.passed is True

    def test_any_pass_all_fail(self, sample_artifact):
        """ANY_PASS fails if all guards fail."""
        guard = ParallelGuard(
            [AlwaysFailGuard(), AlwaysFailGuard()],
            policy=AggregationPolicy.ANY_PASS,
        )
        result = guard.validate(sample_artifact)
        assert result.passed is False

    def test_majority_pass_policy(self, sample_artifact):
        """MAJORITY_PASS passes if >50% guards pass."""
        guard = ParallelGuard(
            [AlwaysPassGuard(), AlwaysPassGuard(), AlwaysFailGuard()],
            policy=AggregationPolicy.MAJORITY_PASS,
        )
        result = guard.validate(sample_artifact)
        assert result.passed is True

    def test_majority_pass_fails(self, sample_artifact):
        """MAJORITY_PASS fails if <=50% pass."""
        guard = ParallelGuard(
            [AlwaysPassGuard(), AlwaysFailGuard(), AlwaysFailGuard()],
            policy=AggregationPolicy.MAJORITY_PASS,
        )
        result = guard.validate(sample_artifact)
        assert result.passed is False

    def test_combined_feedback_on_failure(self, sample_artifact):
        """All failure feedback is aggregated."""
        guard = ParallelGuard([AlwaysFailGuard(), AlwaysFailGuard()])
        result = guard.validate(sample_artifact)
        # Should contain feedback from both failures
        assert result.feedback.count("AlwaysFailGuard") == 2

    def test_concurrent_execution(self, sample_artifact):
        """Verify guards execute concurrently."""
        # Two guards that each take 100ms
        slow1 = SlowGuard(delay_ms=100)
        slow2 = SlowGuard(delay_ms=100)
        guard = ParallelGuard([slow1, slow2])

        start = time.time()
        guard.validate(sample_artifact)
        elapsed = time.time() - start

        # If concurrent, should take ~100ms, not 200ms
        # Allow some margin for overhead
        assert elapsed < 0.18  # 180ms max (with overhead)
        assert slow1.executed
        assert slow2.executed


class TestAggregationPolicy:
    """Tests for AggregationPolicy (Definition 41)."""

    def test_all_pass_all_succeed(self, sample_artifact):
        """ALL_PASS with all successes."""
        guard = SequentialGuard(
            [AlwaysPassGuard(), AlwaysPassGuard()],
            policy=AggregationPolicy.ALL_PASS,
        )
        result = guard.validate(sample_artifact)
        assert result.passed is True

    def test_all_pass_one_fails(self, sample_artifact):
        """ALL_PASS with one failure."""
        guard = SequentialGuard(
            [AlwaysPassGuard(), AlwaysFailGuard()],
            policy=AggregationPolicy.ALL_PASS,
        )
        result = guard.validate(sample_artifact)
        assert result.passed is False

    def test_policy_enum_values(self):
        """Verify policy enum has expected values."""
        assert AggregationPolicy.ALL_PASS.value == "all_pass"
        assert AggregationPolicy.ANY_PASS.value == "any_pass"
        assert AggregationPolicy.MAJORITY_PASS.value == "majority_pass"


class TestNestedComposition:
    """Tests for nested guards (Definition 42)."""

    def test_sequential_containing_parallel(self, sample_artifact):
        """Sequential guard with parallel sub-guard."""
        parallel = ParallelGuard([AlwaysPassGuard(), AlwaysPassGuard()])
        sequential = SequentialGuard([AlwaysPassGuard(), parallel])
        result = sequential.validate(sample_artifact)
        assert result.passed is True

    def test_parallel_containing_sequential(self, sample_artifact):
        """Parallel guard with sequential sub-guards."""
        seq1 = SequentialGuard([AlwaysPassGuard(), AlwaysPassGuard()])
        seq2 = SequentialGuard([AlwaysPassGuard(), AlwaysPassGuard()])
        parallel = ParallelGuard([seq1, seq2])
        result = parallel.validate(sample_artifact)
        assert result.passed is True

    def test_three_level_nesting(self, sample_artifact):
        """Deep nesting works correctly."""
        inner = SequentialGuard([AlwaysPassGuard()])
        middle = ParallelGuard([inner, AlwaysPassGuard()])
        outer = SequentialGuard([middle, AlwaysPassGuard()])
        result = outer.validate(sample_artifact)
        assert result.passed is True

    def test_nested_failure_propagates(self, sample_artifact):
        """Failure in nested guard propagates up."""
        inner = SequentialGuard([AlwaysFailGuard()])
        outer = ParallelGuard([inner, AlwaysPassGuard()])
        result = outer.validate(sample_artifact)
        assert result.passed is False


class TestBackwardsCompatibility:
    """Ensure existing code continues to work."""

    def test_composite_guard_varargs(self, sample_artifact):
        """CompositeGuard(*guards) still works."""
        composite = CompositeGuard(AlwaysPassGuard(), AlwaysPassGuard())
        result = composite.validate(sample_artifact)
        assert result.passed is True

    def test_composite_guard_short_circuits(self, sample_artifact):
        """CompositeGuard maintains short-circuit behavior."""
        counter = CountingGuard()
        composite = CompositeGuard(AlwaysFailGuard(), counter)
        composite.validate(sample_artifact)
        assert counter.call_count == 0

    def test_composite_guard_guards_attribute(self):
        """CompositeGuard.guards tuple is accessible."""
        guards = (AlwaysPassGuard(), AlwaysPassGuard())
        composite = CompositeGuard(*guards)
        assert len(composite.guards) == 2
