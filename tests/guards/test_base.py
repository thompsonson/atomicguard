"""Tests for CompositeGuard."""

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult
from atomicguard.guards import CompositeGuard, SyntaxGuard


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
        assert "All guards passed" in result.feedback

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
