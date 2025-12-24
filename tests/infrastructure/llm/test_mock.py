"""Tests for MockGenerator - predefined response generator for testing."""

import pytest

from atomicguard.domain.models import Artifact, ArtifactStatus
from atomicguard.infrastructure.llm.mock import MockGenerator


class TestMockGeneratorInit:
    """Tests for MockGenerator initialization."""

    def test_init_stores_responses(self) -> None:
        """MockGenerator stores response list."""
        generator = MockGenerator(responses=["a", "b", "c"])

        assert generator._responses == ["a", "b", "c"]

    def test_init_call_count_starts_at_zero(self) -> None:
        """Call count starts at 0."""
        generator = MockGenerator(responses=["x"])

        assert generator._call_count == 0


class TestMockGeneratorGenerate:
    """Tests for MockGenerator.generate() method."""

    def test_generate_returns_first_response(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """generate() returns first response on first call."""
        generator = MockGenerator(responses=["first", "second"])

        artifact = generator.generate(sample_context)

        assert artifact.content == "first"

    def test_generate_returns_sequential_responses(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """generate() returns responses in sequence."""
        generator = MockGenerator(responses=["first", "second", "third"])

        artifact1 = generator.generate(sample_context)
        artifact2 = generator.generate(sample_context)
        artifact3 = generator.generate(sample_context)

        assert artifact1.content == "first"
        assert artifact2.content == "second"
        assert artifact3.content == "third"

    def test_generate_increments_call_count(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """generate() increments call count."""
        generator = MockGenerator(responses=["a", "b"])

        generator.generate(sample_context)
        assert generator.call_count == 1

        generator.generate(sample_context)
        assert generator.call_count == 2

    def test_generate_returns_artifact(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """generate() returns Artifact."""
        generator = MockGenerator(responses=["x = 1"])

        artifact = generator.generate(sample_context)

        assert isinstance(artifact, Artifact)

    def test_generate_sets_attempt_number(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """generate() sets attempt_number based on call count."""
        generator = MockGenerator(responses=["a", "b", "c"])

        artifact1 = generator.generate(sample_context)
        artifact2 = generator.generate(sample_context)

        assert artifact1.attempt_number == 1
        assert artifact2.attempt_number == 2

    def test_generate_assigns_unique_artifact_id(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """generate() assigns unique artifact IDs."""
        generator = MockGenerator(responses=["a", "b"])

        artifact1 = generator.generate(sample_context)
        artifact2 = generator.generate(sample_context)

        assert artifact1.artifact_id != artifact2.artifact_id

    def test_generate_raises_when_exhausted(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """generate() raises RuntimeError when responses exhausted."""
        generator = MockGenerator(responses=["only_one"])

        generator.generate(sample_context)  # Use the only response

        with pytest.raises(RuntimeError, match="exhausted responses"):
            generator.generate(sample_context)

    def test_generate_ignores_template(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """generate() ignores template parameter."""
        from atomicguard.domain.prompts import PromptTemplate

        generator = MockGenerator(responses=["result"])
        template = PromptTemplate(role="test", constraints="none", task="test")

        artifact = generator.generate(sample_context, template)

        assert artifact.content == "result"


class TestMockGeneratorCallCount:
    """Tests for MockGenerator.call_count property."""

    def test_call_count_property(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """call_count property returns current count."""
        generator = MockGenerator(responses=["a", "b", "c"])

        assert generator.call_count == 0
        generator.generate(sample_context)
        assert generator.call_count == 1
        generator.generate(sample_context)
        assert generator.call_count == 2


class TestMockGeneratorReset:
    """Tests for MockGenerator.reset() method."""

    def test_reset_clears_call_count(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """reset() sets call count back to 0."""
        generator = MockGenerator(responses=["a", "b"])

        generator.generate(sample_context)
        generator.generate(sample_context)
        assert generator.call_count == 2

        generator.reset()

        assert generator.call_count == 0

    def test_reset_allows_reuse_of_responses(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """reset() allows responses to be used again."""
        generator = MockGenerator(responses=["x"])

        artifact1 = generator.generate(sample_context)
        generator.reset()
        artifact2 = generator.generate(sample_context)

        assert artifact1.content == "x"
        assert artifact2.content == "x"


class TestMockGeneratorArtifactMetadata:
    """Tests documenting current generator behavior for artifact metadata.

    These tests verify what the generator currently produces.
    The issues stem from generator-set values not being overridden by agent.
    """

    def test_generate_uses_global_call_count_for_attempt_number(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """Generator uses its global call count as attempt_number.

        This documents current behavior. The issue is that agent.py
        should override this with a per-action-pair counter.
        """
        generator = MockGenerator(responses=["a", "b", "c"])

        artifact1 = generator.generate(sample_context)
        artifact2 = generator.generate(sample_context)
        artifact3 = generator.generate(sample_context)

        # Generator uses global call count (current behavior)
        assert artifact1.attempt_number == 1
        assert artifact2.attempt_number == 2
        assert artifact3.attempt_number == 3

    def test_generate_always_sets_previous_attempt_id_to_none(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """Generator always sets previous_attempt_id=None.

        This documents current behavior. The issue is that agent.py
        should link artifacts during retries.
        """
        generator = MockGenerator(responses=["a", "b"])

        artifact1 = generator.generate(sample_context)
        artifact2 = generator.generate(sample_context)

        # All artifacts from generator have None (current behavior)
        assert artifact1.previous_attempt_id is None
        assert artifact2.previous_attempt_id is None

    def test_generate_sets_empty_feedback_history_in_context(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """Generator sets empty feedback_history in context snapshot.

        This documents current behavior. The issue is that agent.py
        should populate feedback_history during retries.
        """
        generator = MockGenerator(responses=["a"])

        artifact = generator.generate(sample_context)

        assert artifact.context.feedback_history == ()

    def test_generate_sets_pending_status(
        self,
        sample_context,  # noqa: ANN001
    ) -> None:
        """Generator correctly sets status=PENDING (agent updates after guard)."""
        generator = MockGenerator(responses=["a"])

        artifact = generator.generate(sample_context)

        assert artifact.status == ArtifactStatus.PENDING
