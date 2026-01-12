"""Tests for ActionPair - atomic generation-verification transaction."""

from atomicguard.application.action_pair import ActionPair
from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, Context, GuardResult
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.guards import SyntaxGuard
from atomicguard.infrastructure.llm.mock import MockGenerator


class TestActionPairInit:
    """Tests for ActionPair initialization."""

    def test_init_stores_generator_and_guard(
        self, mock_generator: MockGenerator
    ) -> None:
        """ActionPair stores generator and guard references."""
        guard = SyntaxGuard()
        pair = ActionPair(generator=mock_generator, guard=guard)

        assert pair.generator is mock_generator
        assert pair.guard is guard

    def test_init_with_prompt_template(self, mock_generator: MockGenerator) -> None:
        """ActionPair accepts optional prompt template."""
        guard = SyntaxGuard()
        template = PromptTemplate(
            role="Python developer",
            constraints="Pure Python only",
            task="Write a function",
        )
        pair = ActionPair(
            generator=mock_generator, guard=guard, prompt_template=template
        )

        assert pair._prompt_template is template


class TestActionPairProperties:
    """Tests for ActionPair property accessors."""

    def test_generator_property_returns_generator(
        self, mock_generator: MockGenerator
    ) -> None:
        """generator property returns the generator."""
        guard = SyntaxGuard()
        pair = ActionPair(generator=mock_generator, guard=guard)

        assert pair.generator is mock_generator

    def test_guard_property_returns_guard(self, mock_generator: MockGenerator) -> None:
        """guard property returns the guard."""
        guard = SyntaxGuard()
        pair = ActionPair(generator=mock_generator, guard=guard)

        assert pair.guard is guard


class TestActionPairExecute:
    """Tests for ActionPair.execute() method."""

    def test_execute_returns_artifact_and_guard_result(
        self, mock_generator: MockGenerator, sample_context: Context
    ) -> None:
        """execute() returns tuple of (Artifact, GuardResult)."""
        guard = SyntaxGuard()
        pair = ActionPair(generator=mock_generator, guard=guard)

        artifact, result = pair.execute(sample_context)

        # MockGenerator returns Artifact which has content attribute
        assert hasattr(artifact, "content")
        assert isinstance(result, GuardResult)

    def test_execute_with_valid_code_passes_guard(
        self, sample_context: Context
    ) -> None:
        """execute() with valid code returns passed=True."""
        generator = MockGenerator(responses=["def add(a, b):\n    return a + b"])
        guard = SyntaxGuard()
        pair = ActionPair(generator=generator, guard=guard)

        artifact, result = pair.execute(sample_context)

        assert result.passed is True
        assert artifact.content == "def add(a, b):\n    return a + b"

    def test_execute_with_invalid_code_fails_guard(
        self, sample_context: Context
    ) -> None:
        """execute() with invalid syntax returns passed=False."""
        generator = MockGenerator(responses=["def add(a, b:\n    return a + b"])
        guard = SyntaxGuard()
        pair = ActionPair(generator=generator, guard=guard)

        artifact, result = pair.execute(sample_context)

        assert result.passed is False
        assert "Syntax error" in result.feedback or "SyntaxError" in result.feedback

    def test_execute_with_dependencies(self, sample_context: Context) -> None:
        """execute() passes dependencies to guard."""
        generator = MockGenerator(responses=["def add(a, b):\n    return a + b"])

        class DependencyCheckingGuard(GuardInterface):
            """Guard that records dependencies for testing."""

            def __init__(self) -> None:
                self.received_deps: dict[str, Artifact] | None = None

            def validate(
                self, _artifact: Artifact, **dependencies: Artifact
            ) -> GuardResult:
                self.received_deps = dependencies
                return GuardResult(passed=True, feedback="")

        guard = DependencyCheckingGuard()
        pair = ActionPair(generator=generator, guard=guard)

        # Create a mock dependency artifact
        from atomicguard.domain.models import ArtifactStatus, ContextSnapshot

        dep_artifact = Artifact(
            artifact_id="dep-001",
            workflow_id="test-workflow-001",
            content="# dependency",
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id="ap-dep",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.ACCEPTED,
            guard_result=None,
            context=ContextSnapshot(
                workflow_id="test-workflow-001",
                specification="",
                constraints="",
                feedback_history=(),
                dependency_artifacts=(),
            ),
        )

        dependencies = {"test": dep_artifact}
        pair.execute(sample_context, dependencies=dependencies)

        assert guard.received_deps is not None
        assert "test" in guard.received_deps
        assert guard.received_deps["test"] is dep_artifact

    def test_execute_with_none_dependencies(self, sample_context: Context) -> None:
        """execute() handles None dependencies gracefully."""
        generator = MockGenerator(responses=["x = 1"])
        guard = SyntaxGuard()
        pair = ActionPair(generator=generator, guard=guard)

        # Should not raise
        artifact, result = pair.execute(sample_context, dependencies=None)

        assert result.passed is True
