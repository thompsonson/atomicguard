"""Tests for SyntaxGuard."""

import pytest

from atomicguard.domain.models import Artifact, ArtifactStatus
from atomicguard.guards.syntax import SyntaxGuard


class TestSyntaxGuard:
    """Tests for SyntaxGuard validation."""

    @pytest.fixture
    def guard(self) -> SyntaxGuard:
        """Create a SyntaxGuard instance."""
        return SyntaxGuard()

    def test_valid_syntax(self, guard, sample_artifact):
        """Valid Python code should pass."""
        result = guard.validate(sample_artifact)
        assert result.passed is True
        assert "valid" in result.feedback.lower()

    def test_invalid_syntax(self, guard, invalid_syntax_artifact):
        """Invalid Python code should fail."""
        result = guard.validate(invalid_syntax_artifact)
        assert result.passed is False
        assert "syntax error" in result.feedback.lower()

    def test_empty_code(self, guard, sample_context_snapshot):
        """Empty code should be valid Python."""
        empty_artifact = Artifact(
            artifact_id="empty-001",
            content="",
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )
        result = guard.validate(empty_artifact)
        assert result.passed is True

    def test_whitespace_only(self, guard, sample_context_snapshot):
        """Whitespace-only code should be valid Python."""
        whitespace_artifact = Artifact(
            artifact_id="whitespace-001",
            content="   \n\t\n   ",
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )
        result = guard.validate(whitespace_artifact)
        assert result.passed is True

    def test_multiline_valid_code(self, guard, sample_context_snapshot):
        """Complex multiline code should validate correctly."""
        complex_code = '''
class Calculator:
    """A simple calculator class."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b

    def multiply(self, a: int, b: int) -> int:
        return a * b
'''
        artifact = Artifact(
            artifact_id="complex-001",
            content=complex_code,
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )
        result = guard.validate(artifact)
        assert result.passed is True

    def test_unterminated_string(self, guard, sample_context_snapshot):
        """Unterminated string literal should fail."""
        artifact = Artifact(
            artifact_id="bad-string-001",
            content='x = "hello',
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )
        result = guard.validate(artifact)
        assert result.passed is False

    def test_indentation_error(self, guard, sample_context_snapshot):
        """Indentation error should fail."""
        artifact = Artifact(
            artifact_id="indent-001",
            content="def foo():\nreturn 1",  # Missing indentation
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )
        result = guard.validate(artifact)
        assert result.passed is False

    def test_python_312_syntax(self, guard, sample_context_snapshot):
        """Python 3.12+ syntax should be valid."""
        # Type parameter syntax (PEP 695) - Python 3.12+
        modern_code = """
def identity[T](x: T) -> T:
    return x
"""
        artifact = Artifact(
            artifact_id="modern-001",
            content=modern_code,
            previous_attempt_id=None,
            action_pair_id="ap-001",
            created_at="2025-01-01T00:00:00Z",
            attempt_number=1,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=sample_context_snapshot,
        )
        result = guard.validate(artifact)
        assert result.passed is True
