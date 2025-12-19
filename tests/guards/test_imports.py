"""Tests for ImportGuard."""

import pytest

from atomicguard.domain.models import Artifact, ArtifactStatus, ContextSnapshot
from atomicguard.guards import ImportGuard


@pytest.fixture
def context_snapshot() -> ContextSnapshot:
    """Create a sample context snapshot."""
    return ContextSnapshot(
        specification="Write tests",
        constraints="",
        feedback_history=(),
        dependency_ids=(),
    )


@pytest.fixture
def guard() -> ImportGuard:
    """Create an ImportGuard instance."""
    return ImportGuard()


def make_artifact(content: str, context: ContextSnapshot) -> Artifact:
    """Create an artifact with given content."""
    return Artifact(
        artifact_id="test-001",
        content=content,
        previous_attempt_id=None,
        action_pair_id="ap-001",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        feedback="",
        context=context,
    )


class TestImportGuardBasicValidation:
    """Tests for basic import validation."""

    def test_valid_code_with_all_imports(self, guard, context_snapshot):
        """Code with all names imported should pass."""
        code = """
import pytest

def test_example():
    with pytest.raises(ValueError):
        raise ValueError("test")
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_missing_import(self, guard, context_snapshot):
        """Code using undefined names should fail."""
        code = """
def test_example():
    with pytest.raises(ValueError):
        raise ValueError("test")
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "pytest" in result.feedback
        assert "Undefined names" in result.feedback

    def test_empty_code(self, guard, context_snapshot):
        """Empty code should pass."""
        artifact = make_artifact("", context_snapshot)
        result = guard.validate(artifact)
        # Empty content should fail because no code provided
        assert result.passed is False
        assert "No code provided" in result.feedback

    def test_whitespace_only(self, guard, context_snapshot):
        """Whitespace-only code should pass (valid Python)."""
        artifact = make_artifact("   \n\t\n   ", context_snapshot)
        result = guard.validate(artifact)
        # Just whitespace is valid code with no undefined names
        assert result.passed is True


class TestImportGuardDefinedNames:
    """Tests for various ways names can be defined."""

    def test_function_definition(self, guard, context_snapshot):
        """Functions define their own names."""
        code = """
def my_function():
    pass

my_function()
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_class_definition(self, guard, context_snapshot):
        """Classes define their own names."""
        code = """
class MyClass:
    pass

obj = MyClass()
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_assignment(self, guard, context_snapshot):
        """Assignments define names."""
        code = """
x = 1
y = x + 1
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_for_loop_variable(self, guard, context_snapshot):
        """For loop variables are defined."""
        code = """
items = [1, 2, 3]
for item in items:
    print(item)
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_with_statement_variable(self, guard, context_snapshot):
        """With statement variables are defined."""
        code = """
with open("file.txt") as f:
    content = f.read()
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_exception_handler_variable(self, guard, context_snapshot):
        """Exception handler variables are defined."""
        code = """
try:
    x = 1
except ValueError as e:
    print(e)
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_comprehension_variable(self, guard, context_snapshot):
        """Comprehension variables are defined."""
        code = """
squares = [x * x for x in range(10)]
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_function_parameters(self, guard, context_snapshot):
        """Function parameters are defined."""
        code = """
def add(a, b):
    return a + b
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_walrus_operator(self, guard, context_snapshot):
        """Walrus operator defines names."""
        code = """
if (n := 10) > 5:
    print(n)
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True


class TestImportGuardImportStatements:
    """Tests for various import styles."""

    def test_simple_import(self, guard, context_snapshot):
        """Simple import statement."""
        code = """
import os
print(os.getcwd())
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_import_as(self, guard, context_snapshot):
        """Import with alias."""
        code = """
import numpy as np
arr = np.array([1, 2, 3])
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_from_import(self, guard, context_snapshot):
        """From import statement."""
        code = """
from os import getcwd
print(getcwd())
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_from_import_as(self, guard, context_snapshot):
        """From import with alias."""
        code = """
from os import getcwd as get_current_dir
print(get_current_dir())
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_multiple_imports(self, guard, context_snapshot):
        """Multiple imports from same module."""
        code = """
from os import getcwd, listdir
getcwd()
listdir(".")
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True


class TestImportGuardBuiltins:
    """Tests for builtin name recognition."""

    def test_builtin_functions(self, guard, context_snapshot):
        """Builtin functions should not need imports."""
        code = """
print("hello")
x = len([1, 2, 3])
y = range(10)
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_builtin_types(self, guard, context_snapshot):
        """Builtin types should not need imports."""
        code = """
x = int("42")
y = str(123)
z = list()
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_builtin_exceptions(self, guard, context_snapshot):
        """Builtin exceptions should not need imports."""
        code = """
raise ValueError("test")
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_builtin_constants(self, guard, context_snapshot):
        """Builtin constants should not need imports."""
        code = """
x = True
y = False
z = None
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True


class TestImportGuardSyntaxErrors:
    """Tests for syntax error handling."""

    def test_syntax_error_caught(self, guard, context_snapshot):
        """Syntax errors should fail with appropriate message."""
        code = "def foo( pass"
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "Syntax error" in result.feedback


class TestImportGuardRealWorldScenarios:
    """Tests for real-world test code scenarios."""

    def test_pytest_test_class_with_import(self, guard, context_snapshot):
        """Pytest test class with proper imports should pass."""
        code = """
import pytest

from implementation import Stack


class TestStack:
    def test_push_pop(self):
        stack = Stack()
        stack.push(1)
        assert stack.pop() == 1

    def test_empty_raises(self):
        stack = Stack()
        with pytest.raises(IndexError):
            stack.pop()
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is True

    def test_pytest_test_class_missing_import(self, guard, context_snapshot):
        """Pytest test class missing import should fail."""
        code = """
from implementation import Stack


class TestStack:
    def test_push_pop(self):
        stack = Stack()
        stack.push(1)
        assert stack.pop() == 1

    def test_empty_raises(self):
        stack = Stack()
        with pytest.raises(IndexError):
            stack.pop()
"""
        artifact = make_artifact(code, context_snapshot)
        result = guard.validate(artifact)
        assert result.passed is False
        assert "pytest" in result.feedback
