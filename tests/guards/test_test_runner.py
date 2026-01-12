"""Tests for TestGuard and DynamicTestGuard dependency handling."""

import pytest

from atomicguard.domain.models import Artifact, ArtifactStatus, ContextSnapshot
from atomicguard.guards.dynamic import test_runner


@pytest.fixture
def test_context_snapshot() -> ContextSnapshot:
    """Create a context snapshot for test artifacts."""
    return ContextSnapshot(
        workflow_id="test-workflow-001",
        specification="Test implementation",
        constraints="",
        feedback_history=(),
        dependency_artifacts=(),
    )


@pytest.fixture
def impl_artifact(test_context_snapshot: ContextSnapshot) -> Artifact:
    """Create a sample implementation artifact."""
    return Artifact(
        artifact_id="impl-001",
        workflow_id="test-workflow-001",
        content="class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def pop(self):\n        return self._items.pop()",
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_impl",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=test_context_snapshot,
    )


@pytest.fixture
def test_artifact(test_context_snapshot: ContextSnapshot) -> Artifact:
    """Create a sample test artifact with passing tests (for sync TestGuard)."""
    return Artifact(
        artifact_id="test-001",
        workflow_id="test-workflow-001",
        content="def test_stack_push_pop():\n    s = Stack()\n    s.push(1)\n    assert s.pop() == 1",
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_test",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=test_context_snapshot,
    )


@pytest.fixture
def test_artifact_with_import(test_context_snapshot: ContextSnapshot) -> Artifact:
    """Create test artifact with import (for DynamicTestGuard subprocess)."""
    return Artifact(
        artifact_id="test-002",
        workflow_id="test-workflow-001",
        content="from implementation import Stack\ndef test_stack_push_pop():\n    s = Stack()\n    s.push(1)\n    assert s.pop() == 1",
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_test",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=test_context_snapshot,
    )


@pytest.fixture
def failing_test_artifact(test_context_snapshot: ContextSnapshot) -> Artifact:
    """Create a test artifact with a test that fails against the impl."""
    # Test that checks for a method that doesn't exist
    return Artifact(
        artifact_id="test-003",
        workflow_id="test-workflow-001",
        content="def test_missing_method():\n    s = Stack()\n    s.nonexistent_method()  # This will fail",
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_test",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=test_context_snapshot,
    )


@pytest.fixture
def failing_test_artifact_standalone(
    test_context_snapshot: ContextSnapshot,
) -> Artifact:
    """Create a standalone failing test (doesn't need impl in namespace).

    Note: For sync TestGuard, assertions must be at module level since exec()
    just defines functions without calling them. For DynamicTestGuard, pytest
    discovers and runs test_ functions automatically.
    """
    return Artifact(
        artifact_id="test-004",
        workflow_id="test-workflow-001",
        # Module-level assertion for sync TestGuard
        content="assert False, 'intentional failure'",
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_test",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=test_context_snapshot,
    )


@pytest.fixture
def failing_test_artifact_for_dynamic(
    test_context_snapshot: ContextSnapshot,
) -> Artifact:
    """Create a pytest-style failing test for DynamicTestGuard."""
    return Artifact(
        artifact_id="test-005",
        workflow_id="test-workflow-001",
        content="def test_always_fails():\n    assert False, 'intentional failure'",
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="g_test",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=test_context_snapshot,
    )


class TestSyncGuardDependencyHandling:
    """Tests for TestGuard (sync) auto-detecting dependencies."""

    def test_autodetects_first_dependency_with_standard_key(
        self, impl_artifact: Artifact, test_artifact: Artifact
    ) -> None:
        """TestGuard uses first dependency regardless of key name."""
        guard = test_runner.TestGuard()
        result = guard.validate(impl_artifact, g_test=test_artifact)
        assert result.passed is True

    def test_autodetects_first_dependency_with_custom_key(
        self, impl_artifact: Artifact, test_artifact: Artifact
    ) -> None:
        """TestGuard works with arbitrary dependency key names."""
        guard = test_runner.TestGuard()
        result = guard.validate(impl_artifact, my_custom_tests=test_artifact)
        assert result.passed is True

    def test_autodetects_first_dependency_without_prefix(
        self, impl_artifact: Artifact, test_artifact: Artifact
    ) -> None:
        """TestGuard works with non-prefixed key names."""
        guard = test_runner.TestGuard()
        result = guard.validate(impl_artifact, unit_tests=test_artifact)
        assert result.passed is True

    def test_static_test_code_takes_precedence(
        self, impl_artifact: Artifact, test_artifact: Artifact
    ) -> None:
        """Static test code in constructor takes precedence over dependencies."""
        static_test = "def test_static():\n    s = Stack()\n    s.push(42)\n    assert s.pop() == 42"
        guard = test_runner.TestGuard(test_code=static_test)
        result = guard.validate(impl_artifact, g_test=test_artifact)
        assert result.passed is True

    def test_no_test_code_fails(self, impl_artifact: Artifact) -> None:
        """Guard fails when no test code is provided."""
        guard = test_runner.TestGuard()
        result = guard.validate(impl_artifact)
        assert result.passed is False
        assert "No test code provided" in result.feedback

    def test_failing_tests_return_failure(
        self, impl_artifact: Artifact, failing_test_artifact_standalone: Artifact
    ) -> None:
        """Guard returns failure when tests fail."""
        guard = test_runner.TestGuard()
        result = guard.validate(impl_artifact, tests=failing_test_artifact_standalone)
        assert result.passed is False
        assert "failed" in result.feedback.lower()


class TestDynamicGuardDependencyHandling:
    """Tests for DynamicTestGuard auto-detecting dependencies."""

    def test_autodetects_first_dependency_with_standard_key(
        self, impl_artifact: Artifact, test_artifact_with_import: Artifact
    ) -> None:
        """DynamicTestGuard uses first dependency regardless of key name."""
        guard = test_runner.DynamicTestGuard(timeout=30.0)
        result = guard.validate(impl_artifact, g_test=test_artifact_with_import)
        assert result.passed is True

    def test_autodetects_first_dependency_with_custom_key(
        self, impl_artifact: Artifact, test_artifact_with_import: Artifact
    ) -> None:
        """DynamicTestGuard works with arbitrary dependency key names."""
        guard = test_runner.DynamicTestGuard(timeout=30.0)
        result = guard.validate(
            impl_artifact, my_custom_tests=test_artifact_with_import
        )
        assert result.passed is True

    def test_autodetects_first_dependency_without_prefix(
        self, impl_artifact: Artifact, test_artifact_with_import: Artifact
    ) -> None:
        """DynamicTestGuard works with non-prefixed key names."""
        guard = test_runner.DynamicTestGuard(timeout=30.0)
        result = guard.validate(impl_artifact, unit_tests=test_artifact_with_import)
        assert result.passed is True

    def test_static_test_code_takes_precedence(
        self, impl_artifact: Artifact, test_artifact_with_import: Artifact
    ) -> None:
        """Static test code in constructor takes precedence over dependencies."""
        static_test = "from implementation import Stack\ndef test_static():\n    s = Stack()\n    s.push(42)\n    assert s.pop() == 42"
        guard = test_runner.DynamicTestGuard(timeout=30.0, test_code=static_test)
        result = guard.validate(impl_artifact, g_test=test_artifact_with_import)
        assert result.passed is True

    def test_no_test_code_fails(self, impl_artifact: Artifact) -> None:
        """Guard fails when no test code is provided."""
        guard = test_runner.DynamicTestGuard(timeout=30.0)
        result = guard.validate(impl_artifact)
        assert result.passed is False
        assert "No test code provided" in result.feedback

    def test_failing_tests_return_failure(
        self, impl_artifact: Artifact, failing_test_artifact_for_dynamic: Artifact
    ) -> None:
        """Guard returns failure when tests fail."""
        guard = test_runner.DynamicTestGuard(timeout=30.0)
        result = guard.validate(impl_artifact, tests=failing_test_artifact_for_dynamic)
        assert result.passed is False
