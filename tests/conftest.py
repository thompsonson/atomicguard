"""Shared pytest fixtures for atomicguard tests."""

import pytest

from atomicguard.domain.models import (
    AmbientEnvironment,
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
)
from atomicguard.infrastructure.llm.mock import MockGenerator
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


@pytest.fixture
def sample_context_snapshot() -> ContextSnapshot:
    """Create a sample ContextSnapshot for testing."""
    return ContextSnapshot(
        workflow_id="test-workflow-001",
        specification="Write a function that adds two numbers",
        constraints="Must be pure Python, no imports",
        feedback_history=(),
        dependency_artifacts=(),
    )


@pytest.fixture
def sample_artifact(sample_context_snapshot: ContextSnapshot) -> Artifact:
    """Create a sample Artifact for testing."""
    return Artifact(
        artifact_id="test-artifact-001",
        workflow_id="test-workflow-001",
        content="def add(a, b):\n    return a + b",
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="ap-001",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        feedback="",
        context=sample_context_snapshot,
    )


@pytest.fixture
def invalid_syntax_artifact(sample_context_snapshot: ContextSnapshot) -> Artifact:
    """Create an artifact with invalid Python syntax."""
    return Artifact(
        artifact_id="invalid-syntax-001",
        workflow_id="test-workflow-001",
        content="def add(a, b\n    return a + b",  # Missing closing paren
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="ap-001",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        feedback="",
        context=sample_context_snapshot,
    )


@pytest.fixture
def memory_dag() -> InMemoryArtifactDAG:
    """Create an in-memory artifact DAG."""
    return InMemoryArtifactDAG()


@pytest.fixture
def mock_generator() -> MockGenerator:
    """Create a mock generator with sample responses."""
    return MockGenerator(
        responses=[
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return a + b  # Fixed version",
        ]
    )


@pytest.fixture
def ambient_env(memory_dag: InMemoryArtifactDAG) -> AmbientEnvironment:
    """Create an ambient environment for testing."""
    return AmbientEnvironment(
        repository=memory_dag,
        constraints="Must be valid Python 3.12+",
    )


@pytest.fixture
def sample_context(ambient_env: AmbientEnvironment) -> Context:
    """Create a full Context for testing."""
    return Context(
        ambient=ambient_env,
        specification="Write a function that adds two numbers",
        current_artifact=None,
        feedback_history=(),
        dependency_artifacts=(),
    )
