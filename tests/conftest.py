"""Shared pytest fixtures for atomicguard tests."""

import pytest

from atomicguard.domain.models import (
    AmbientEnvironment,
    AmendmentType,
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    FailureType,
    HumanAmendment,
    WorkflowCheckpoint,
)
from atomicguard.infrastructure.llm.mock import MockGenerator
from atomicguard.infrastructure.persistence.checkpoint import (
    FilesystemCheckpointDAG,
    InMemoryCheckpointDAG,
)
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


# =============================================================================
# CHECKPOINT FIXTURES
# =============================================================================


@pytest.fixture
def sample_checkpoint() -> WorkflowCheckpoint:
    """Create a sample WorkflowCheckpoint for testing."""
    return WorkflowCheckpoint(
        checkpoint_id="chk-test-001",
        workflow_id="wf-test-001",
        created_at="2025-01-05T10:00:00Z",
        specification="Test specification content",
        constraints="Must be valid Python 3.12+",
        rmax=3,
        completed_steps=("g_test",),
        artifact_ids=(("g_test", "art-001"),),
        failure_type=FailureType.RMAX_EXHAUSTED,
        failed_step="g_impl",
        failed_artifact_id="art-002",
        failure_feedback="Syntax error on line 5",
        provenance_ids=("art-002", "art-003"),
    )


@pytest.fixture
def sample_checkpoint_escalation() -> WorkflowCheckpoint:
    """Create a checkpoint for an escalation failure."""
    return WorkflowCheckpoint(
        checkpoint_id="chk-test-002",
        workflow_id="wf-test-002",
        created_at="2025-01-05T11:00:00Z",
        specification="Test specification for escalation",
        constraints="No external dependencies",
        rmax=3,
        completed_steps=("g_config", "g_test"),
        artifact_ids=(("g_config", "art-010"), ("g_test", "art-011")),
        failure_type=FailureType.ESCALATION,
        failed_step="g_impl",
        failed_artifact_id="art-012",
        failure_feedback="FATAL: Security vulnerability detected",
        provenance_ids=("art-012",),
    )


@pytest.fixture
def sample_checkpoint_empty_completed() -> WorkflowCheckpoint:
    """Create a checkpoint with no completed steps (failed at first step)."""
    return WorkflowCheckpoint(
        checkpoint_id="chk-test-003",
        workflow_id="wf-test-003",
        created_at="2025-01-05T12:00:00Z",
        specification="Test specification",
        constraints="",
        rmax=1,
        completed_steps=(),
        artifact_ids=(),
        failure_type=FailureType.RMAX_EXHAUSTED,
        failed_step="g_first",
        failed_artifact_id="art-020",
        failure_feedback="Failed on first attempt",
        provenance_ids=("art-020",),
    )


@pytest.fixture
def sample_amendment() -> HumanAmendment:
    """Create a sample HumanAmendment with artifact type."""
    return HumanAmendment(
        amendment_id="amd-test-001",
        checkpoint_id="chk-test-001",
        amendment_type=AmendmentType.ARTIFACT,
        created_at="2025-01-05T10:05:00Z",
        created_by="test-user",
        content="def fixed_function():\n    return True",
        context="Fixed the syntax error",
        parent_artifact_id="art-002",
        additional_rmax=0,
    )


@pytest.fixture
def sample_amendment_feedback() -> HumanAmendment:
    """Create a sample HumanAmendment with feedback type."""
    return HumanAmendment(
        amendment_id="amd-test-002",
        checkpoint_id="chk-test-001",
        amendment_type=AmendmentType.FEEDBACK,
        created_at="2025-01-05T10:10:00Z",
        created_by="reviewer",
        content="Use only standard library functions, no external imports",
        context="The previous attempt used numpy which is not allowed",
        parent_artifact_id="art-002",
        additional_rmax=2,
    )


@pytest.fixture
def sample_amendment_skip() -> HumanAmendment:
    """Create a sample HumanAmendment with skip type."""
    return HumanAmendment(
        amendment_id="amd-test-003",
        checkpoint_id="chk-test-002",
        amendment_type=AmendmentType.SKIP,
        created_at="2025-01-05T11:05:00Z",
        created_by="admin",
        content="",
        context="This step is optional for MVP",
        parent_artifact_id=None,
        additional_rmax=0,
    )


@pytest.fixture
def memory_checkpoint_dag() -> InMemoryCheckpointDAG:
    """Create InMemoryCheckpointDAG for testing."""
    return InMemoryCheckpointDAG()


@pytest.fixture
def fs_checkpoint_dag(tmp_path) -> FilesystemCheckpointDAG:
    """Create FilesystemCheckpointDAG with temp directory."""
    return FilesystemCheckpointDAG(str(tmp_path / "checkpoints"))
