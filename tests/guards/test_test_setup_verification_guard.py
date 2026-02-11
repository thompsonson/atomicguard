"""Tests for TestSetupVerificationGuard."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from examples.swe_bench_ablation.guards import test_setup_verification_guard

from atomicguard.domain.models import Artifact, ArtifactStatus, ContextSnapshot

# Alias to avoid pytest collection warning
_TestSetupVerificationGuard = test_setup_verification_guard.TestSetupVerificationGuard


@pytest.fixture
def sample_context_snapshot() -> ContextSnapshot:
    """Create a sample ContextSnapshot for testing."""
    return ContextSnapshot(
        workflow_id="test-workflow-001",
        specification="Verify test setup",
        constraints="",
        feedback_history=(),
        dependency_artifacts=(),
    )


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository."""
    return tmp_path


def make_artifact(content: str, context: ContextSnapshot) -> Artifact:
    """Create an artifact with the given JSON content."""
    return Artifact(
        artifact_id="test-artifact-001",
        workflow_id="test-workflow-001",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="ap_localise_tests",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=context,
    )


def make_valid_localization(
    test_library: str = "pytest",
    test_plugins: list[str] | None = None,
    test_invocation: str = "pytest tests/",
) -> dict:
    """Create a valid TestLocalization dict."""
    return {
        "test_files": ["tests/test_module.py"],
        "test_patterns": ["test_*.py"],
        "test_library": test_library,
        "test_plugins": test_plugins or [],
        "test_fixtures": [],
        "conftest_files": [],
        "test_style": "function-based",
        "test_invocation": test_invocation,
        "reasoning": "Test setup for verification",
    }


class TestNoRepoRoot:
    """Tests when repo_root is not provided."""

    def test_skips_without_repo_root(self, sample_context_snapshot: ContextSnapshot):
        """Without repo_root, verification should be skipped."""
        data = make_valid_localization()
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestSetupVerificationGuard()  # No repo_root
        result = guard.validate(artifact)

        assert result.passed is True
        assert "skipped" in result.feedback.lower()


class TestSchemaValidation:
    """Tests for JSON/schema validation."""

    def test_invalid_json_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Invalid JSON should fail."""
        artifact = make_artifact("not valid json", sample_context_snapshot)

        guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "Invalid JSON" in result.feedback

    def test_invalid_schema_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Invalid schema should fail."""
        data = {"invalid": "data"}
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "Schema validation failed" in result.feedback


class TestFrameworkCheck:
    """Tests for framework installation check."""

    def test_framework_installed_passes(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Installed framework should pass."""
        data = make_valid_localization(test_library="pytest")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "pytest 7.0.0"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert mock_run.called
        assert result.passed is True

    def test_framework_not_installed_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Missing framework should fail."""
        data = make_valid_localization(test_library="pytest")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "command not found"

        with patch("subprocess.run", return_value=mock_result):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is False
        assert "not installed" in result.feedback.lower()

    def test_framework_command_not_found(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Command not found should fail gracefully."""
        data = make_valid_localization(test_library="pytest")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        with patch("subprocess.run", side_effect=FileNotFoundError):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is False
        assert "not found" in result.feedback.lower()

    def test_framework_timeout(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Framework check timeout should fail."""
        import subprocess

        data = make_valid_localization(test_library="pytest")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10)):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is False
        assert "timed out" in result.feedback.lower()

    def test_unknown_framework_skipped(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Unknown framework should skip check, not fail."""
        data = make_valid_localization(test_library="unknown_framework")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        # Mock successful collect
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            # Call internal method directly for clarity
            internal_result = guard._check_framework_installed("unknown_framework")

        assert internal_result.passed is True
        assert "skipping" in internal_result.feedback.lower()


class TestPluginCheck:
    """Tests for plugin importability check."""

    def test_plugin_importable_passes(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Importable plugin should pass."""
        data = make_valid_localization(test_plugins=["pytest-asyncio"])
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is True

    def test_plugin_not_importable_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Non-importable plugin should fail."""
        data = make_valid_localization(test_plugins=["pytest-nonexistent"])
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        def run_side_effect(cmd, **kwargs):
            result = MagicMock()
            # pytest --version succeeds
            if "--version" in cmd or "--collect-only" in cmd:
                result.returncode = 0
            # Plugin import fails
            else:
                result.returncode = 1
            return result

        with patch("subprocess.run", side_effect=run_side_effect):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is False
        assert "not importable" in result.feedback.lower()

    def test_multiple_plugins_all_checked(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Multiple plugins should all be checked."""
        data = make_valid_localization(
            test_plugins=["pytest-asyncio", "pytest-qt", "pytest-xdist"]
        )
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        call_count = 0

        def run_side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=run_side_effect):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        # Should be: 1 framework check + 3 plugin checks + 1 collect check = 5
        assert call_count == 5
        assert result.passed is True


class TestCollectTests:
    """Tests for test collection check."""

    def test_collect_succeeds_passes(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Successful test collection should pass."""
        data = make_valid_localization(test_invocation="pytest tests/")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "<Module test_example.py>\n  <Function test_foo>"

        with patch("subprocess.run", return_value=mock_result):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is True
        assert "collectible" in result.feedback.lower()

    def test_collect_fails_reports_error(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Failed test collection should report error."""
        data = make_valid_localization(test_invocation="pytest tests/")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        def run_side_effect(cmd, **kwargs):
            result = MagicMock()
            # Framework check passes
            if "--version" in cmd:
                result.returncode = 0
            # Collect fails
            elif "--collect-only" in cmd:
                result.returncode = 1
                result.stderr = "ImportError: No module named 'missing_dep'"
            else:
                result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=run_side_effect):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is False
        assert "collection failed" in result.feedback.lower()

    def test_collect_timeout_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Collection timeout should fail."""
        import subprocess

        data = make_valid_localization(test_invocation="pytest tests/")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        call_count = 0

        def run_side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call (framework check) succeeds
            if call_count == 1:
                result = MagicMock()
                result.returncode = 0
                return result
            # Second call (collect) times out
            raise subprocess.TimeoutExpired("pytest", 30)

        with patch("subprocess.run", side_effect=run_side_effect):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is False
        assert "timed out" in result.feedback.lower()

    def test_non_pytest_runner_skipped(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Non-pytest runners should skip collection check."""
        data = make_valid_localization(
            test_library="unittest",
            test_invocation="python -m unittest discover tests/",
        )
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is True

    def test_python_m_pytest_collects(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """python -m pytest should collect with --collect-only."""
        data = make_valid_localization(test_invocation="python -m pytest tests/")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        collected_cmd = None

        def run_side_effect(cmd, **kwargs):
            nonlocal collected_cmd
            if "--collect-only" in cmd:
                collected_cmd = cmd
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=run_side_effect):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is True
        assert collected_cmd is not None
        assert "--collect-only" in collected_cmd


class TestMultipleErrors:
    """Tests for multiple error reporting."""

    def test_all_errors_reported(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """All errors should be reported."""
        data = make_valid_localization(
            test_plugins=["pytest-missing"],
            test_invocation="pytest tests/",
        )
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        call_count = 0

        def run_side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            # Framework check fails
            if "--version" in cmd or "import" in str(cmd):
                result.returncode = 1
            # Collect fails
            elif "--collect-only" in cmd:
                result.returncode = 1
                result.stderr = "Error during collection"
            else:
                result.returncode = 1
            return result

        with patch("subprocess.run", side_effect=run_side_effect):
            guard = _TestSetupVerificationGuard(repo_root=str(temp_repo))
            result = guard.validate(artifact)

        assert result.passed is False
        # All errors should be reported
        feedback_lower = result.feedback.lower()
        assert "not installed" in feedback_lower or "broken" in feedback_lower


class TestTimeoutConfiguration:
    """Tests for timeout configuration."""

    def test_custom_timeout_used(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Custom timeout should be used for collection."""
        data = make_valid_localization()
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        captured_timeout = None

        def run_side_effect(cmd, **kwargs):
            nonlocal captured_timeout
            if "--collect-only" in cmd:
                captured_timeout = kwargs.get("timeout")
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=run_side_effect):
            guard = _TestSetupVerificationGuard(
                repo_root=str(temp_repo),
                timeout_seconds=60,
            )
            guard.validate(artifact)

        assert captured_timeout == 60
