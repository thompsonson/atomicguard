"""Tests for TestLocalizationGuard."""

import json
from pathlib import Path

import pytest
from examples.swe_bench_common.guards import test_localization_guard

from atomicguard.domain.models import Artifact, ArtifactStatus, ContextSnapshot

# Alias to avoid pytest collection warning
_TestLocalizationGuard = test_localization_guard.TestLocalizationGuard


@pytest.fixture
def sample_context_snapshot() -> ContextSnapshot:
    """Create a sample ContextSnapshot for testing."""
    return ContextSnapshot(
        workflow_id="test-workflow-001",
        specification="Localize tests for bug fix",
        constraints="",
        feedback_history=(),
        dependency_artifacts=(),
    )


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository with test files."""
    # Create test directory structure
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_module.py").write_text("def test_example(): pass\n")
    (tests_dir / "conftest.py").write_text("import pytest\n")
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
    test_files: list[str] | None = None,
    test_invocation: str = "pytest tests/test_module.py",
    test_library: str = "pytest",
    test_style: str = "function-based",
    proposed_test_file: str | None = None,
) -> dict:
    """Create a valid TestLocalization dict."""
    result = {
        "test_files": ["tests/test_module.py"] if test_files is None else test_files,
        "test_patterns": ["test_*.py"],
        "test_library": test_library,
        "test_plugins": [],
        "test_fixtures": ["tmp_path"],
        "conftest_files": ["tests/conftest.py"],
        "test_style": test_style,
        "test_invocation": test_invocation,
        "reasoning": "Found test file in tests directory",
    }
    if proposed_test_file is not None:
        result["proposed_test_file"] = proposed_test_file
    return result


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_valid_localization_passes(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Valid TestLocalization should pass."""
        data = make_valid_localization()
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True
        assert "1 test files" in result.feedback
        assert "library=pytest" in result.feedback

    def test_invalid_json_fails(self, sample_context_snapshot: ContextSnapshot):
        """Invalid JSON should fail."""
        artifact = make_artifact("not valid json", sample_context_snapshot)

        guard = _TestLocalizationGuard()
        result = guard.validate(artifact)

        assert result.passed is False
        assert "Invalid JSON" in result.feedback

    def test_missing_test_files_fails(self, sample_context_snapshot: ContextSnapshot):
        """Missing test_files field should fail schema validation."""
        data = {
            "test_patterns": ["test_*.py"],
            "test_invocation": "pytest tests/",
        }
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard()
        result = guard.validate(artifact)

        assert result.passed is False
        assert "Schema validation failed" in result.feedback

    def test_empty_test_files_no_proposed_fails(
        self, sample_context_snapshot: ContextSnapshot
    ):
        """Empty test_files with no proposed_test_file should fail validation."""
        data = make_valid_localization(test_files=[])
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard()
        result = guard.validate(artifact)

        assert result.passed is False
        assert "Schema validation failed" in result.feedback

    def test_error_field_fails(self, sample_context_snapshot: ContextSnapshot):
        """Generator error should fail."""
        data = {"error": "Could not locate tests"}
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard()
        result = guard.validate(artifact)

        assert result.passed is False
        assert "Generator returned error" in result.feedback


class TestLibraryValidation:
    """Tests for test library validation."""

    @pytest.mark.parametrize(
        "library",
        ["pytest", "unittest", "nose", "doctest", "hypothesis"],
    )
    def test_valid_libraries_pass(
        self,
        library: str,
        temp_repo: Path,
        sample_context_snapshot: ContextSnapshot,
    ):
        """Valid test libraries should pass."""
        data = make_valid_localization(test_library=library)
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True

    def test_invalid_library_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Invalid test library should fail."""
        data = make_valid_localization(test_library="unknown_framework")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "Unknown test library" in result.feedback


class TestStyleValidation:
    """Tests for test style validation."""

    @pytest.mark.parametrize(
        "style",
        ["function-based", "class-based", "bdd", "mixed"],
    )
    def test_valid_styles_pass(
        self,
        style: str,
        temp_repo: Path,
        sample_context_snapshot: ContextSnapshot,
    ):
        """Valid test styles should pass."""
        data = make_valid_localization(test_style=style)
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True

    def test_invalid_style_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Invalid test style should fail schema validation."""
        data = make_valid_localization()
        data["test_style"] = "invalid_style"
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        # Pydantic will reject invalid literal value
        assert "Schema validation failed" in result.feedback


class TestFileExistenceValidation:
    """Tests for file existence validation."""

    def test_missing_test_file_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Missing test file should fail."""
        data = make_valid_localization(test_files=["tests/nonexistent.py"])
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "do not exist in the repository" in result.feedback
        assert "nonexistent.py" in result.feedback

    def test_missing_conftest_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Missing conftest file should fail."""
        data = make_valid_localization()
        data["conftest_files"] = ["tests/missing_conftest.py"]
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "do not exist in the repository" in result.feedback

    def test_no_repo_root_skips_file_check(
        self, sample_context_snapshot: ContextSnapshot
    ):
        """Without repo_root, file existence check should be skipped."""
        data = make_valid_localization(test_files=["tests/nonexistent.py"])
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard()  # No repo_root
        result = guard.validate(artifact)

        # Should pass because we can't verify files
        assert result.passed is True


class TestInvocationValidation:
    """Tests for test invocation validation."""

    def test_valid_pytest_invocation_passes(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Valid pytest invocation should pass."""
        data = make_valid_localization(test_invocation="pytest tests/test_module.py -v")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True

    def test_python_m_pytest_invocation_passes(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Python -m pytest invocation should pass."""
        data = make_valid_localization(
            test_invocation="python -m pytest tests/test_module.py"
        )
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True

    def test_invalid_path_in_invocation_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Invalid path in invocation should fail."""
        data = make_valid_localization(
            test_invocation="pytest tests/nonexistent_dir/test_foo.py"
        )
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "does not exist in the repository" in result.feedback

    def test_unknown_runner_warns(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Unknown test runner should warn but not fail if paths exist."""
        data = make_valid_localization(
            test_invocation="weird_runner tests/test_module.py"
        )
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "Unknown test runner" in result.feedback

    def test_empty_invocation_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Empty invocation should fail."""
        data = make_valid_localization(test_invocation="")
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "empty" in result.feedback.lower()

    def test_pytest_path_function_syntax(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Pytest path::function syntax should validate the path part."""
        data = make_valid_localization(
            test_invocation="pytest tests/test_module.py::test_example"
        )
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True


class TestMultipleErrors:
    """Tests for multiple error reporting."""

    def test_multiple_errors_all_reported(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Multiple errors should all be reported."""
        data = make_valid_localization(
            test_files=["tests/nonexistent.py"],
            test_library="unknown_lib",
            test_invocation="weird_runner bad/path.py",
        )
        data["conftest_files"] = ["missing/conftest.py"]
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        # All errors should be in feedback
        assert "Unknown test library" in result.feedback
        assert "do not exist in the repository" in result.feedback


class TestProposedTestFile:
    """Tests for proposed_test_file support."""

    def test_proposed_with_empty_test_files_passes(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """proposed_test_file with empty test_files should pass when parent dir exists."""
        data = make_valid_localization(
            test_files=[],
            proposed_test_file="tests/test_new.py",
            test_invocation="pytest tests/test_new.py",
        )
        data["conftest_files"] = []
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True
        assert "0 existing + 1 proposed test file" in result.feedback

    def test_proposed_no_ancestor_dir_fails(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """proposed_test_file with no existing ancestor directory should fail."""
        data = make_valid_localization(
            test_files=[],
            proposed_test_file="nonexistent_dir/subdir/test_new.py",
            test_invocation="pytest nonexistent_dir/subdir/test_new.py",
        )
        data["conftest_files"] = []
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "no existing ancestor directory" in result.feedback

    def test_proposed_ancestor_dir_exists_passes(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """proposed_test_file where an ancestor (not immediate parent) exists should pass."""
        # tests/ exists but tests/subdir/ doesn't
        data = make_valid_localization(
            test_files=[],
            proposed_test_file="tests/subdir/test_new.py",
            test_invocation="pytest tests/subdir/test_new.py",
        )
        data["conftest_files"] = []
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True

    def test_neither_test_files_nor_proposed_fails(
        self, sample_context_snapshot: ContextSnapshot
    ):
        """Both test_files=[] and proposed_test_file=None should fail schema validation."""
        data = make_valid_localization(test_files=[])
        # proposed_test_file defaults to None
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard()
        result = guard.validate(artifact)

        assert result.passed is False
        assert "Schema validation failed" in result.feedback

    def test_both_test_files_and_proposed_passes(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Both test_files and proposed_test_file provided should pass."""
        data = make_valid_localization(
            test_files=["tests/test_module.py"],
            proposed_test_file="tests/test_new.py",
        )
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True
        assert "1 existing + 1 proposed test file" in result.feedback

    def test_invocation_referencing_proposed_file_passes(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Test invocation using proposed_test_file path should not be rejected as missing."""
        data = make_valid_localization(
            test_files=[],
            proposed_test_file="tests/test_new_feature.py",
            test_invocation="pytest tests/test_new_feature.py -v",
        )
        data["conftest_files"] = []
        artifact = make_artifact(json.dumps(data), sample_context_snapshot)

        guard = _TestLocalizationGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True
