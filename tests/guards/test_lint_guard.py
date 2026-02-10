"""Tests for LintGuard."""

import json
from pathlib import Path

import pytest
from examples.swe_bench_ablation.guards.lint_guard import LintGuard

from atomicguard.domain.models import Artifact, ArtifactStatus, ContextSnapshot


@pytest.fixture
def sample_context_snapshot() -> ContextSnapshot:
    """Create a sample ContextSnapshot for testing."""
    return ContextSnapshot(
        workflow_id="test-workflow-001",
        specification="Fix the bug",
        constraints="",
        feedback_history=(),
        dependency_artifacts=(),
    )


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository with test files."""
    return tmp_path


def make_artifact(content: str, context: ContextSnapshot) -> Artifact:
    """Create an artifact with the given JSON content."""
    return Artifact(
        artifact_id="test-artifact-001",
        workflow_id="test-workflow-001",
        content=content,
        previous_attempt_id=None,
        parent_action_pair_id=None,
        action_pair_id="ap-001",
        created_at="2025-01-01T00:00:00Z",
        attempt_number=1,
        status=ArtifactStatus.PENDING,
        guard_result=None,
        context=context,
    )


class TestLintGuard:
    """Tests for LintGuard validation."""

    def test_catches_missing_import(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Using IntEnum without importing it should fail."""
        # Create original file without any imports
        original_file = temp_repo / "models.py"
        original_file.write_text("# Original file\nclass Foo:\n    pass\n")

        # Edit adds IntEnum usage without import
        edits = [
            {
                "file": "models.py",
                "search": "class Foo:\n    pass",
                "replace": "class Status(IntEnum):\n    PENDING = 1\n    DONE = 2",
            }
        ]
        artifact = make_artifact(
            json.dumps({"edits": edits, "patch": ""}), sample_context_snapshot
        )

        guard = LintGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "IntEnum" in result.feedback
        assert "undefined" in result.feedback.lower() or "LINT ERRORS" in result.feedback

    def test_passes_with_import(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Using IntEnum with proper import should pass."""
        # Create original file
        original_file = temp_repo / "models.py"
        original_file.write_text("# Original file\nclass Foo:\n    pass\n")

        # Edit adds IntEnum usage WITH import
        edits = [
            {
                "file": "models.py",
                "search": "# Original file\nclass Foo:\n    pass",
                "replace": "from enum import IntEnum\n\nclass Status(IntEnum):\n    PENDING = 1\n    DONE = 2",
            }
        ]
        artifact = make_artifact(
            json.dumps({"edits": edits, "patch": ""}), sample_context_snapshot
        )

        guard = LintGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True
        assert "passed" in result.feedback.lower() or "no" in result.feedback.lower()

    def test_ignores_preexisting_errors(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Errors that exist before the patch should not cause failure."""
        # Create original file with pre-existing undefined name
        original_file = temp_repo / "broken.py"
        original_file.write_text("x = undefined_var\n\ndef foo():\n    return 1\n")

        # Edit doesn't add new errors
        edits = [
            {
                "file": "broken.py",
                "search": "def foo():\n    return 1",
                "replace": "def foo():\n    return 2",
            }
        ]
        artifact = make_artifact(
            json.dumps({"edits": edits, "patch": ""}), sample_context_snapshot
        )

        guard = LintGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True

    def test_ignores_unused_imports(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """UnusedImport (F401) is not in FATAL_MESSAGES and should be ignored."""
        # Create original file
        original_file = temp_repo / "utils.py"
        original_file.write_text("def helper():\n    return 1\n")

        # Edit adds an unused import (pyflakes F401)
        edits = [
            {
                "file": "utils.py",
                "search": "def helper():\n    return 1",
                "replace": "import os\n\ndef helper():\n    return 1",
            }
        ]
        artifact = make_artifact(
            json.dumps({"edits": edits, "patch": ""}), sample_context_snapshot
        )

        guard = LintGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is True

    def test_no_edits_passes(self, sample_context_snapshot: ContextSnapshot):
        """Artifact with no edits should pass."""
        artifact = make_artifact(
            json.dumps({"edits": [], "patch": ""}), sample_context_snapshot
        )

        guard = LintGuard(repo_root="/tmp")
        result = guard.validate(artifact)

        assert result.passed is True

    def test_no_repo_root_skips(self, sample_context_snapshot: ContextSnapshot):
        """Without repo_root, lint check should be skipped."""
        edits = [{"file": "test.py", "search": "a", "replace": "b"}]
        artifact = make_artifact(
            json.dumps({"edits": edits, "patch": ""}), sample_context_snapshot
        )

        guard = LintGuard()  # No repo_root
        result = guard.validate(artifact)

        assert result.passed is True
        assert "skipped" in result.feedback.lower()

    def test_invalid_json_fails(self, sample_context_snapshot: ContextSnapshot):
        """Invalid JSON should fail."""
        artifact = make_artifact("not valid json", sample_context_snapshot)

        guard = LintGuard(repo_root="/tmp")
        result = guard.validate(artifact)

        assert result.passed is False
        assert "JSON" in result.feedback

    def test_catches_undefined_local(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """UndefinedLocal (F823) should be caught."""
        # Create original file
        original_file = temp_repo / "logic.py"
        original_file.write_text("def process():\n    return None\n")

        # Edit introduces variable used before assignment
        edits = [
            {
                "file": "logic.py",
                "search": "def process():\n    return None",
                "replace": "def process():\n    print(x)\n    x = 1\n    return x",
            }
        ]
        artifact = make_artifact(
            json.dumps({"edits": edits, "patch": ""}), sample_context_snapshot
        )

        guard = LintGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        assert "x" in result.feedback.lower() or "undefined" in result.feedback.lower()

    def test_non_python_files_ignored(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Non-Python files should be ignored."""
        # Create a non-Python file
        config_file = temp_repo / "config.yaml"
        config_file.write_text("key: value\n")

        # Edit only affects non-Python file
        edits = [
            {
                "file": "config.yaml",
                "search": "key: value",
                "replace": "key: new_value",
            }
        ]
        artifact = make_artifact(
            json.dumps({"edits": edits, "patch": ""}), sample_context_snapshot
        )

        guard = LintGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        # Should pass since no Python files were affected
        assert result.passed is True

    def test_multiple_files_multiple_errors(
        self, temp_repo: Path, sample_context_snapshot: ContextSnapshot
    ):
        """Multiple errors across multiple files should all be reported."""
        # Create two files
        (temp_repo / "file1.py").write_text("def a():\n    pass\n")
        (temp_repo / "file2.py").write_text("def b():\n    pass\n")

        # Edits introduce errors in both files
        edits = [
            {
                "file": "file1.py",
                "search": "def a():\n    pass",
                "replace": "def a():\n    return UndefinedOne",
            },
            {
                "file": "file2.py",
                "search": "def b():\n    pass",
                "replace": "def b():\n    return UndefinedTwo",
            },
        ]
        artifact = make_artifact(
            json.dumps({"edits": edits, "patch": ""}), sample_context_snapshot
        )

        guard = LintGuard(repo_root=str(temp_repo))
        result = guard.validate(artifact)

        assert result.passed is False
        # Both files should be mentioned
        assert "file1.py" in result.feedback
        assert "file2.py" in result.feedback


class TestLintPython:
    """Tests for the _lint_python helper method."""

    def test_lint_undefined_name(self):
        """Direct test of _lint_python for undefined names."""
        guard = LintGuard()
        code = "x = undefined_var"
        errors = guard._lint_python(code, "test.py")

        assert len(errors) == 1
        error = list(errors)[0]
        assert "undefined_var" in error

    def test_lint_valid_code(self):
        """Valid code should have no errors."""
        guard = LintGuard()
        code = "x = 1\ny = x + 1"
        errors = guard._lint_python(code, "test.py")

        assert len(errors) == 0

    def test_lint_syntax_error_ignored(self):
        """Syntax errors are handled by PatchGuard, not here."""
        guard = LintGuard()
        code = "def foo(\n"  # Syntax error
        errors = guard._lint_python(code, "test.py")

        # Should not raise, and should return empty set
        assert len(errors) == 0


class TestApplyEdits:
    """Tests for the _apply_edits helper method."""

    def test_apply_single_edit(self, temp_repo: Path):
        """Single edit should be applied correctly."""
        original = temp_repo / "test.py"
        original.write_text("def foo():\n    return 1\n")

        guard = LintGuard(repo_root=str(temp_repo))
        edits = [
            {
                "file": "test.py",
                "search": "return 1",
                "replace": "return 2",
            }
        ]
        results = guard._apply_edits(edits, str(temp_repo))

        assert len(results) == 1
        file_path, content = results[0]
        assert file_path == "test.py"
        assert "return 2" in content
        assert "return 1" not in content

    def test_apply_multiple_edits_same_file(self, temp_repo: Path):
        """Multiple edits to the same file should all be applied."""
        original = temp_repo / "test.py"
        original.write_text("a = 1\nb = 2\n")

        guard = LintGuard(repo_root=str(temp_repo))
        edits = [
            {"file": "test.py", "search": "a = 1", "replace": "a = 10"},
            {"file": "test.py", "search": "b = 2", "replace": "b = 20"},
        ]
        results = guard._apply_edits(edits, str(temp_repo))

        assert len(results) == 1
        file_path, content = results[0]
        assert "a = 10" in content
        assert "b = 20" in content
