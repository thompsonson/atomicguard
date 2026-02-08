"""LintGuard: Static analysis guard for catching undefined names and missing imports.

Uses pyflakes to perform fast (~10ms) semantic checking before expensive Docker
test execution. Only flags errors INTRODUCED by the patch (compares baseline vs
patched).
"""

import io
import json
import logging
from pathlib import Path
from typing import Any

from pyflakes import api as pyflakes_api
from pyflakes import messages as pyflakes_messages
from pyflakes.checker import Checker

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

logger = logging.getLogger("swe_bench_ablation.guards")

# Fatal pyflakes message types that indicate code will fail at runtime.
# These are the only errors we care about - style warnings are ignored.
FATAL_MESSAGE_TYPES = (
    pyflakes_messages.UndefinedName,  # F821: name 'X' is not defined
    pyflakes_messages.UndefinedExport,  # F822: undefined name in __all__
    pyflakes_messages.ImportStarUsed,  # F405: may be undefined from star import
    pyflakes_messages.UndefinedLocal,  # F823: local variable referenced before assignment
)


class LintGuard(GuardInterface):
    """Static analysis guard that catches undefined names and missing imports.

    This guard runs pyflakes on the patched code and compares against the
    baseline to identify NEW errors introduced by the patch. It's designed
    to catch issues like:
    - Using IntEnum without importing it from enum
    - Referencing undefined variables
    - Missing imports that would cause NameError at runtime

    The guard is fast (~10ms) and runs before expensive Docker test execution.
    """

    def __init__(
        self,
        repo_root: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the guard.

        Args:
            repo_root: Repository root for finding and reading files.
        """
        self._repo_root = repo_root

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the patch artifact for lint errors.

        Args:
            artifact: The patch artifact containing edits to validate.
            **deps: Artifacts from prior workflow steps.

        Returns:
            GuardResult with pass/fail and specific error messages.
        """
        logger.info("[LintGuard] Validating artifact %s...", artifact.artifact_id[:8])

        if not self._repo_root:
            logger.warning("[LintGuard] No repo_root provided, skipping lint check")
            return GuardResult(
                passed=True,
                feedback="Lint check skipped (no repo_root)",
                guard_name="LintGuard",
            )

        # Parse JSON to get edits
        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="LintGuard",
            )

        edits = data.get("edits", [])
        if not edits:
            # No edits to validate
            logger.info("[LintGuard] PASSED: No edits to validate")
            return GuardResult(
                passed=True,
                feedback="No edits to validate",
                guard_name="LintGuard",
            )

        # Apply edits and check for introduced errors
        introduced_errors: list[str] = []

        for file_path, patched_content in self._apply_edits(edits, self._repo_root):
            if not file_path.endswith(".py"):
                continue

            full_path = Path(self._repo_root) / file_path
            if not full_path.exists():
                continue

            try:
                original_content = full_path.read_text()
            except Exception as e:
                logger.warning("[LintGuard] Could not read %s: %s", file_path, e)
                continue

            # Lint original (baseline errors)
            baseline_errors = self._lint_python(original_content, file_path)

            # Lint patched (new + baseline errors)
            patched_errors = self._lint_python(patched_content, file_path)

            # Find errors introduced by the patch
            new_errors = patched_errors - baseline_errors

            for error in sorted(new_errors):
                introduced_errors.append(f"{file_path}: {error}")

        if introduced_errors:
            # Limit to first 5 errors to avoid overwhelming feedback
            shown_errors = introduced_errors[:5]
            hidden_count = len(introduced_errors) - len(shown_errors)

            feedback = "LINT ERRORS INTRODUCED BY PATCH:\n- " + "\n- ".join(shown_errors)
            if hidden_count > 0:
                feedback += f"\n... and {hidden_count} more errors"
            feedback += (
                "\n\nFix these issues before generating the patch. "
                "Common fixes:\n"
                "- Add missing imports (e.g., 'from enum import IntEnum')\n"
                "- Check variable names are defined before use"
            )

            logger.info(
                "[LintGuard] ✗ REJECTED: %d lint error(s) introduced",
                len(introduced_errors),
            )
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="LintGuard",
            )

        logger.info("[LintGuard] ✓ PASSED: No lint errors introduced")
        return GuardResult(
            passed=True,
            feedback="Lint check passed: no undefined names or missing imports",
            guard_name="LintGuard",
        )

    def _lint_python(self, code: str, filename: str) -> set[str]:
        """Run pyflakes on Python code and return fatal error messages.

        Args:
            code: Python source code to lint.
            filename: Filename for error reporting.

        Returns:
            Set of error message strings (only fatal errors, not warnings).
        """
        import ast

        errors: set[str] = set()

        try:
            # Parse code to AST (pyflakes Checker needs AST, not compiled code)
            tree = ast.parse(code, filename)
            checker = Checker(tree, filename)

            for message in checker.messages:
                # Only include fatal message types
                if isinstance(message, FATAL_MESSAGE_TYPES):
                    # Format: "line N: message"
                    error_str = f"line {message.lineno}: {message.message % message.message_args}"
                    errors.add(error_str)

        except SyntaxError:
            # Syntax errors are handled by PatchGuard, skip here
            pass
        except Exception as e:
            logger.debug("[LintGuard] Error linting %s: %s", filename, e)

        return errors

    def _apply_edits(
        self,
        edits: list[dict[str, str]],
        repo_root: str,
    ) -> list[tuple[str, str]]:
        """Apply edits to files in memory and return patched content.

        Args:
            edits: List of edit dicts with 'file', 'search', 'replace' keys.
            repo_root: Repository root path.

        Returns:
            List of (file_path, patched_content) tuples for modified Python files.
        """
        # Group edits by file
        edits_by_file: dict[str, list[dict[str, str]]] = {}
        for edit in edits:
            file_path = edit.get("file", "")
            if not file_path:
                continue
            if file_path not in edits_by_file:
                edits_by_file[file_path] = []
            edits_by_file[file_path].append(edit)

        results: list[tuple[str, str]] = []

        for file_path, file_edits in edits_by_file.items():
            if not file_path.endswith(".py"):
                continue

            full_path = Path(repo_root) / file_path
            if not full_path.exists():
                continue

            try:
                content = full_path.read_text()

                # Apply all edits for this file
                for edit in file_edits:
                    search = edit.get("search", "")
                    replace = edit.get("replace", "")
                    if search and search in content:
                        content = content.replace(search, replace, 1)

                results.append((file_path, content))

            except Exception as e:
                logger.warning("[LintGuard] Error applying edits to %s: %s", file_path, e)

        return results
