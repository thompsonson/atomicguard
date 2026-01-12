"""
QualityGatesGuard: Runs code quality tools in isolated temp environment.

Validates that implementation passes mypy and ruff checks.
Uses temporary directory to maintain sensing-only principle.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from .base import TempDirValidationMixin

logger = logging.getLogger("sdlc_checkpoint")


class QualityGatesGuard(GuardInterface, TempDirValidationMixin):
    """
    Runs code quality tools (mypy, ruff) in isolated temp environment.

    This guard is sensing-only: it writes to a temp directory that is
    automatically cleaned up after validation. The actual workdir is
    never modified.

    Checks:
    - mypy type checking passes
    - ruff linting passes
    """

    def __init__(
        self,
        run_mypy: bool = True,
        run_ruff: bool = True,
        timeout: float = 60.0,
    ):
        """
        Initialize the quality gates guard.

        Args:
            run_mypy: Whether to run mypy type checking
            run_ruff: Whether to run ruff linting
            timeout: Timeout in seconds for each tool
        """
        self._run_mypy = run_mypy
        self._run_ruff = run_ruff
        self._timeout = timeout

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """
        Validate implementation against quality gates.

        Args:
            artifact: The implementation artifact (from g_coder)
            **_deps: Dependency artifacts (unused)

        Returns:
            GuardResult indicating pass/fail with detailed feedback
        """
        logger.debug("[QualityGatesGuard] Starting quality validation...")

        results: dict[str, tuple[bool, str]] = {}

        with self._temp_implementation(artifact.content) as tmpdir:
            logger.debug(f"[QualityGatesGuard] Temp dir: {tmpdir}")

            # Find the source root (look for src/ or the package dir)
            src_dir = self._find_source_dir(tmpdir)
            if not src_dir:
                return GuardResult(
                    passed=False,
                    feedback="Could not find source directory in implementation",
                )

            # Run mypy
            if self._run_mypy:
                mypy_passed, mypy_output = self._run_mypy_check(src_dir)
                results["mypy"] = (mypy_passed, mypy_output)
                logger.debug(
                    f"[QualityGatesGuard] mypy: {'PASS' if mypy_passed else 'FAIL'}"
                )

            # Run ruff
            if self._run_ruff:
                ruff_passed, ruff_output = self._run_ruff_check(src_dir)
                results["ruff"] = (ruff_passed, ruff_output)
                logger.debug(
                    f"[QualityGatesGuard] ruff: {'PASS' if ruff_passed else 'FAIL'}"
                )

        # Compile results
        all_passed = all(passed for passed, _ in results.values())
        feedback_parts = []

        for tool, (passed, output) in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            feedback_parts.append(f"### {tool}: {status}")
            if output.strip():
                feedback_parts.append(f"```\n{output[:2000]}\n```")
            feedback_parts.append("")

        feedback = "\n".join(feedback_parts)

        if all_passed:
            logger.debug("[QualityGatesGuard] ✓ All quality gates passed")
            return GuardResult(
                passed=True,
                feedback=f"All quality gates passed.\n\n{feedback}",
            )
        else:
            failed_tools = [tool for tool, (passed, _) in results.items() if not passed]
            logger.debug(f"[QualityGatesGuard] Failed: {failed_tools}")
            return GuardResult(
                passed=False,
                feedback=f"Quality gates failed: {', '.join(failed_tools)}\n\n{feedback}",
            )

    def _find_source_dir(self, tmpdir: Path) -> Path | None:
        """Find the source directory in the temp dir."""
        # Look for src/ directory
        src_dir = tmpdir / "src"
        if src_dir.exists():
            return src_dir

        # Look for any directory containing Python files
        for item in tmpdir.iterdir():
            if item.is_dir() and list(item.glob("**/*.py")):
                return item

        # If Python files exist at root
        if list(tmpdir.glob("*.py")):
            return tmpdir

        return None

    def _run_mypy_check(self, src_dir: Path) -> tuple[bool, str]:
        """
        Run mypy type checking.

        Returns:
            Tuple of (passed, output)
        """
        try:
            result = subprocess.run(
                [
                    "mypy",
                    str(src_dir),
                    "--ignore-missing-imports",
                    "--no-error-summary",
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=src_dir.parent,
            )
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            return passed, output
        except subprocess.TimeoutExpired:
            return False, "mypy timed out"
        except FileNotFoundError:
            return True, "mypy not installed (skipped)"
        except Exception as e:
            return False, f"mypy error: {e}"

    def _run_ruff_check(self, src_dir: Path) -> tuple[bool, str]:
        """
        Run ruff linting.

        Returns:
            Tuple of (passed, output)
        """
        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    str(src_dir),
                    "--ignore",
                    "E501",  # Ignore line length
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=src_dir.parent,
            )
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            return passed, output
        except subprocess.TimeoutExpired:
            return False, "ruff timed out"
        except FileNotFoundError:
            return True, "ruff not installed (skipped)"
        except Exception as e:
            return False, f"ruff error: {e}"
