"""
All Tests Pass Guard: Validates that implementation passes all tests.

Runs pytest on the workspace to validate implementation correctness.
"""

import subprocess
from pathlib import Path
from typing import Any

from ..interfaces import GuardResult, IGuard, WorkspaceManifest


class AllTestsPassGuard(IGuard):
    """Validate that all tests pass for the implementation.

    Responsibilities:
    - Run pytest on workspace
    - Parse test results
    - Return pass/fail with feedback

    Does NOT:
    - Call LLM
    - Generate code
    - Store artifacts

    Note:
        For the PoC, we expect the Coder to generate both implementation
        and tests. In a full Multi-Agent SDLC, separate TDD/BDD agents
        would generate tests first.
    """

    def __init__(self, timeout: int = 60):
        """Initialize guard.

        Args:
            timeout: Maximum seconds for test execution (default: 60)
        """
        self.timeout = timeout

    def validate(
        self, manifest: WorkspaceManifest, workspace: Path, context: dict[str, Any]
    ) -> GuardResult:
        """Validate implementation by running tests.

        Args:
            manifest: Artifact content (implementation files)
            workspace: Filesystem location with code
            context: Dependencies and configuration

        Returns:
            GuardResult with test results

        Validation approach:
        1. Check if tests directory exists
        2. Run pytest with verbose output
        3. Parse results (passed/failed/errors)
        4. Return feedback if failures
        """
        # Check 1: Tests directory exists
        tests_dir = workspace / "tests"
        if not tests_dir.exists():
            return GuardResult(
                passed=False,
                feedback="No tests directory found. Expected: tests/",
                artifacts={"tests_found": False},
            )

        # Check if there are any test files
        test_files = list(tests_dir.rglob("test_*.py"))
        if not test_files:
            return GuardResult(
                passed=False,
                feedback=f"No test files found in {tests_dir}",
                artifacts={"tests_found": False, "test_files": []},
            )

        # Check 2: Run pytest
        try:
            result = subprocess.run(
                [
                    "pytest",
                    str(tests_dir),
                    "-v",
                    "--tb=short",
                    "--no-header",
                    "--color=no",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=workspace,
            )

            # Parse output
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # Extract test counts
            test_counts = self._parse_test_counts(output)

            if passed:
                return GuardResult(
                    passed=True,
                    feedback="",
                    artifacts={
                        "tests_found": True,
                        "test_files": [str(f.relative_to(workspace)) for f in test_files],
                        "test_counts": test_counts,
                    },
                )
            else:
                # Extract failure details
                failure_summary = self._extract_failure_summary(output)
                return GuardResult(
                    passed=False,
                    feedback=f"Tests failed:\n{failure_summary}",
                    artifacts={
                        "tests_found": True,
                        "test_files": [str(f.relative_to(workspace)) for f in test_files],
                        "test_counts": test_counts,
                        "failures": failure_summary,
                    },
                )

        except subprocess.TimeoutExpired:
            return GuardResult(
                passed=False,
                feedback=f"Tests timed out after {self.timeout} seconds",
                artifacts={"tests_found": True, "timeout": True},
            )
        except FileNotFoundError:
            return GuardResult(
                passed=False,
                feedback="pytest not found. Install with: pip install pytest",
                artifacts={"pytest_installed": False},
            )

    def _parse_test_counts(self, output: str) -> dict[str, int]:
        """Extract test counts from pytest output.

        Args:
            output: pytest stdout/stderr

        Returns:
            Dict with passed, failed, error counts
        """
        counts = {"passed": 0, "failed": 0, "errors": 0}

        # Look for summary line like "5 passed, 2 failed in 1.23s"
        for line in output.split("\n"):
            if "passed" in line.lower() or "failed" in line.lower():
                if " passed" in line:
                    try:
                        counts["passed"] = int(line.split(" passed")[0].split()[-1])
                    except (ValueError, IndexError):
                        pass
                if " failed" in line:
                    try:
                        counts["failed"] = int(line.split(" failed")[0].split()[-1])
                    except (ValueError, IndexError):
                        pass
                if " error" in line:
                    try:
                        counts["errors"] = int(line.split(" error")[0].split()[-1])
                    except (ValueError, IndexError):
                        pass

        return counts

    def _extract_failure_summary(self, output: str) -> str:
        """Extract concise failure summary from pytest output.

        Args:
            output: pytest stdout/stderr

        Returns:
            Concise failure summary (first 500 chars)
        """
        # Look for FAILED lines
        failures = []
        for line in output.split("\n"):
            if line.startswith("FAILED ") or line.startswith("ERROR "):
                failures.append(line)

        if failures:
            summary = "\n".join(failures[:10])  # First 10 failures
        else:
            # Fallback: last 500 chars of output
            summary = output[-500:]

        return summary[:500]  # Truncate to 500 chars
