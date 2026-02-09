"""TDD Green Phase Guard for SWE-Bench Pro.

Validates that a generated test PASSES after the patch is applied.
This ensures the patch actually fixes the bug caught by the test.

Key feature: When edits are present, regenerates the unified diff from
the validated edits to ensure it always applies cleanly (Option 3).
"""

import json
import logging
import re
from pathlib import Path

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..dataset import SWEBenchProInstance
from .diff_utils import regenerate_diff_from_edits
from .quick_test_runner import QuickTestRunner

logger = logging.getLogger("swe_bench_pro.guards.test_green")

# Keywords that indicate diagnostic lines in test output
_DIAGNOSTIC_KEYWORDS = (
    "AssertionError",
    "assert ",
    "FAILED",
    "Error",
    "raise ",
    "TypeError",
    "ValueError",
    "AttributeError",
    "KeyError",
    "ImportError",
    "NameError",
    "expected",
    "actual",
    "!=",
    "not equal",
)


def _extract_diagnostic_lines(output: str, max_lines: int = 15) -> str:
    """Extract the most diagnostic lines from test output.

    Prioritises assertion errors, tracebacks, and FAILED lines
    that tell the generator *what* went wrong rather than boilerplate.

    Args:
        output: Raw test output string.
        max_lines: Maximum number of diagnostic lines to extract.

    Returns:
        Extracted diagnostic lines joined by newlines, or empty string.
    """
    lines = output.splitlines()
    hits: list[str] = []
    for i, line in enumerate(lines):
        if any(kw in line for kw in _DIAGNOSTIC_KEYWORDS):
            # Include surrounding context (1 line before, 2 after)
            start = max(0, i - 1)
            end = min(len(lines), i + 3)
            for ctx_line in lines[start:end]:
                if ctx_line not in hits:
                    hits.append(ctx_line)
        if len(hits) >= max_lines:
            break
    return "\n".join(hits) if hits else ""


class TestGreenGuard(GuardInterface):
    """Validate that test PASSES after patch applied.

    TDD Green Phase: After the fix is applied, the test should pass.
    If the test still fails, the patch doesn't fix the bug.
    """

    def __init__(
        self,
        instance: SWEBenchProInstance,
        dockerhub_username: str = "jefzda",
        timeout_seconds: int = 300,
        repo_root: str | None = None,
        **kwargs,
    ):
        """Initialize the green guard.

        Args:
            instance: The SWE-Bench Pro instance being evaluated.
            dockerhub_username: DockerHub account with pre-built images.
            timeout_seconds: Maximum time for test execution.
            repo_root: Path to the cloned repository (for diff regeneration).
            **kwargs: Ignored (for compatibility with registry pattern).
        """
        self._instance = instance
        self._repo_root = repo_root
        self._runner = QuickTestRunner(
            instance=instance,
            dockerhub_username=dockerhub_username,
            timeout_seconds=timeout_seconds,
        )

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        """Validate that the test passes after patch is applied.

        Args:
            artifact: The patch artifact (JSON with "patch" key).
            **deps: Dependencies - must include the test artifact.
                    Expected key: "ap_gen_test"

        Returns:
            GuardResult with passed=True if test PASSES after patch,
            passed=False if test still fails or errors.
        """
        # Extract test code from dependencies
        test_artifact = deps.get("ap_gen_test")
        if not test_artifact:
            return GuardResult(
                passed=False,
                feedback=(
                    "No test artifact found in dependencies. "
                    "Expected 'ap_gen_test' dependency."
                ),
                guard_name="TestGreenGuard",
            )

        test_code = test_artifact.content.strip()
        if not test_code:
            return GuardResult(
                passed=False,
                feedback="Test code is empty.",
                guard_name="TestGreenGuard",
            )

        # Extract patch from artifact
        try:
            data = json.loads(artifact.content)
            patch_diff = data.get("patch", "")
            edits = data.get("edits", [])
        except json.JSONDecodeError:
            # Maybe the content is the raw patch?
            patch_diff = artifact.content.strip()
            edits = []

        # Regenerate diff from edits if available
        # This ensures the diff always applies cleanly by generating it
        # from the validated search/replace edits
        if edits and self._repo_root:
            regenerated = regenerate_diff_from_edits(edits, self._repo_root)
            if regenerated:
                logger.info(
                    "Regenerated diff from %d edits for %s",
                    len(edits),
                    self._instance.instance_id,
                )
                patch_diff = regenerated

        if not patch_diff:
            return GuardResult(
                passed=False,
                feedback=(
                    "Patch is empty. Generate a patch that fixes the bug.\n"
                    "The patch should modify the source files to make the test pass."
                ),
                guard_name="TestGreenGuard",
            )

        # Check if Docker image is available (will auto-pull if missing)
        available, message = self._runner.ensure_image_available()
        if not available:
            logger.error(
                "Docker image not available for %s - cannot verify patch fixes the bug: %s",
                self._instance.instance_id,
                message,
            )
            return GuardResult(
                passed=False,
                fatal=True,  # ⊥fatal - cannot validate, must escalate
                feedback=(
                    f"FATAL: {message}\n\n"
                    "The guard cannot validate that the patch fixes the bug.\n"
                    "To proceed, either:\n"
                    "1. Ensure Docker is running and you have network access\n"
                    "2. Use a workflow without Docker verification (e.g., s1_tdd)"
                ),
                guard_name="TestGreenGuard",
            )

        # Run test WITH patch
        logger.info(
            "Running green phase verification for %s",
            self._instance.instance_id,
        )
        result = self._runner.run_test(test_code, patch_diff=patch_diff)

        if result.status == "ERROR":
            # Patch couldn't be applied or test errored
            error_msg = result.error_message or "Unknown error"
            raw_output = result.output or ""
            output_snippet = raw_output[-2000:] if raw_output else ""

            # Check if it's a patch application error
            if "patch" in error_msg.lower() or "git apply" in raw_output.lower():
                return GuardResult(
                    passed=False,
                    feedback=(
                        f"Patch could not be applied:\n"
                        f"Target commit: {self._instance.base_commit[:12]}\n"
                        f"Error: {error_msg}\n\n"
                        f"Output (last {len(output_snippet)} chars):\n{output_snippet}\n\n"
                        "Ensure the patch uses correct file paths and matches "
                        "the exact content at the target commit."
                    ),
                    guard_name="TestGreenGuard",
                )

            # Extract diagnostic lines for other errors
            diagnostic_lines = _extract_diagnostic_lines(raw_output)
            diagnostic_section = (
                f"\n## Key error lines\n{diagnostic_lines}\n"
                if diagnostic_lines
                else ""
            )

            return GuardResult(
                passed=False,
                feedback=(
                    f"Test execution failed after patch:\n"
                    f"Error: {error_msg}\n"
                    f"{diagnostic_section}"
                    f"\nOutput (last {len(output_snippet)} chars):\n{output_snippet}\n\n"
                    "The patch may have introduced syntax errors or broken imports."
                ),
                guard_name="TestGreenGuard",
            )

        if result.status == "FAILED":
            # Test still fails after patch - fix didn't work
            raw_output = result.output or ""

            # Build rich, actionable feedback
            parts = [
                f"Test still FAILS after patch (exit code {result.exit_code})."
            ]

            # Extract key diagnostic lines first (most actionable)
            diagnostic_lines = _extract_diagnostic_lines(raw_output)
            if diagnostic_lines:
                parts.append(f"\n## Key failure lines\n{diagnostic_lines}")

            # Add structured failure analysis
            failure_details = self._parse_test_failure(raw_output)
            if failure_details and "Could not parse" not in failure_details:
                parts.append(f"\n{failure_details}")

            # Include broader output (2000 chars for full context)
            output_snippet = raw_output[-2000:] if raw_output else ""
            if output_snippet:
                parts.append(
                    f"\n## Full test output (last {len(output_snippet)} chars)\n"
                    f"{output_snippet}"
                )

            parts.append(
                "\nAnalyze the test failure and adjust the patch to:\n"
                "1. Fix the root cause of the bug, not just symptoms\n"
                "2. Ensure the fix doesn't break other functionality\n"
                "3. Match the expected behavior asserted in the test"
            )

            return GuardResult(
                passed=False,
                feedback="\n".join(parts),
                guard_name="TestGreenGuard",
            )

        # PASSED - the patch fixes the bug
        logger.info(
            "Green phase passed for %s: test passes after patch (%.1fs)",
            self._instance.instance_id,
            result.duration_seconds,
        )
        return GuardResult(
            passed=True,
            feedback=(
                f"Test PASSES after patch (TDD green phase verified).\n"
                f"Duration: {result.duration_seconds}s"
            ),
            guard_name="TestGreenGuard",
        )

    def _parse_test_failure(self, output: str) -> str:
        """Extract structured failure details from pytest output.

        Parses pytest output to find assertion errors, expected vs actual values,
        and returns a structured summary to help the model understand the failure.

        Args:
            output: Raw pytest output string.

        Returns:
            Structured summary of the test failure.
        """
        if not output:
            return "No test output available."

        details: list[str] = []

        # Find assertion error lines
        # Pattern: "AssertionError: ..." or "assert ... == ..."
        assertion_patterns = [
            r"AssertionError:\s*(.+?)(?:\n|$)",
            r"assert\s+(.+?)\s*(?:==|!=|is|in)\s*(.+?)(?:\n|$)",
            r"E\s+assert\s+(.+)",
            r"E\s+AssertionError:\s*(.+)",
        ]

        for pattern in assertion_patterns:
            matches = re.findall(pattern, output, re.MULTILINE)
            if matches:
                for match in matches[:3]:  # Limit to 3 matches
                    if isinstance(match, tuple):
                        details.append(f"ASSERTION: {' '.join(match)}")
                    else:
                        details.append(f"ASSERTION: {match}")
                break

        # Find "expected" vs "actual" patterns (common in pytest output)
        expected_patterns = [
            r"Expected:\s*(.+?)(?:\n|Actual|$)",
            r"expected:\s*(.+?)(?:\n|actual|$)",
            r"E\s+Expected:\s*(.+)",
        ]
        actual_patterns = [
            r"Actual:\s*(.+?)(?:\n|$)",
            r"actual:\s*(.+?)(?:\n|$)",
            r"E\s+Actual:\s*(.+)",
        ]

        for pattern in expected_patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
            if match:
                details.append(f"EXPECTED: {match.group(1).strip()}")
                break

        for pattern in actual_patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
            if match:
                details.append(f"ACTUAL: {match.group(1).strip()}")
                break

        # Find comparison failures (e.g., "1 != 2")
        comparison_pattern = r"E\s+(\S+)\s*(==|!=|<|>|<=|>=|is not|is)\s*(\S+)"
        comparison_matches = re.findall(comparison_pattern, output)
        for match in comparison_matches[:2]:
            left, op, right = match
            details.append(f"COMPARISON: {left} {op} {right}")

        # Find the failed test name
        test_name_patterns = [
            r"FAILED\s+(\S+::\S+)",
            r"(\S+::\S+)\s+FAILED",
            r"test_\w+.*?FAILED",
        ]
        for pattern in test_name_patterns:
            match = re.search(pattern, output)
            if match:
                details.insert(0, f"FAILED TEST: {match.group(1) if match.groups() else match.group(0)}")
                break

        # Find traceback location (last file:line before assertion)
        traceback_pattern = r"(\S+\.py):(\d+).*?\n.*?(?:assert|raise|Error)"
        traceback_matches = re.findall(traceback_pattern, output, re.DOTALL)
        if traceback_matches:
            file, line = traceback_matches[-1]
            details.append(f"LOCATION: {file}:{line}")

        if details:
            return "## Test Failure Analysis\n" + "\n".join(details)

        return "Could not parse specific failure details from test output."
