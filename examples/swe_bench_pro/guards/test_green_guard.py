"""TDD Green Phase Guard for SWE-Bench Pro.

Validates that a generated test PASSES after the patch is applied.
This ensures the patch actually fixes the bug caught by the test.

Key feature: When edits are present, regenerates the unified diff from
the validated edits to ensure it always applies cleanly (Option 3).
"""

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..dataset import SWEBenchProInstance
from .quick_test_runner import QuickTestRunner

logger = logging.getLogger("swe_bench_pro.guards.test_green")


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

        # Option 3: Regenerate diff from edits if available
        # This ensures the diff always applies cleanly by generating it
        # from the validated search/replace edits
        if edits and self._repo_root:
            regenerated = self._regenerate_diff_from_edits(
                edits, self._repo_root
            )
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
            output_snippet = result.output[-800:] if result.output else ""

            # Check if it's a patch application error
            if "patch" in error_msg.lower() or "git apply" in output_snippet.lower():
                return GuardResult(
                    passed=False,
                    feedback=(
                        f"Patch could not be applied:\n"
                        f"Target commit: {self._instance.base_commit[:12]}\n"
                        f"Error: {error_msg}\n"
                        f"Output:\n{output_snippet}\n\n"
                        "Ensure the patch uses correct file paths and matches "
                        "the exact content at the target commit."
                    ),
                    guard_name="TestGreenGuard",
                )

            return GuardResult(
                passed=False,
                feedback=(
                    f"Test execution failed after patch:\n"
                    f"Error: {error_msg}\n"
                    f"Output:\n{output_snippet}\n\n"
                    "The patch may have introduced syntax errors or broken imports."
                ),
                guard_name="TestGreenGuard",
            )

        if result.status == "FAILED":
            # Test still fails after patch - fix didn't work
            output_snippet = result.output[-800:] if result.output else ""
            # Extract structured failure details for better feedback
            failure_details = self._parse_test_failure(result.output or "")
            return GuardResult(
                passed=False,
                feedback=(
                    "Test still FAILS after patch - the fix didn't work!\n\n"
                    f"{failure_details}\n\n"
                    f"Full test output (last 800 chars):\n{output_snippet}\n\n"
                    "Analyze the test failure and adjust the patch to:\n"
                    "1. Fix the root cause of the bug, not just symptoms\n"
                    "2. Ensure the fix doesn't break other functionality\n"
                    "3. Match the expected behavior asserted in the test"
                ),
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

    def _regenerate_diff_from_edits(
        self,
        edits: list[dict[str, str]],
        repo_root: str,
    ) -> str | None:
        """Regenerate unified diff from validated search/replace edits.

        This ensures the diff always applies cleanly by:
        1. Copying modified files to a temp directory
        2. Applying edits via search/replace
        3. Generating a fresh git diff

        Args:
            edits: List of {"file": path, "search": str, "replace": str}
            repo_root: Path to the repository

        Returns:
            Unified diff string, or None if regeneration failed.
        """
        if not edits:
            return None

        repo_path = Path(repo_root)
        temp_dir = None

        try:
            # Create temp directory for modified files
            temp_dir = tempfile.mkdtemp(prefix="atomicguard_diff_")
            temp_path = Path(temp_dir)

            # Track which files we modify
            modified_files: set[str] = set()

            for edit in edits:
                file_path = edit.get("file", "")
                search = edit.get("search", "")
                replace = edit.get("replace", "")

                if not file_path or not search:
                    continue

                src_file = repo_path / file_path
                if not src_file.exists():
                    logger.warning(
                        "File not found for diff regeneration: %s", file_path
                    )
                    continue

                # Read original content
                try:
                    original = src_file.read_text()
                except Exception as e:
                    logger.warning("Failed to read %s: %s", file_path, e)
                    continue

                # Check if search string exists
                if search not in original:
                    logger.warning(
                        "Search string not found in %s during diff regeneration",
                        file_path,
                    )
                    continue

                # Apply the edit
                modified = original.replace(search, replace, 1)

                # Write to temp location (preserving directory structure)
                temp_file = temp_path / file_path
                temp_file.parent.mkdir(parents=True, exist_ok=True)
                temp_file.write_text(modified)

                # Also copy original for git diff
                orig_dir = temp_path / "orig"
                orig_file = orig_dir / file_path
                orig_file.parent.mkdir(parents=True, exist_ok=True)
                orig_file.write_text(original)

                modified_files.add(file_path)

            if not modified_files:
                return None

            # Generate unified diff for each modified file
            diff_parts = []
            for file_path in sorted(modified_files):
                orig_file = temp_path / "orig" / file_path
                new_file = temp_path / file_path

                try:
                    result = subprocess.run(
                        [
                            "diff",
                            "-u",
                            f"--label=a/{file_path}",
                            f"--label=b/{file_path}",
                            str(orig_file),
                            str(new_file),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    # diff returns 1 when files differ (which is expected)
                    if result.returncode in (0, 1) and result.stdout:
                        diff_parts.append(result.stdout)
                except Exception as e:
                    logger.warning("Failed to generate diff for %s: %s", file_path, e)

            if diff_parts:
                return "\n".join(diff_parts)
            return None

        except Exception as e:
            logger.warning("Failed to regenerate diff from edits: %s", e)
            return None

        finally:
            # Clean up temp directory
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
