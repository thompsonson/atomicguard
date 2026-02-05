"""TDD Green Phase Guard for SWE-Bench Pro.

Validates that a generated test PASSES after the patch is applied.
This ensures the patch actually fixes the bug caught by the test.
"""

import json
import logging

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
        **kwargs,
    ):
        """Initialize the green guard.

        Args:
            instance: The SWE-Bench Pro instance being evaluated.
            dockerhub_username: DockerHub account with pre-built images.
            timeout_seconds: Maximum time for test execution.
            **kwargs: Ignored (for compatibility with registry pattern).
        """
        self._instance = instance
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
        except json.JSONDecodeError:
            # Maybe the content is the raw patch?
            patch_diff = artifact.content.strip()

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
                        f"Error: {error_msg}\n"
                        f"Output:\n{output_snippet}\n\n"
                        "Ensure the patch uses correct file paths and matches "
                        "the exact content in the repository."
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
            return GuardResult(
                passed=False,
                feedback=(
                    "Test still FAILS after patch - the fix didn't work!\n\n"
                    f"Test output:\n{output_snippet}\n\n"
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
