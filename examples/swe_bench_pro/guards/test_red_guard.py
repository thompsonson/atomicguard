"""TDD Red Phase Guard for SWE-Bench Pro.

Validates that a generated test FAILS on the buggy (base) code.
This ensures the test actually captures the bug before we attempt
to fix it.
"""

import logging

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..dataset import SWEBenchProInstance
from .quick_test_runner import QuickTestRunner
from .test_green_guard import _extract_diagnostic_lines

logger = logging.getLogger("swe_bench_pro.guards.test_red")


class TestRedGuard(GuardInterface):
    """Validate that test FAILS on base (buggy) code.

    TDD Red Phase: The test should fail before the fix is applied.
    If the test passes on buggy code, it doesn't capture the bug
    and needs to be regenerated.
    """

    def __init__(
        self,
        instance: SWEBenchProInstance,
        dockerhub_username: str = "jefzda",
        timeout_seconds: int = 300,
        **kwargs,
    ):
        """Initialize the red guard.

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
        """Validate that the test fails on buggy code.

        Args:
            artifact: The test code artifact to validate.
            **deps: Dependencies (not used for red phase).

        Returns:
            GuardResult with passed=True if test FAILS (expected),
            passed=False if test passes (unexpected) or errors.
        """
        test_code = artifact.content.strip()

        if not test_code:
            return GuardResult(
                passed=False,
                feedback="Test code is empty. Generate a test that reproduces the bug.",
                guard_name="TestRedGuard",
            )

        # Check if Docker image is available (will auto-pull if missing)
        available, message = self._runner.ensure_image_available()
        if not available:
            logger.error(
                "Docker image not available for %s - cannot verify test fails on buggy code: %s",
                self._instance.instance_id,
                message,
            )
            return GuardResult(
                passed=False,
                fatal=True,  # ⊥fatal - cannot validate, must escalate
                feedback=(
                    f"FATAL: {message}\n\n"
                    "The guard cannot validate that the test correctly fails on buggy code.\n"
                    "To proceed, either:\n"
                    "1. Ensure Docker is running and you have network access\n"
                    "2. Use a workflow without Docker verification (e.g., s1_tdd)"
                ),
                guard_name="TestRedGuard",
            )

        # Run test WITHOUT patch (on buggy code)
        logger.info(
            "Running red phase verification for %s",
            self._instance.instance_id,
        )
        result = self._runner.run_test(test_code, patch_diff=None)

        if result.status == "ERROR":
            # Test has syntax errors or couldn't run
            error_msg = result.error_message or "Unknown error"
            raw_output = result.output or ""
            output_snippet = raw_output[-2000:] if raw_output else ""

            # Extract diagnostic lines
            diagnostic_lines = _extract_diagnostic_lines(raw_output)
            diagnostic_section = (
                f"\n## Key error lines\n{diagnostic_lines}\n"
                if diagnostic_lines
                else ""
            )

            return GuardResult(
                passed=False,
                feedback=(
                    f"Test has errors and couldn't execute:\n"
                    f"Error: {error_msg}\n"
                    f"{diagnostic_section}"
                    f"\nOutput (last {len(output_snippet)} chars):\n{output_snippet}\n\n"
                    "Fix the test syntax and ensure it can be imported."
                ),
                guard_name="TestRedGuard",
            )

        if result.status == "PASSED":
            # Test passes on buggy code - it doesn't capture the bug!
            raw_output = result.output or ""
            output_snippet = raw_output[-1500:] if raw_output else ""
            return GuardResult(
                passed=False,
                feedback=(
                    "Test PASSES on buggy code - it doesn't capture the bug!\n\n"
                    "WHY THIS HAPPENS:\n"
                    "- Your test asserts something TRUE even in buggy code\n"
                    "- Common mistakes: testing existence, types, or imports\n\n"
                    "HOW TO FIX:\n"
                    "1. Identify INPUTS that trigger the bug\n"
                    "2. Call the buggy function with those inputs\n"
                    "3. Assert the EXPECTED OUTPUT (what it SHOULD return)\n"
                    "4. This will FAIL because buggy code returns something WRONG\n\n"
                    f"Test output (last {len(output_snippet)} chars):\n{output_snippet}\n\n"
                    "Example pattern:\n"
                    "  result = buggy_function(trigger_input)\n"
                    "  assert result == expected_correct_value"
                ),
                guard_name="TestRedGuard",
            )

        # FAILED is the expected status - test correctly catches the bug
        logger.info(
            "Red phase passed for %s: test correctly fails on buggy code (%.1fs)",
            self._instance.instance_id,
            result.duration_seconds,
        )
        return GuardResult(
            passed=True,
            feedback=(
                f"Test correctly FAILS on buggy code (TDD red phase verified).\n"
                f"Duration: {result.duration_seconds}s"
            ),
            guard_name="TestRedGuard",
        )
