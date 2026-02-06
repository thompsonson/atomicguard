"""Docker execution guards for Arms 05/06 (CompositeGuard with verification).

These guards implement the G_ver (verification) phase of the two-phase
validation model from the design document. They require Docker execution
inside the target instance's environment.

Guard catalogue:
- TestRedGuard:   Test FAILS on buggy code (confirms test discriminates)
- TestGreenGuard: Test PASSES after applying patch (confirms fix works)
- FullEvalGuard:  Full test suite passes after patch (regression check)
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

logger = logging.getLogger("swe_bench_ablation.guards")


class TestRedGuard(GuardInterface):
    """Verification guard: test must FAIL on buggy code (base_commit).

    Runs the generated test inside the instance's Docker container against
    the unmodified (buggy) codebase. If the test passes, it is a
    false-positive test that doesn't discriminate — rejected with feedback.

    This is the "red" phase of red-green-refactor TDD.
    """

    def __init__(
        self,
        docker_image: str = "",
        test_command: str = "",
        repo_root: str | None = None,
        timeout_seconds: int = 120,
        **kwargs: Any,  # noqa: ARG002
    ):
        self._docker_image = docker_image
        self._test_command = test_command
        self._repo_root = repo_root
        self._timeout = timeout_seconds

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate that the test fails on buggy code.

        Args:
            artifact: Test code artifact
            **deps: Unused dependency artifacts

        Returns:
            GuardResult — passed=True if test FAILS on buggy code
        """
        logger.info("[TestRedGuard] Verifying test fails on buggy code...")

        if not self._docker_image:
            return GuardResult(
                passed=False,
                feedback=(
                    "TestRedGuard requires docker_image configuration. "
                    "Set docker_image in guard_config or instance metadata."
                ),
                guard_name="TestRedGuard",
            )

        test_code = artifact.content.strip()
        if not test_code:
            return GuardResult(
                passed=False,
                feedback="Empty test code",
                guard_name="TestRedGuard",
            )

        try:
            exit_code, stdout, stderr = self._run_test_in_docker(test_code)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Docker execution error: {e}",
                guard_name="TestRedGuard",
            )

        if exit_code != 0:
            # Test failed on buggy code — this is what we want
            logger.info("[TestRedGuard] PASSED: test fails on buggy code (exit=%d)", exit_code)
            return GuardResult(
                passed=True,
                feedback=f"Test correctly fails on buggy code (exit code {exit_code})",
                guard_name="TestRedGuard",
            )
        else:
            # Test passed on buggy code — false positive
            logger.info("[TestRedGuard] REJECTED: test passes on buggy code")
            output_snippet = (stdout or stderr)[:500]
            return GuardResult(
                passed=False,
                feedback=(
                    "Test PASSED on buggy code — your test doesn't discriminate. "
                    "The test must FAIL on the unmodified code to confirm it "
                    "actually detects the bug.\n\n"
                    f"Test output:\n{output_snippet}"
                ),
                guard_name="TestRedGuard",
            )

    def _run_test_in_docker(self, test_code: str) -> tuple[int, str, str]:
        """Run test code inside the Docker container.

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_test.py", delete=False
        ) as f:
            f.write(test_code)
            test_file = f.name

        try:
            test_cmd = self._test_command or f"python -m pytest {test_file} -x -v"

            cmd = [
                "docker", "run", "--rm",
                "-v", f"{test_file}:/tmp/generated_test.py:ro",
            ]
            if self._repo_root:
                cmd.extend(["-v", f"{self._repo_root}:/workspace:ro"])
                cmd.extend(["-w", "/workspace"])

            cmd.extend([self._docker_image, "bash", "-c", test_cmd])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Docker test execution timed out after {self._timeout}s"
            )
        finally:
            Path(test_file).unlink(missing_ok=True)


class TestGreenGuard(GuardInterface):
    """Verification guard: test must PASS after applying the patch.

    Applies the generated patch to the codebase, then runs the generated
    test inside Docker. If the test still fails, the patch doesn't fix
    the bug — rejected with the test failure output.

    This is the "green" phase of red-green-refactor TDD.
    """

    def __init__(
        self,
        docker_image: str = "",
        test_command: str = "",
        repo_root: str | None = None,
        timeout_seconds: int = 120,
        **kwargs: Any,  # noqa: ARG002
    ):
        self._docker_image = docker_image
        self._test_command = test_command
        self._repo_root = repo_root
        self._timeout = timeout_seconds

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,
    ) -> GuardResult:
        """Validate that the test passes after applying the patch.

        Args:
            artifact: Patch artifact (JSON with patch/edits)
            **deps: Must include ap_gen_test artifact

        Returns:
            GuardResult — passed=True if test PASSES after patch
        """
        logger.info("[TestGreenGuard] Verifying test passes after patch...")

        if not self._docker_image:
            return GuardResult(
                passed=False,
                feedback=(
                    "TestGreenGuard requires docker_image configuration. "
                    "Set docker_image in guard_config or instance metadata."
                ),
                guard_name="TestGreenGuard",
            )

        # Get test artifact from dependencies
        test_artifact = deps.get("ap_gen_test")
        if not test_artifact:
            return GuardResult(
                passed=False,
                feedback="TestGreenGuard requires ap_gen_test dependency",
                guard_name="TestGreenGuard",
            )

        # Extract patch content
        try:
            data = json.loads(artifact.content)
            patch_content = data.get("patch", "")
        except (json.JSONDecodeError, TypeError):
            patch_content = artifact.content

        if not patch_content:
            return GuardResult(
                passed=False,
                feedback="No patch content to apply",
                guard_name="TestGreenGuard",
            )

        test_code = test_artifact.content.strip()

        try:
            exit_code, stdout, stderr = self._run_test_with_patch(
                test_code, patch_content
            )
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Docker execution error: {e}",
                guard_name="TestGreenGuard",
            )

        if exit_code == 0:
            logger.info("[TestGreenGuard] PASSED: test passes after patch")
            return GuardResult(
                passed=True,
                feedback="Test passes after applying patch (green phase confirmed)",
                guard_name="TestGreenGuard",
            )
        else:
            output_snippet = (stdout or stderr)[:500]
            logger.info("[TestGreenGuard] REJECTED: test still fails after patch")
            return GuardResult(
                passed=False,
                feedback=(
                    "Test still FAILS after applying patch. "
                    "The fix does not resolve the bug detected by the test.\n\n"
                    f"Test output:\n{output_snippet}"
                ),
                guard_name="TestGreenGuard",
            )

    def _run_test_with_patch(
        self, test_code: str, patch_content: str
    ) -> tuple[int, str, str]:
        """Apply patch and run test inside Docker.

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_test.py", delete=False
        ) as tf:
            tf.write(test_code)
            test_file = tf.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False
        ) as pf:
            pf.write(patch_content)
            patch_file = pf.name

        try:
            test_cmd = (
                f"git apply /tmp/generated_patch.patch && "
                f"{self._test_command or 'python -m pytest /tmp/generated_test.py -x -v'}"
            )

            cmd = [
                "docker", "run", "--rm",
                "-v", f"{test_file}:/tmp/generated_test.py:ro",
                "-v", f"{patch_file}:/tmp/generated_patch.patch:ro",
            ]
            if self._repo_root:
                cmd.extend(["-v", f"{self._repo_root}:/workspace"])
                cmd.extend(["-w", "/workspace"])

            cmd.extend([self._docker_image, "bash", "-c", test_cmd])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Docker test execution timed out after {self._timeout}s"
            )
        finally:
            Path(test_file).unlink(missing_ok=True)
            Path(patch_file).unlink(missing_ok=True)


class FullEvalGuard(GuardInterface):
    """Verification guard: full test suite must pass after patch (regression).

    Applies the generated patch, then runs the full evaluation suite
    (fail_to_pass + pass_to_pass) inside Docker. Catches regressions
    that the generated test alone cannot detect.
    """

    def __init__(
        self,
        docker_image: str = "",
        eval_command: str = "",
        repo_root: str | None = None,
        timeout_seconds: int = 300,
        **kwargs: Any,  # noqa: ARG002
    ):
        self._docker_image = docker_image
        self._eval_command = eval_command
        self._repo_root = repo_root
        self._timeout = timeout_seconds

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate that full test suite passes after patch.

        Args:
            artifact: Patch artifact (JSON with patch/edits)
            **deps: Unused dependency artifacts

        Returns:
            GuardResult — passed=True if full eval passes
        """
        logger.info("[FullEvalGuard] Running full evaluation suite...")

        if not self._docker_image:
            return GuardResult(
                passed=False,
                feedback=(
                    "FullEvalGuard requires docker_image configuration. "
                    "Set docker_image in guard_config or instance metadata."
                ),
                guard_name="FullEvalGuard",
            )

        # Extract patch content
        try:
            data = json.loads(artifact.content)
            patch_content = data.get("patch", "")
        except (json.JSONDecodeError, TypeError):
            patch_content = artifact.content

        if not patch_content:
            return GuardResult(
                passed=False,
                feedback="No patch content to evaluate",
                guard_name="FullEvalGuard",
            )

        try:
            exit_code, stdout, stderr = self._run_full_eval(patch_content)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Docker evaluation error: {e}",
                guard_name="FullEvalGuard",
            )

        if exit_code == 0:
            logger.info("[FullEvalGuard] PASSED: full evaluation suite passes")
            return GuardResult(
                passed=True,
                feedback="Full evaluation suite passes (no regressions detected)",
                guard_name="FullEvalGuard",
            )
        else:
            output_snippet = (stdout or stderr)[:800]
            logger.info("[FullEvalGuard] REJECTED: evaluation failures detected")
            return GuardResult(
                passed=False,
                feedback=(
                    "Full evaluation suite FAILED after applying patch. "
                    "Regressions detected in existing tests.\n\n"
                    f"Evaluation output:\n{output_snippet}"
                ),
                guard_name="FullEvalGuard",
            )

    def _run_full_eval(self, patch_content: str) -> tuple[int, str, str]:
        """Apply patch and run full eval inside Docker.

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False
        ) as pf:
            pf.write(patch_content)
            patch_file = pf.name

        try:
            eval_cmd = (
                f"git apply /tmp/generated_patch.patch && "
                f"{self._eval_command or 'python -m pytest -x'}"
            )

            cmd = [
                "docker", "run", "--rm",
                "-v", f"{patch_file}:/tmp/generated_patch.patch:ro",
            ]
            if self._repo_root:
                cmd.extend(["-v", f"{self._repo_root}:/workspace"])
                cmd.extend(["-w", "/workspace"])

            cmd.extend([self._docker_image, "bash", "-c", eval_cmd])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Docker evaluation timed out after {self._timeout}s"
            )
        finally:
            Path(patch_file).unlink(missing_ok=True)
