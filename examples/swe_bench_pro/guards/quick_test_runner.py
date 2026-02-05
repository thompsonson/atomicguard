"""Quick test runner for SWE-Bench Pro TDD verification.

Runs a single generated test in a Docker container to verify:
- Test FAILS on buggy code (TDD red phase)
- Test PASSES after patch applied (TDD green phase)
"""

import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..dataset import SWEBenchProInstance

logger = logging.getLogger("swe_bench_pro.guards.quick_test")

# Default Docker image path pattern
_DOCKER_IMAGE_PATTERN = "{dockerhub_username}/swe-bench-pro:{instance_id}"


@dataclass(frozen=True)
class QuickTestResult:
    """Result from running a single test in Docker."""

    status: Literal["PASSED", "FAILED", "ERROR"]
    output: str
    exit_code: int
    duration_seconds: float
    error_message: str = ""


class QuickTestRunner:
    """Run a single generated test in Docker.

    Uses the pre-built SWE-Bench Pro Docker images to execute
    tests in the exact environment expected by the evaluation.
    """

    def __init__(
        self,
        instance: SWEBenchProInstance,
        dockerhub_username: str = "jefzda",
        timeout_seconds: int = 300,
        cache_dir: str = "~/.cache/swe_bench_pro",
    ):
        """Initialize the test runner.

        Args:
            instance: The SWE-Bench Pro instance being evaluated.
            dockerhub_username: DockerHub account with pre-built images.
            timeout_seconds: Maximum time for test execution.
            cache_dir: Directory containing evaluation infrastructure.
        """
        self._instance = instance
        self._dockerhub_username = dockerhub_username
        self._timeout = timeout_seconds
        self._cache_dir = Path(cache_dir).expanduser()

        # Normalize instance ID for Docker image tag
        # e.g., "django/django-1234" -> "django__django-1234"
        self._image_tag = instance.instance_id.replace("/", "__")

    def _get_docker_image(self) -> str:
        """Get the Docker image name for this instance."""
        return f"{self._dockerhub_username}/swe-bench-pro:{self._image_tag}"

    def _get_test_filename(self) -> str:
        """Get the appropriate test filename based on language."""
        lang = self._instance.repo_language.lower()
        if lang == "python":
            return "test_atomicguard.py"
        elif lang == "go":
            return "atomicguard_test.go"
        elif lang in ("javascript", "typescript"):
            return "atomicguard.test.js"
        else:
            return "test_atomicguard.py"

    def _get_test_command(self) -> list[str]:
        """Get the test execution command based on language."""
        lang = self._instance.repo_language.lower()
        test_file = self._get_test_filename()

        if lang == "python":
            return ["python", "-m", "pytest", test_file, "-v", "--tb=short"]
        elif lang == "go":
            return ["go", "test", "-v", "-run", "TestAtomicGuard"]
        elif lang in ("javascript", "typescript"):
            return ["npm", "test", "--", test_file]
        else:
            return ["python", "-m", "pytest", test_file, "-v", "--tb=short"]

    def _create_test_script(
        self,
        test_code: str,
        patch_diff: str | None = None,
    ) -> str:
        """Create a shell script to run inside Docker.

        The script:
        1. Resets the repo to base_commit
        2. Optionally applies the patch
        3. Writes the test file
        4. Runs the test
        """
        test_filename = self._get_test_filename()
        test_cmd = " ".join(self._get_test_command())

        # Escape test code for heredoc
        escaped_test = test_code.replace("'", "'\"'\"'")

        script_lines = [
            "#!/bin/bash",
            "set -e",
            "",
            "# Reset to base commit",
            f"git checkout -f {self._instance.base_commit}",
            "git clean -fdx",
            "",
        ]

        if patch_diff:
            # Escape patch for heredoc
            escaped_patch = patch_diff.replace("'", "'\"'\"'")
            script_lines.extend([
                "# Apply patch",
                "cat << 'PATCH_EOF' | git apply -",
                escaped_patch,
                "PATCH_EOF",
                "",
            ])

        script_lines.extend([
            "# Write test file",
            f"cat << 'TEST_EOF' > {test_filename}",
            escaped_test,
            "TEST_EOF",
            "",
            "# Run test",
            f"exec {test_cmd}",
        ])

        return "\n".join(script_lines)

    def run_test(
        self,
        test_code: str,
        patch_diff: str | None = None,
    ) -> QuickTestResult:
        """Run a test in Docker.

        Args:
            test_code: The test source code to execute.
            patch_diff: Optional patch to apply before running the test.

        Returns:
            QuickTestResult with status, output, and timing information.
        """
        start_time = time.time()
        image = self._get_docker_image()

        # Create temporary script file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            delete=False,
        ) as f:
            script = self._create_test_script(test_code, patch_diff)
            f.write(script)
            script_path = f.name

        try:
            # Make script executable
            Path(script_path).chmod(0o755)

            # Run in Docker
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--network=none",  # Block network for security
                "-v", f"{script_path}:/run_test.sh:ro",
                image,
                "/bin/bash",
                "/run_test.sh",
            ]

            logger.debug("Running Docker command: %s", " ".join(docker_cmd))

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )

            duration = time.time() - start_time
            output = result.stdout + result.stderr

            # Determine status based on exit code
            if result.returncode == 0:
                status = "PASSED"
            else:
                # Check if it's a test failure or an error
                if "FAILED" in output or "AssertionError" in output or "Error" in output:
                    status = "FAILED"
                else:
                    status = "ERROR"

            return QuickTestResult(
                status=status,
                output=output,
                exit_code=result.returncode,
                duration_seconds=round(duration, 2),
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return QuickTestResult(
                status="ERROR",
                output="",
                exit_code=-1,
                duration_seconds=round(duration, 2),
                error_message=f"Test execution timed out after {self._timeout}s",
            )

        except FileNotFoundError:
            duration = time.time() - start_time
            return QuickTestResult(
                status="ERROR",
                output="",
                exit_code=-1,
                duration_seconds=round(duration, 2),
                error_message="Docker not found. Ensure Docker is installed and running.",
            )

        except Exception as e:
            duration = time.time() - start_time
            return QuickTestResult(
                status="ERROR",
                output="",
                exit_code=-1,
                duration_seconds=round(duration, 2),
                error_message=f"Unexpected error: {e}",
            )

        finally:
            # Clean up temp file
            try:
                Path(script_path).unlink()
            except OSError:
                pass

    def check_image_available(self) -> bool:
        """Check if the Docker image is available locally.

        Returns:
            True if the image exists, False otherwise.
        """
        image = self._get_docker_image()
        return self._check_image_local(image)

    def _check_image_local(self, image: str) -> bool:
        """Check if image exists locally (no pull).

        Args:
            image: The Docker image name to check.

        Returns:
            True if the image exists locally, False otherwise.
        """
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def ensure_image_available(self, pull_if_missing: bool = True) -> tuple[bool, str]:
        """Ensure Docker image is available, pulling if necessary.

        Args:
            pull_if_missing: If True, attempt to pull image when not found locally.

        Returns:
            Tuple of (available: bool, message: str)
        """
        image = self._get_docker_image()

        # Fast path: check local
        if self._check_image_local(image):
            return True, f"Image available locally: {image}"

        if not pull_if_missing:
            return False, f"Image not found locally: {image}"

        # Attempt to pull
        logger.info("Pulling Docker image: %s", image)
        try:
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for large images
            )
            if result.returncode == 0:
                return True, f"Successfully pulled: {image}"
            else:
                # Pass through Docker's error message
                error_msg = result.stderr.strip() or result.stdout.strip()
                return False, error_msg or f"Failed to pull {image}"
        except subprocess.TimeoutExpired:
            return False, f"Timeout pulling image: {image}"
        except FileNotFoundError:
            return False, "Docker not installed or not in PATH"
