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

# Default Docker image path pattern (official sweap-images format)
_DOCKER_IMAGE_PATTERN = "{dockerhub_username}/sweap-images:{tag}"


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

        # Build Docker image tag using official SWE-Bench Pro format
        # Format: {repo_base}.{repo_name}-{uid}
        # Example: qutebrowser.qutebrowser-qutebrowser__qutebrowser-f91ace96...
        repo_parts = instance.repo.lower().split("/")
        repo_base = repo_parts[0]
        repo_name = repo_parts[1] if len(repo_parts) > 1 else repo_parts[0]

        # Remove "instance_" prefix if present (as per official eval script)
        uid = instance.instance_id
        if uid.startswith("instance_"):
            uid = uid[9:]

        tag = f"{repo_base}.{repo_name}-{uid}"
        # Docker tags have a 128 character limit; official images are truncated
        self._image_tag = tag[:128]

    def _get_docker_image(self) -> str:
        """Get the Docker image name for this instance."""
        return f"{self._dockerhub_username}/sweap-images:{self._image_tag}"

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
            # --noconftest: Don't load project's conftest.py (may import Qt/Django)
            # -p no:cacheprovider: Avoid cache issues in Docker
            return [
                "python", "-m", "pytest", test_file,
                "-v", "--tb=short", "--noconftest", "-p", "no:cacheprovider"
            ]
        elif lang == "go":
            return ["go", "test", "-v", "-run", "TestAtomicGuard"]
        elif lang in ("javascript", "typescript"):
            return ["npm", "test", "--", test_file]
        else:
            return [
                "python", "-m", "pytest", test_file,
                "-v", "--tb=short", "--noconftest", "-p", "no:cacheprovider"
            ]

    def _create_test_script(
        self,
        test_code: str,
        patch_diff: str | None = None,
    ) -> str:
        """Create a shell script to run inside Docker.

        The script:
        1. Resets the repo to base_commit
        2. Verifies the checkout succeeded
        3. Optionally applies the patch
        4. Writes the test file
        5. Runs the test
        """
        test_filename = self._get_test_filename()
        test_cmd = " ".join(self._get_test_command())
        expected_commit = self._instance.base_commit

        # Note: Using quoted heredoc delimiters (<< 'EOF') means no escaping needed.
        # The content is passed through literally without shell expansion.

        script_lines = [
            "#!/bin/sh",
            "set -e",
            "",
            "# Fetch and reset to base commit",
            f"git fetch --depth=1 origin {expected_commit} 2>/dev/null || true",
            f"git checkout -f {expected_commit}",
            "git clean -fdx",
            "",
            "# Verify commit",
            "ACTUAL_COMMIT=$(git rev-parse HEAD)",
            f'EXPECTED_PREFIX="{expected_commit[:12]}"',
            'if [ "${ACTUAL_COMMIT#$EXPECTED_PREFIX}" = "$ACTUAL_COMMIT" ]; then',
            f'    echo "ERROR: Commit mismatch. Expected {expected_commit[:12]}..., got $ACTUAL_COMMIT"',
            "    exit 1",
            "fi",
            "",
        ]

        if patch_diff:
            # Using quoted heredoc delimiter - no escaping needed
            # --whitespace=fix auto-corrects trailing whitespace issues
            script_lines.extend([
                "# Apply patch",
                "cat << 'PATCH_EOF' | git apply -v --whitespace=fix -",
                patch_diff,
                "PATCH_EOF",
                "",
            ])

        script_lines.extend([
            "# Write test file",
            f"cat << 'TEST_EOF' > {test_filename}",
            test_code,
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
            # Use --entrypoint to ensure shell works (some images have broken default shells)
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--network=none",  # Block network for security
                "--entrypoint", "/bin/sh",
                "-v", f"{script_path}:/run_test.sh:ro",
                image,
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
