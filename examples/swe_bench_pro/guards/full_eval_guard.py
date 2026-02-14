"""Full Evaluation Guard for SWE-Bench Pro.

Runs the complete SWE-Bench Pro evaluation suite to verify:
- FAIL_TO_PASS tests now pass
- PASS_TO_PASS tests still pass (no regressions)
"""

import contextlib
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..dataset import SWEBenchProInstance
from ..evaluation import ensure_eval_repo
from .quick_test_runner import QuickTestRunner

logger = logging.getLogger("swe_bench_pro.guards.full_eval")


class FullEvalGuard(GuardInterface):
    """Run full SWE-Bench Pro evaluation suite.

    This guard runs the official evaluation script to verify:
    1. All FAIL_TO_PASS tests now pass with the patch
    2. All PASS_TO_PASS tests still pass (no regressions)
    """

    def __init__(
        self,
        instance: SWEBenchProInstance,
        dockerhub_username: str = "jefzda",
        timeout_seconds: int = 1800,  # 30 minutes for full eval
        cache_dir: str = "~/.cache/swe_bench_pro",
        **kwargs,  # noqa: ARG002
    ):
        """Initialize the full eval guard.

        Args:
            instance: The SWE-Bench Pro instance being evaluated.
            dockerhub_username: DockerHub account with pre-built images.
            timeout_seconds: Maximum time for evaluation.
            cache_dir: Directory for evaluation infrastructure.
            **kwargs: Ignored (for compatibility with registry pattern).
        """
        self._instance = instance
        self._dockerhub_username = dockerhub_username
        self._timeout = timeout_seconds
        self._cache_dir = Path(cache_dir).expanduser()
        # Use QuickTestRunner to check Docker image availability
        self._docker_checker = QuickTestRunner(
            instance=instance,
            dockerhub_username=dockerhub_username,
        )

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:  # noqa: ARG002
        """Run full evaluation on the patch.

        Args:
            artifact: The patch artifact (JSON with "patch" key).
            **deps: Dependencies (not used directly).

        Returns:
            GuardResult with passed=True if all tests pass,
            passed=False with detailed feedback on failures.
        """
        # Check if Docker image is available (will auto-pull if missing)
        available, message = self._docker_checker.ensure_image_available()
        if not available:
            logger.error(
                "Docker image not available for %s - cannot run full evaluation: %s",
                self._instance.instance_id,
                message,
            )
            return GuardResult(
                passed=False,
                fatal=True,  # ⊥fatal - cannot validate, must escalate
                feedback=(
                    f"FATAL: {message}\n\n"
                    "The guard cannot validate the patch against the full test suite.\n"
                    "To proceed, either:\n"
                    "1. Ensure Docker is running and you have network access\n"
                    "2. Use a workflow without full evaluation"
                ),
                guard_name="FullEvalGuard",
            )

        # Extract patch from artifact
        try:
            data = json.loads(artifact.content)
            patch_diff = data.get("patch", "")
        except json.JSONDecodeError:
            patch_diff = artifact.content.strip()

        if not patch_diff:
            return GuardResult(
                passed=False,
                feedback="Patch is empty. Cannot run evaluation without a patch.",
                guard_name="FullEvalGuard",
            )

        # Ensure eval repo is available
        try:
            eval_repo = ensure_eval_repo(cache_dir=str(self._cache_dir))
        except Exception as e:
            logger.error("Failed to set up eval repo: %s", e)
            return GuardResult(
                passed=False,
                fatal=True,  # ⊥fatal - cannot validate, must escalate
                feedback=(
                    f"FATAL: Cannot run full evaluation - eval repo setup failed.\n"
                    f"Error: {e}\n\n"
                    "The guard cannot validate the patch against the full test suite.\n"
                    "To proceed, ensure the SWE-Bench Pro eval repo can be cloned."
                ),
                guard_name="FullEvalGuard",
            )

        # Create temporary prediction file for single instance
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            prediction = {
                "instance_id": self._instance.instance_id,
                "patch": patch_diff,
                "prefix": "atomicguard-tdd-verified",
            }
            json.dump([prediction], f)
            pred_path = Path(f.name)

        try:
            result = self._run_evaluation(eval_repo, pred_path)
            return result
        finally:
            with contextlib.suppress(OSError):
                pred_path.unlink()

    def _run_evaluation(
        self,
        eval_repo: Path,
        pred_path: Path,
    ) -> GuardResult:
        """Run the official evaluation script.

        Args:
            eval_repo: Path to the SWE-Bench Pro eval repository.
            pred_path: Path to the prediction JSON file.

        Returns:
            GuardResult based on evaluation outcome.
        """
        eval_script = eval_repo / "swe_bench_pro_eval.py"
        dataset_csv = eval_repo / "swe_bench_pro_full.csv"

        if not eval_script.exists():
            return GuardResult(
                passed=False,
                fatal=True,  # ⊥fatal - cannot validate, must escalate
                feedback=(
                    f"FATAL: Cannot run full evaluation - eval script not found.\n"
                    f"Expected: {eval_script}\n\n"
                    "The guard cannot validate the patch against the full test suite.\n"
                    "Ensure the SWE-Bench Pro eval repo is properly set up."
                ),
                guard_name="FullEvalGuard",
            )

        # Create temp output directory
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)

            cmd = [
                sys.executable,
                str(eval_script),
                f"--raw_sample_path={dataset_csv}",
                f"--patch_path={pred_path}",
                f"--output_dir={output_path}",
                f"--scripts_dir={eval_repo / 'run_scripts'}",
                "--num_workers=1",
                f"--dockerhub_username={self._dockerhub_username}",
                "--use_local_docker",
                "--block_network",
            ]

            logger.info(
                "Running full eval for %s",
                self._instance.instance_id,
            )

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )

                # Parse results
                results_file = output_path / "eval_results.json"
                if results_file.exists():
                    eval_results = json.loads(results_file.read_text())
                    return self._parse_eval_results(
                        eval_results, result.stdout + result.stderr
                    )

                # No results file - check for errors
                if result.returncode != 0:
                    output_snippet = (result.stdout + result.stderr)[-1000:]
                    return GuardResult(
                        passed=False,
                        feedback=(
                            f"Evaluation failed with exit code {result.returncode}:\n"
                            f"{output_snippet}"
                        ),
                        guard_name="FullEvalGuard",
                    )

                return GuardResult(
                    passed=False,
                    feedback="Evaluation completed but no results file was produced.",
                    guard_name="FullEvalGuard",
                )

            except subprocess.TimeoutExpired:
                return GuardResult(
                    passed=False,
                    feedback=f"Evaluation timed out after {self._timeout}s.",
                    guard_name="FullEvalGuard",
                )

            except Exception as e:
                logger.error("Evaluation error: %s", e)
                return GuardResult(
                    passed=False,
                    feedback=f"Evaluation error: {e}",
                    guard_name="FullEvalGuard",
                )

    def _parse_eval_results(
        self,
        eval_results: dict,
        output: str,
    ) -> GuardResult:
        """Parse evaluation results and produce feedback.

        Args:
            eval_results: The parsed eval_results.json content.
            output: Raw output from the evaluation script.

        Returns:
            GuardResult based on evaluation outcome.
        """
        instance_id = self._instance.instance_id
        result = eval_results.get(instance_id)

        if result is None:
            return GuardResult(
                passed=False,
                feedback=f"Instance {instance_id} not found in evaluation results.",
                guard_name="FullEvalGuard",
            )

        # Handle both formats: bool or dict with "resolved" key
        if isinstance(result, bool):
            resolved = result
            details = {}
        elif isinstance(result, dict):
            resolved = result.get("resolved", False)
            details = result
        else:
            return GuardResult(
                passed=False,
                feedback=f"Unexpected result format: {type(result).__name__}",
                guard_name="FullEvalGuard",
            )

        if resolved:
            logger.info("Full eval PASSED for %s", instance_id)
            return GuardResult(
                passed=True,
                feedback=(
                    "Full evaluation PASSED!\n"
                    "- All FAIL_TO_PASS tests now pass\n"
                    "- All PASS_TO_PASS tests still pass (no regressions)"
                ),
                guard_name="FullEvalGuard",
            )

        # Build failure feedback
        feedback_parts = ["Full evaluation FAILED:"]

        if "fail_to_pass" in details:
            f2p = details["fail_to_pass"]
            if isinstance(f2p, dict):
                passed = f2p.get("passed", [])
                failed = f2p.get("failed", [])
                feedback_parts.append(
                    f"\nFAIL_TO_PASS: {len(passed)} passed, {len(failed)} still failing"
                )
                if failed:
                    feedback_parts.append(f"  Still failing: {', '.join(failed[:5])}")

        if "pass_to_pass" in details:
            p2p = details["pass_to_pass"]
            if isinstance(p2p, dict):
                passed = p2p.get("passed", [])
                failed = p2p.get("failed", [])
                feedback_parts.append(
                    f"\nPASS_TO_PASS: {len(passed)} passed, {len(failed)} regressed"
                )
                if failed:
                    feedback_parts.append(f"  Regressed: {', '.join(failed[:5])}")

        # Add relevant output if available
        if output:
            output_snippet = output[-500:]
            feedback_parts.append(f"\nOutput:\n{output_snippet}")

        feedback_parts.append(
            "\nAdjust the patch to fix all failing tests without breaking passing ones."
        )

        return GuardResult(
            passed=False,
            feedback="\n".join(feedback_parts),
            guard_name="FullEvalGuard",
        )
