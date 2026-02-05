"""Evaluation harness integration for SWE-Bench Pro.

Wraps the official ``scaleapi/SWE-bench_Pro-os`` evaluation script via
subprocess.  Prediction files are formatted in the JSON schema expected
by ``swe_bench_pro_eval.py``, and per-instance evaluation logs are
written using the logging utilities from the ablation example.
"""

import json
import logging
import subprocess
from pathlib import Path

from examples.swe_bench_ablation.evaluation import (
    EvalResult,
    write_eval_logs,
)
from examples.swe_bench_ablation.experiment_runner import ArmResult

logger = logging.getLogger("swe_bench_pro.evaluation")

_EVAL_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
_EVAL_REPO_COMMIT = "main"  # pin to a tag/SHA for reproducibility


# =========================================================================
# Eval repo management
# =========================================================================


def ensure_eval_repo(
    cache_dir: str | Path = "~/.cache/swe_bench_pro",
    commit: str = _EVAL_REPO_COMMIT,
) -> Path:
    """Clone (or update) the SWE-Bench Pro evaluation repo.

    Args:
        cache_dir: Local directory for the clone.
        commit: Branch, tag, or SHA to check out.

    Returns:
        Path to the cloned repository.
    """
    cache_dir = Path(cache_dir).expanduser()
    repo_dir = cache_dir / "SWE-bench_Pro-os"

    if repo_dir.exists():
        logger.info("Updating eval repo at %s", repo_dir)
        try:
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=120,
                check=True,
            )
            subprocess.run(
                ["git", "checkout", commit],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=30,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to update eval repo at {repo_dir}: {e.stderr}"
            ) from e
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning eval repo to %s", repo_dir)
        subprocess.run(
            ["git", "clone", _EVAL_REPO_URL, str(repo_dir)],
            capture_output=True,
            timeout=300,
            check=True,
        )
        if commit != "main":
            subprocess.run(
                ["git", "checkout", commit],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=30,
                check=True,
            )

    return repo_dir


# =========================================================================
# Prediction formatting
# =========================================================================


def prepare_predictions(
    results: list[ArmResult],
    output_dir: str,
) -> dict[str, Path]:
    """Convert :class:`ArmResult` objects to SWE-Bench Pro prediction JSON.

    SWE-Bench Pro expects a JSON file containing a list of objects::

        [{"instance_id": "...", "patch": "...", "prefix": "..."}]

    Args:
        results: Results from the experiment runner.
        output_dir: Directory to write prediction files into.

    Returns:
        Mapping of arm name to the written prediction file path.
    """
    out_path = Path(output_dir) / "predictions"
    out_path.mkdir(parents=True, exist_ok=True)

    by_arm: dict[str, list[ArmResult]] = {}
    for r in results:
        by_arm.setdefault(r.arm, []).append(r)

    prediction_files: dict[str, Path] = {}

    for arm, arm_results in by_arm.items():
        predictions = []
        skipped: list[str] = []
        for r in arm_results:
            if r.workflow_status != "success" or not r.patch_content:
                skipped.append(r.instance_id)
                continue
            predictions.append(
                {
                    "instance_id": r.instance_id,
                    "patch": r.patch_content,
                    "prefix": f"atomicguard-{arm}",
                }
            )

        if skipped:
            logger.warning(
                "arm=%s: skipped %d instances with no successful patch: %s",
                arm,
                len(skipped),
                ", ".join(skipped),
            )

        pred_file = out_path / f"{arm}.json"
        pred_file.write_text(json.dumps(predictions, indent=2))
        prediction_files[arm] = pred_file

        logger.info(
            "Wrote %d/%d predictions for arm=%s to %s",
            len(predictions),
            len(arm_results),
            arm,
            pred_file,
        )

    return prediction_files


# =========================================================================
# Evaluation
# =========================================================================


def run_evaluation(
    predictions_path: str | Path,
    eval_repo_path: str | Path,
    dataset_csv: str | Path | None = None,
    mode: str = "local",
    max_workers: int = 4,
    timeout: int = 7200,
    block_network: bool = True,
    dockerhub_username: str = "jefzda",
) -> dict[str, object]:
    """Run SWE-Bench Pro evaluation via the official eval script.

    Args:
        predictions_path: Path to prediction JSON file.
        eval_repo_path: Path to the cloned ``SWE-bench_Pro-os`` repo.
        dataset_csv: Path to the ``swe_bench_pro_full.csv`` file inside
            the eval repo.  Defaults to ``{eval_repo_path}/swe_bench_pro_full.csv``.
        mode: ``"local"`` for local Docker, ``"modal"`` for Modal cloud.
        max_workers: Number of parallel evaluation workers.
        timeout: Overall timeout in seconds.
        block_network: Disable container networking during tests.
        dockerhub_username: DockerHub account hosting the pre-built images.

    Returns:
        Dict with ``status``, ``stdout``, ``stderr`` keys.
    """
    predictions_path = Path(predictions_path)
    eval_repo_path = Path(eval_repo_path)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    eval_script = eval_repo_path / "swe_bench_pro_eval.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Eval script not found: {eval_script}")

    if dataset_csv is None:
        dataset_csv = eval_repo_path / "swe_bench_pro_full.csv"
    dataset_csv = Path(dataset_csv)

    output_dir = predictions_path.parent / "eval_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(eval_script),
        f"--raw_sample_path={dataset_csv}",
        f"--patch_path={predictions_path}",
        f"--output_dir={output_dir}",
        f"--scripts_dir={eval_repo_path / 'run_scripts'}",
        f"--num_workers={max_workers}",
        f"--dockerhub_username={dockerhub_username}",
    ]

    if mode == "local":
        cmd.append("--use_local_docker")
    if block_network:
        cmd.append("--block_network")

    logger.info("Running SWE-Bench Pro evaluation: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            logger.error("Evaluation failed: %s", result.stderr[-500:])
            return {"status": "error", "stdout": result.stdout, "stderr": result.stderr}

        logger.info("Evaluation complete")
        return {"status": "success", "stdout": result.stdout, "stderr": result.stderr}

    except subprocess.TimeoutExpired:
        logger.error("Evaluation timed out after %ds", timeout)
        return {"status": "timeout"}
    except FileNotFoundError:
        logger.error("Python not found for eval script execution")
        return {"status": "not_found"}


# =========================================================================
# Result loading
# =========================================================================


def load_evaluation_results(
    results_dir: str | Path,
    run_id: str = "swe_bench_pro",
    write_logs: bool = True,
) -> dict[str, bool]:
    """Load evaluation results from the SWE-Bench Pro eval output.

    Parses ``eval_results.json`` produced by ``swe_bench_pro_eval.py``
    and optionally writes per-instance log files.

    Args:
        results_dir: Directory containing ``eval_output/eval_results.json``.
        run_id: Identifier used for the log subdirectory.
        write_logs: Whether to write per-instance ``.log`` files.

    Returns:
        Mapping of instance_id to resolved bool.
    """
    results_dir = Path(results_dir)
    resolved: dict[str, bool] = {}

    # The eval script writes to eval_output/eval_results.json
    candidates = [
        results_dir / "eval_output" / "eval_results.json",
        results_dir / "eval_results.json",
    ]
    found = False
    for candidate in candidates:
        if candidate.exists():
            found = True
            data = json.loads(candidate.read_text())
            if not isinstance(data, dict):
                raise ValueError(
                    f"Expected dict in {candidate}, got {type(data).__name__}"
                )
            for iid, val in data.items():
                if isinstance(val, bool):
                    resolved[iid] = val
                elif isinstance(val, dict):
                    if "resolved" not in val:
                        raise KeyError(
                            f"Instance {iid!r} in {candidate} has no 'resolved' key. "
                            f"Available keys: {sorted(val)}"
                        )
                    resolved[iid] = val["resolved"]
                else:
                    raise TypeError(
                        f"Unexpected value type for {iid!r}: {type(val).__name__}"
                    )
            break

    if not found:
        logger.warning(
            "No eval_results.json found in %s (searched: %s)",
            results_dir,
            ", ".join(str(c) for c in candidates),
        )

    logger.info("Loaded %d evaluation results", len(resolved))

    if write_logs and resolved:
        eval_results = {
            iid: EvalResult(instance_id=iid, resolved=ok)
            for iid, ok in resolved.items()
        }
        write_eval_logs(eval_results, str(results_dir), run_id)

    return resolved


# =========================================================================
# Inline evaluation for experiment integration
# =========================================================================


def evaluate_predictions_inline(
    predictions_dir: Path,
    eval_max_workers: int = 4,
    eval_timeout: int = 7200,
) -> dict[tuple[str, str], bool]:
    """Run evaluation on all prediction files and return resolved mapping.

    This function is designed for integration with the experiment command
    to enable a single-command workflow: patches → evaluation → visualization.

    Args:
        predictions_dir: Directory containing prediction JSON files (one per arm).
        eval_max_workers: Number of parallel evaluation workers.
        eval_timeout: Timeout in seconds for evaluation.

    Returns:
        Mapping of ``(instance_id, arm) → resolved`` bool.

    Raises:
        RuntimeError: If evaluation fails for any arm (explicit failure, no fallbacks).
    """
    eval_repo = ensure_eval_repo()
    resolved_map: dict[tuple[str, str], bool] = {}

    pred_files = sorted(predictions_dir.glob("*.json"))
    if not pred_files:
        logger.warning("No prediction files found in %s", predictions_dir)
        return resolved_map

    for pred_file in pred_files:
        arm = pred_file.stem
        logger.info("Evaluating arm=%s from %s", arm, pred_file)

        # Check if prediction file has any entries
        predictions = json.loads(pred_file.read_text())
        if not predictions:
            logger.info(
                "arm=%s: no predictions to evaluate (all instances failed workflow)", arm
            )
            continue

        result = run_evaluation(
            predictions_path=pred_file,
            eval_repo_path=eval_repo,
            max_workers=eval_max_workers,
            timeout=eval_timeout,
        )

        if result["status"] == "success":
            arm_resolved = load_evaluation_results(
                str(pred_file.parent),
                run_id=arm,
            )
            for instance_id, is_resolved in arm_resolved.items():
                resolved_map[(instance_id, arm)] = is_resolved
            logger.info(
                "arm=%s: %d/%d resolved",
                arm,
                sum(1 for v in arm_resolved.values() if v),
                len(arm_resolved),
            )
        else:
            raise RuntimeError(
                f"Evaluation failed for arm={arm}: status={result['status']}.\n"
                f"stderr:\n{result.get('stderr', 'N/A')[-1000:]}"
            )

    return resolved_map
