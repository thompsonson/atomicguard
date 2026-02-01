"""Evaluation harness integration for SWE-bench.

Converts experiment results to SWE-bench prediction format
and runs the swebench evaluation harness.
"""

import json
import logging
import subprocess
from pathlib import Path

from .experiment_runner import ArmResult

logger = logging.getLogger("swe_bench_ablation.evaluation")


def prepare_predictions(
    results: list[ArmResult],
    output_dir: str,
) -> dict[str, Path]:
    """Convert ArmResults to SWE-bench prediction JSONL files.

    Creates one JSONL file per arm with the format expected by swebench:
    {"instance_id": "...", "model_patch": "...", "model_name_or_path": "..."}

    Args:
        results: List of ArmResults from experiment runner
        output_dir: Directory to write prediction files

    Returns:
        Dict mapping arm name to prediction file path
    """
    out_path = Path(output_dir) / "predictions"
    out_path.mkdir(parents=True, exist_ok=True)

    # Group results by arm
    by_arm: dict[str, list[ArmResult]] = {}
    for r in results:
        by_arm.setdefault(r.arm, []).append(r)

    prediction_files: dict[str, Path] = {}

    for arm, arm_results in by_arm.items():
        pred_file = out_path / f"{arm}.jsonl"
        with open(pred_file, "w") as f:
            for r in arm_results:
                if r.workflow_status != "success" or not r.patch_content:
                    continue

                prediction = {
                    "instance_id": r.instance_id,
                    "model_patch": r.patch_content,
                    "model_name_or_path": f"atomicguard-{arm}",
                }
                f.write(json.dumps(prediction) + "\n")

        prediction_files[arm] = pred_file
        logger.info(
            "Wrote %d predictions for arm=%s to %s",
            sum(1 for r in arm_results if r.workflow_status == "success"),
            arm,
            pred_file,
        )

    return prediction_files


def run_evaluation(
    predictions_path: str | Path,
    dataset_name: str = "AmazonScience/SWE-PolyBench",
    split: str = "test",
    max_workers: int = 4,
    run_id: str = "experiment_7_2",
) -> dict[str, object]:
    """Run SWE-bench evaluation harness on predictions.

    Args:
        predictions_path: Path to prediction JSONL file
        dataset_name: HuggingFace dataset name
        split: Dataset split
        max_workers: Number of parallel Docker workers
        run_id: Identifier for this evaluation run

    Returns:
        Dict with evaluation results
    """
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    logger.info("Running swebench evaluation on %s", predictions_path)

    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours max
        )

        if result.returncode != 0:
            logger.error("swebench evaluation failed: %s", result.stderr)
            return {"status": "error", "stderr": result.stderr}

        logger.info("swebench evaluation complete")

        # Try to parse results from output
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        logger.error("swebench evaluation timed out")
        return {"status": "timeout"}
    except FileNotFoundError:
        logger.error("swebench not found. Install with: pip install swebench")
        return {"status": "not_installed"}


def load_evaluation_results(
    results_dir: str,
    run_id: str = "experiment_7_2",
) -> dict[str, bool]:
    """Load evaluation results from swebench output.

    Args:
        results_dir: Directory containing swebench output
        run_id: The run ID used during evaluation

    Returns:
        Dict mapping instance_id to resolved (True/False)
    """
    results_path = Path(results_dir) / run_id
    resolved: dict[str, bool] = {}

    # swebench writes results to a JSON file
    for json_file in results_path.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            if isinstance(data, dict):
                for instance_id, instance_result in data.items():
                    if isinstance(instance_result, dict):
                        resolved[instance_id] = instance_result.get("resolved", False)
                    elif isinstance(instance_result, bool):
                        resolved[instance_id] = instance_result
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Error reading %s: %s", json_file, e)

    logger.info("Loaded %d evaluation results", len(resolved))
    return resolved
