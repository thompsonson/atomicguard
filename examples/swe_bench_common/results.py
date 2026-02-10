"""Shared result persistence utilities for SWE-bench experiment runners."""

import json
import logging
from pathlib import Path

from .models import ArmResult

logger = logging.getLogger("swe_bench_common.results")


def load_existing_results(
    results_dir: str | Path,
) -> tuple[list[ArmResult], set[tuple[str, str]]]:
    """Load existing results from JSONL for resume support.

    Args:
        results_dir: Directory containing results.jsonl file.

    Returns:
        Tuple of (list of ArmResult objects, set of (instance_id, arm) tuples
        representing completed runs).
    """
    results_path = Path(results_dir) / "results.jsonl"
    results: list[ArmResult] = []
    completed: set[tuple[str, str]] = set()

    if not results_path.exists():
        return results, completed

    # Get valid field names from the dataclass
    valid_fields = {f.name for f in ArmResult.__dataclass_fields__.values()}

    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Filter out unknown fields (backward compatibility)
                filtered_data = {k: v for k, v in data.items() if k in valid_fields}
                arm_result = ArmResult(**filtered_data)
                results.append(arm_result)
                completed.add((arm_result.instance_id, arm_result.arm))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Skipping malformed result line: %s", e)

    return results, completed
