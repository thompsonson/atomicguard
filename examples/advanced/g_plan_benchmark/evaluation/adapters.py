"""Dataset adapters that convert external benchmarks into ProblemSet objects.

Each adapter loads a benchmark dataset (from HuggingFace Hub or local files)
and maps its fields to the Problem schema used by the evaluation harness.

Supported datasets:
    - SWE-PolyBench (PB500): Multi-language with explicit task_category
    - SWE-bench Verified: Python-only with difficulty labels

Usage:
    from evaluation.adapters import load_swe_polybench, load_swe_bench

    ps = load_swe_polybench()           # from HuggingFace Hub
    ps = load_swe_polybench(limit=50)   # first 50 instances
    ps = load_swe_bench(path="local/")  # from local files
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .problem import Problem, ProblemSet, TYPE_TO_STRATEGY


# ---------------------------------------------------------------------------
# SWE-PolyBench field mappings
# ---------------------------------------------------------------------------

# SWE-PolyBench uses human-readable task_category values.
# Map to our internal expected_type vocabulary.
_POLYBENCH_CATEGORY_MAP: dict[str, str] = {
    "Bug Fix": "bug_fix",
    "bug_fix": "bug_fix",
    "Feature": "feature",
    "feature": "feature",
    "Refactoring": "refactoring",
    "refactoring": "refactoring",
}

# SWE-PolyBench language field may use various casings.
_POLYBENCH_LANGUAGE_MAP: dict[str, str] = {
    "Python": "python",
    "python": "python",
    "Java": "java",
    "java": "java",
    "JavaScript": "javascript",
    "javascript": "javascript",
    "TypeScript": "typescript",
    "typescript": "typescript",
}


def _require_datasets() -> Any:
    """Import the ``datasets`` library or raise a clear error."""
    try:
        import datasets  # type: ignore[import-untyped]

        return datasets
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load HuggingFace datasets. "
            "Install it with: pip install datasets"
        ) from exc


# ---------------------------------------------------------------------------
# SWE-PolyBench adapter
# ---------------------------------------------------------------------------

def _polybench_instance_to_problem(instance: dict[str, Any]) -> Problem:
    """Convert a single SWE-PolyBench instance to a Problem."""
    # Instance ID: use instance_id if present, otherwise construct from repo
    problem_id = str(
        instance.get("instance_id")
        or instance.get("id")
        or instance.get("repo", "unknown")
    )

    description = instance.get("problem_statement", "")
    if not description:
        raise ValueError(f"Instance {problem_id} has no problem_statement")

    raw_category = str(instance.get("task_category", "unknown"))
    expected_type = _POLYBENCH_CATEGORY_MAP.get(raw_category, "unknown")

    raw_language = str(instance.get("language", "unknown"))
    language = _POLYBENCH_LANGUAGE_MAP.get(raw_language, raw_language.lower())

    expected_strategy = TYPE_TO_STRATEGY.get(expected_type, "")

    metadata: dict[str, Any] = {}
    for key in ("repo", "base_commit", "patch", "test_patch", "F2P", "P2P",
                "Dockerfile", "test_command"):
        if key in instance and instance[key] is not None:
            metadata[key] = instance[key]

    return Problem(
        problem_id=problem_id,
        description=description,
        expected_type=expected_type,
        language=language,
        expected_strategy=expected_strategy,
        difficulty="unknown",
        source="swe-polybench",
        metadata=metadata,
    )


def load_swe_polybench(
    *,
    path: str | Path | None = None,
    dataset_name: str = "AmazonScience/SWE-PolyBench_500",
    split: str = "test",
    limit: int | None = None,
) -> ProblemSet:
    """Load SWE-PolyBench as a ProblemSet.

    Args:
        path: Local file or directory to load from (JSON/JSONL).
            If None, loads from HuggingFace Hub.
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to use.
        limit: Maximum number of instances to load (None = all).

    Returns:
        ProblemSet with SWE-PolyBench instances mapped to Problem objects.
    """
    if path is not None:
        return _load_local_dataset(Path(path), "swe-polybench",
                                   _polybench_instance_to_problem, limit)

    ds_lib = _require_datasets()
    ds = ds_lib.load_dataset(dataset_name, split=split)

    problems: list[Problem] = []
    for i, instance in enumerate(ds):
        if limit is not None and i >= limit:
            break
        problems.append(_polybench_instance_to_problem(instance))

    return ProblemSet(problems)


# ---------------------------------------------------------------------------
# SWE-bench Verified adapter
# ---------------------------------------------------------------------------

def _swebench_instance_to_problem(instance: dict[str, Any]) -> Problem:
    """Convert a single SWE-bench Verified instance to a Problem."""
    problem_id = str(instance.get("instance_id", "unknown"))

    description = instance.get("problem_statement", "")
    if not description:
        raise ValueError(f"Instance {problem_id} has no problem_statement")

    # SWE-bench has no explicit task_category.
    # Most instances are bug fixes, but we mark as unknown for honest scoring.
    expected_type = "unknown"

    # SWE-bench Verified difficulty field
    raw_difficulty = str(instance.get("difficulty", "unknown")).lower()
    difficulty_map = {
        "easy": "easy",
        "medium": "medium",
        "hard": "hard",
        "very hard": "hard",  # collapse to 3-level scale
    }
    difficulty = difficulty_map.get(raw_difficulty, "unknown")

    metadata: dict[str, Any] = {}
    for key in ("repo", "base_commit", "patch", "test_patch",
                "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text"):
        if key in instance and instance[key] is not None:
            metadata[key] = instance[key]

    return Problem(
        problem_id=problem_id,
        description=description,
        expected_type=expected_type,
        language="python",  # SWE-bench is Python-only
        expected_strategy="",  # no ground truth without task_category
        difficulty=difficulty,
        source="swe-bench",
        metadata=metadata,
    )


def load_swe_bench(
    *,
    path: str | Path | None = None,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    limit: int | None = None,
) -> ProblemSet:
    """Load SWE-bench Verified as a ProblemSet.

    Args:
        path: Local file or directory to load from (JSON/JSONL).
            If None, loads from HuggingFace Hub.
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to use.
        limit: Maximum number of instances to load (None = all).

    Returns:
        ProblemSet with SWE-bench instances mapped to Problem objects.
    """
    if path is not None:
        return _load_local_dataset(Path(path), "swe-bench",
                                   _swebench_instance_to_problem, limit)

    ds_lib = _require_datasets()
    ds = ds_lib.load_dataset(dataset_name, split=split)

    problems: list[Problem] = []
    for i, instance in enumerate(ds):
        if limit is not None and i >= limit:
            break
        problems.append(_swebench_instance_to_problem(instance))

    return ProblemSet(problems)


# ---------------------------------------------------------------------------
# Local file loading (shared)
# ---------------------------------------------------------------------------

def _load_local_dataset(
    path: Path,
    source: str,
    converter: Any,
    limit: int | None,
) -> ProblemSet:
    """Load instances from a local JSON or JSONL file/directory.

    Supports:
        - Single JSON file with a list of instances or {"problems": [...]}
        - JSONL file (one JSON object per line)
        - Directory of individual JSON files
    """
    import json

    if path.is_dir():
        instances: list[dict[str, Any]] = []
        for json_file in sorted(path.glob("*.json")):
            with open(json_file) as f:
                instances.append(json.load(f))
        for jsonl_file in sorted(path.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        instances.append(json.loads(line))
    elif path.suffix == ".jsonl":
        instances = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    instances.append(json.loads(line))
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            instances = data
        elif isinstance(data, dict) and "problems" in data:
            instances = data["problems"]
        elif isinstance(data, dict) and "instances" in data:
            instances = data["instances"]
        else:
            raise ValueError(
                f"Expected JSON array or object with 'problems'/'instances' key: {path}"
            )
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (expected .json or .jsonl)")

    if not instances:
        raise ValueError(f"No instances found in {path}")

    if limit is not None:
        instances = instances[:limit]

    return ProblemSet([converter(inst) for inst in instances])
