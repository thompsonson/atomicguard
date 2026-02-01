"""SWE-PolyBench dataset loader.

Loads and filters instances from the AmazonScience/SWE-PolyBench dataset
for the Experiment 7.2 bug fix strategy comparison.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("swe_bench_ablation.dataset")


@dataclass(frozen=True)
class SWEInstance:
    """A single SWE-PolyBench instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str  # Gold patch
    test_patch: str
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    task_category: str = ""
    language: str = ""
    dockerfile: str = ""


def load_swe_polybench(
    split: str = "test",
    language: str = "python",
    task_category: str | None = "bug fix",
) -> list[SWEInstance]:
    """Load SWE-PolyBench dataset filtered by language and task category.

    Args:
        split: Dataset split to load
        language: Programming language filter (default: "python")
        task_category: Task category filter (default: "bug", None for all)

    Returns:
        List of SWEInstance objects matching the filters
    """
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("datasets library required: pip install datasets") from err

    logger.info(
        "Loading SWE-PolyBench (split=%s, language=%s, category=%s)",
        split,
        language,
        task_category,
    )

    ds = load_dataset("AmazonScience/SWE-PolyBench", split=split)

    instances = []
    for row in ds:
        # Filter by language
        row_lang = row.get("language", "")
        if language and row_lang.lower() != language.lower():
            continue

        # Filter by task category
        row_category = row.get("task_category", "")
        if task_category and row_category.lower() != task_category.lower():
            continue

        # Parse fail_to_pass and pass_to_pass (may be JSON strings)
        # SWE-PolyBench uses "F2P"/"P2P" column names
        fail_to_pass = _parse_test_list(row.get("fail_to_pass", row.get("F2P", [])))
        pass_to_pass = _parse_test_list(row.get("pass_to_pass", row.get("P2P", [])))

        instance = SWEInstance(
            instance_id=row["instance_id"],
            repo=row.get("repo", ""),
            base_commit=row.get("base_commit", ""),
            problem_statement=row.get("problem_statement", ""),
            patch=row.get("patch", ""),
            test_patch=row.get("test_patch", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            task_category=row_category,
            language=row_lang,
            dockerfile=row.get("Dockerfile", row.get("dockerfile", "")),
        )
        instances.append(instance)

    logger.info("Loaded %d instances", len(instances))
    return instances


def _parse_test_list(value: str | list[str]) -> list[str]:
    """Parse test list from string or list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        if value.strip():
            return [value]
    return []
