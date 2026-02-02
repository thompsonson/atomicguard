"""SWE-Bench Pro dataset loader.

Loads and filters instances from the ScaleAI/SWE-bench_Pro dataset on
HuggingFace for multi-language bug-fix evaluation.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("swe_bench_pro.dataset")


@dataclass(frozen=True)
class SWEBenchProInstance:
    """A single SWE-Bench Pro instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str  # Gold patch
    test_patch: str
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    repo_language: str = ""
    requirements: str = ""
    interface: str = ""


def load_swe_bench_pro(
    split: str = "test",
    language: str | None = None,
    max_instances: int | None = None,
) -> list[SWEBenchProInstance]:
    """Load SWE-Bench Pro dataset, optionally filtered by language.

    Args:
        split: Dataset split to load.
        language: Programming language filter (e.g. ``"python"``).
            ``None`` loads all languages.
        max_instances: Maximum number of instances to return. ``None``
            returns all matching instances.

    Returns:
        List of :class:`SWEBenchProInstance` objects.
    """
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("datasets library required: pip install datasets") from err

    # Reuse the test-list parser from the ablation example.
    from examples.swe_bench_ablation.dataset import _parse_test_list

    logger.info(
        "Loading SWE-Bench Pro (split=%s, language=%s, max=%s)",
        split,
        language,
        max_instances,
    )

    ds = load_dataset("ScaleAI/SWE-bench_Pro", split=split)

    _REQUIRED_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")

    instances: list[SWEBenchProInstance] = []
    skipped = 0
    for row in ds:
        # Validate required fields are present and non-empty.
        missing = [f for f in _REQUIRED_FIELDS if not row.get(f)]
        if missing:
            row_id = row.get("instance_id", "<unknown>")
            logger.warning(
                "Skipping instance %s: missing required fields %s", row_id, missing
            )
            skipped += 1
            continue

        row_lang = row.get("repo_language") or ""
        if not row_lang:
            logger.warning(
                "Instance %s has no repo_language field", row["instance_id"]
            )

        if language and row_lang.lower() != language.lower():
            continue

        fail_to_pass = _parse_test_list(row.get("fail_to_pass", []))
        pass_to_pass = _parse_test_list(row.get("pass_to_pass", []))

        instances.append(
            SWEBenchProInstance(
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                problem_statement=row["problem_statement"],
                patch=row.get("patch") or "",
                test_patch=row.get("test_patch") or "",
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                repo_language=row_lang,
                requirements=row.get("requirements") or "",
                interface=row.get("interface") or "",
            )
        )

        if max_instances and len(instances) >= max_instances:
            break

    if skipped:
        logger.warning("Skipped %d instances due to missing required fields", skipped)
    logger.info("Loaded %d instances", len(instances))
    return instances
