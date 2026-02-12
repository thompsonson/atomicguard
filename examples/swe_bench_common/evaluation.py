"""Shared evaluation utilities for SWE-bench experiment runners.

Contains:
- EvalResult dataclass: Result of evaluating a single instance
- write_eval_logs: Write per-instance log files and summary
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("swe_bench_common.evaluation")


@dataclass
class EvalResult:
    """Result of evaluating a single instance."""

    instance_id: str
    resolved: bool
    fail_to_pass_results: dict[str, bool] = field(default_factory=dict)
    pass_to_pass_results: dict[str, bool] = field(default_factory=dict)
    error: str | None = None
    log: str = ""
    wall_time_seconds: float = 0.0


def _sanitize_instance_id(instance_id: str) -> str:
    """Sanitize instance ID for use as a filename.

    Replaces path separators with double underscores so that IDs like
    ``astropy/astropy-12907`` become ``astropy__astropy-12907``.
    """
    return instance_id.replace("/", "__").replace("\\", "__")


def format_instance_log(result: EvalResult) -> str:
    """Format an EvalResult as a human-readable log string.

    Args:
        result: Evaluation result for a single instance.

    Returns:
        Multi-line string suitable for writing to a ``.log`` file.
    """
    lines: list[str] = [
        f"Instance: {result.instance_id}",
        f"Resolved: {result.resolved}",
        f"Wall time: {result.wall_time_seconds:.1f}s",
        "",
    ]

    if result.error:
        lines.append(f"Error: {result.error}")
        lines.append("")

    if result.fail_to_pass_results:
        passed = sum(1 for v in result.fail_to_pass_results.values() if v)
        total = len(result.fail_to_pass_results)
        lines.append(f"Fail-to-Pass Tests ({passed}/{total} passed):")
        for test_id, ok in sorted(result.fail_to_pass_results.items()):
            status = "PASS" if ok else "FAIL"
            lines.append(f"  [{status}] {test_id}")
        lines.append("")

    if result.pass_to_pass_results:
        passed = sum(1 for v in result.pass_to_pass_results.values() if v)
        total = len(result.pass_to_pass_results)
        lines.append(f"Pass-to-Pass Tests ({passed}/{total} passed):")
        for test_id, ok in sorted(result.pass_to_pass_results.items()):
            status = "PASS" if ok else "FAIL"
            lines.append(f"  [{status}] {test_id}")
        lines.append("")

    if result.log:
        lines.append("--- Execution Log ---")
        lines.append(result.log)
        lines.append("")

    return "\n".join(lines)


def write_instance_log(result: EvalResult, log_dir: Path) -> Path:
    """Write a human-readable log file for a single evaluation result.

    Args:
        result: Evaluation result for a single instance.
        log_dir: Directory to write the log file into.

    Returns:
        Path to the written log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_sanitize_instance_id(result.instance_id)}.log"
    log_path = log_dir / filename
    log_path.write_text(format_instance_log(result))
    return log_path


def write_eval_logs(
    results: dict[str, EvalResult],
    output_dir: str | Path,
    run_id: str = "experiment_7_2",
) -> Path:
    """Write per-instance log files and a summary for an evaluation run.

    Creates a directory ``{output_dir}/eval_logs/{run_id}/`` containing one
    ``.log`` file per instance and a ``_summary.log`` with aggregate stats.

    Args:
        results: Mapping of instance_id to EvalResult.
        output_dir: Base output directory (e.g. ``output/experiment_7_2``).
        run_id: Identifier for this evaluation run.

    Returns:
        Path to the log directory that was written.
    """
    log_dir = Path(output_dir) / "eval_logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    for result in results.values():
        write_instance_log(result, log_dir)

    # Write summary
    total = len(results)
    resolved_count = sum(1 for r in results.values() if r.resolved)
    error_count = sum(1 for r in results.values() if r.error)

    summary_lines: list[str] = [
        f"Evaluation Summary (run_id={run_id})",
        f"{'=' * 50}",
        f"Timestamp: {datetime.now(UTC).isoformat()}",
        f"Total instances: {total}",
    ]
    if total > 0:
        summary_lines.append(
            f"Resolved: {resolved_count}/{total} ({resolved_count / total * 100:.1f}%)"
        )
    else:
        summary_lines.append("Resolved: 0/0")
    summary_lines += [
        f"Errors: {error_count}",
        "",
        "Per-instance:",
    ]

    for instance_id in sorted(results):
        r = results[instance_id]
        if r.error:
            status = f"ERROR ({r.error[:80]})"
        elif r.resolved:
            status = "RESOLVED"
        else:
            status = "FAILED"
        summary_lines.append(f"  {instance_id}: {status} ({r.wall_time_seconds:.1f}s)")

    summary_path = log_dir / "_summary.log"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    logger.info("Wrote %d instance logs + summary to %s", total, log_dir)
    return log_dir
