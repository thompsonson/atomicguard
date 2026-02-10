"""Final Error Analysis for SWE-Bench Pro experiments.

Extracts and analyzes the "final error" for each failed arm run - the LAST
guard feedback before the agent gave up (retry exhaustion).

This analysis reveals what the agent was "stuck on" when retries were exhausted,
identifies "brick wall" guards that block all attempts, and helps understand
what improvements would help progress.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("swe_bench_pro.final_error_analysis")


@dataclass
class ExperimentOverview:
    """Overview statistics for the entire experiment."""

    model: str | None
    arms: list[str]
    instance_count: int
    total_runs: int
    successful_runs: int
    failed_runs: int
    total_tokens: int
    avg_wall_time: float
    avg_init_time: float = 0.0
    avg_workflow_time: float = 0.0


@dataclass
class RetryError:
    """Error from a single retry attempt."""

    attempt: int  # 1-indexed retry number
    action_pair: str  # e.g., "ap_gen_patch"
    guard: str
    feedback: str
    error_summary: str


@dataclass
class FinalError:
    """Final error information for a failed run."""

    instance_id: str
    arm: str
    failed_step: str
    failed_guard: str
    retry_count: int
    final_feedback: str
    error_summary: str
    all_retries: list[RetryError] = field(default_factory=list)


def extract_error_summary(feedback: str, guard: str) -> str:
    """Extract a concise error message from guard feedback.

    Parses guard-specific feedback formats to extract the actual
    error (not generic descriptions).

    Args:
        feedback: Full guard feedback text
        guard: Guard name (TestGreenGuard, LintGuard, PatchGuard, etc.)

    Returns:
        Concise error summary (typically one line)
    """
    if not feedback:
        return "No feedback"

    lines = feedback.strip().split("\n")

    # TestGreenGuard: look for test failure exceptions
    if guard == "TestGreenGuard":
        # Look for common test errors
        for line in lines:
            # ImportError, ModuleNotFoundError
            if "ImportError:" in line or "ModuleNotFoundError:" in line:
                return line.strip()
            # AssertionError
            if "AssertionError:" in line:
                return line.strip()
            # AttributeError
            if "AttributeError:" in line:
                return line.strip()
            # NameError
            if "NameError:" in line:
                return line.strip()
            # TypeError
            if "TypeError:" in line:
                return line.strip()
            # FAILED line from pytest
            if line.startswith("FAILED "):
                return line.strip()
            # UserWarning special case
            if "UserWarning:" in line:
                return line.strip()
        # Look for E   prefix (pytest exception lines)
        for line in lines:
            if line.startswith("E   "):
                return line[4:].strip()
        # Fallback: first non-empty line after "Test still FAILS"
        for i, line in enumerate(lines):
            if "still FAILS" in line and i + 1 < len(lines):
                for j in range(i + 1, min(i + 10, len(lines))):
                    if lines[j].strip() and not lines[j].startswith("#"):
                        return lines[j].strip()[:100]

    # LintGuard: look for lint errors
    if guard == "LintGuard":
        # Look for "undefined name" pattern
        for line in lines:
            if "undefined name" in line.lower():
                return line.strip()
            if "F821" in line:  # flake8 undefined name
                return line.strip()
        # Look for other common lint patterns
        for line in lines:
            # E999: syntax error
            if ":E999" in line or "E999:" in line or "SyntaxError" in line:
                return line.strip()
            # Import errors
            if "imported but unused" in line or "unable to detect" in line:
                return line.strip()

    # PatchGuard: look for edit validation errors
    if guard == "PatchGuard":
        for line in lines:
            if "Search string not found" in line:
                # Extract file name
                match = re.search(r"not found in (.+?)(?:\.|$)", line)
                if match:
                    return f"Search string not found in {match.group(1)}"
                return "Search string not found"
            if "No patch found" in line or "no valid edits" in line.lower():
                return "No patch found in output"
            if "Syntax error" in line.lower():
                return line.strip()[:100]
            if "context mismatch" in line.lower():
                return "Patch context mismatch"

    # Generic fallback: return first meaningful line (up to 100 chars)
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and len(stripped) > 10:
            return stripped[:100]

    return feedback[:100] if feedback else "Unknown error"


def load_artifact_feedback(
    dag_dir: Path, instance_id: str, arm: str, failed_step: str
) -> tuple[str, list[RetryError]]:
    """Load guard feedback from artifact DAG for a failed run.

    Args:
        dag_dir: Base directory for artifact DAGs
        instance_id: Instance ID (e.g., "instance_qutebrowser__qutebrowser-...")
        arm: Arm name (e.g., "05_s1_tdd_verified")
        failed_step: Action pair that failed (e.g., "ap_gen_patch")

    Returns:
        Tuple of (final_feedback, list of all retry errors)
    """
    index_path = dag_dir / instance_id / arm / "index.json"
    if not index_path.exists():
        return "", []

    try:
        with open(index_path) as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load index %s: %s", index_path, e)
        return "", []

    # Get all artifact UUIDs for the failed step
    step_uuids = index.get("action_pairs", {}).get(failed_step, [])
    if not step_uuids:
        return "", []

    artifacts_dir = dag_dir / instance_id / arm

    retry_errors: list[RetryError] = []
    final_feedback = ""

    for attempt, uuid in enumerate(step_uuids, 1):
        artifact_meta = index.get("artifacts", {}).get(uuid, {})
        artifact_path = artifacts_dir / artifact_meta.get("path", "")
        action_pair_id = artifact_meta.get("action_pair_id", failed_step)

        if not artifact_path.exists():
            continue

        try:
            with open(artifact_path) as f:
                artifact = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        guard_result = artifact.get("guard_result", {})
        if not guard_result:
            continue

        feedback = guard_result.get("feedback", "")
        guard_name = guard_result.get("guard_name", "Unknown")
        passed = guard_result.get("passed", True)

        if not passed and feedback:
            error_summary = extract_error_summary(feedback, guard_name)
            retry_errors.append(
                RetryError(
                    attempt=attempt,
                    action_pair=action_pair_id,
                    guard=guard_name,
                    feedback=feedback,
                    error_summary=error_summary,
                )
            )
            final_feedback = feedback

    return final_feedback, retry_errors


def extract_final_errors(
    results_path: str | Path,
    artifact_dags_dir: str | Path | None = None,
) -> list[FinalError]:
    """Extract final errors for all failed runs.

    Args:
        results_path: Path to results.jsonl file
        artifact_dags_dir: Path to artifact_dags directory (inferred if None)

    Returns:
        List of FinalError objects for failed runs
    """
    results_path = Path(results_path)
    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        return []

    # Infer artifact_dags_dir if not provided
    if artifact_dags_dir is None:
        artifact_dags_dir = results_path.parent / "artifact_dags"
    else:
        artifact_dags_dir = Path(artifact_dags_dir)

    errors: list[FinalError] = []

    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip successful runs
            if not result.get("failed_step"):
                continue

            instance_id = result.get("instance_id", "")
            arm = result.get("arm", "")
            failed_step = result.get("failed_step", "")
            failed_guard = result.get("failed_guard", "Unknown")
            retry_count = result.get("retry_count", 0)

            # Load feedback from artifact DAG
            final_feedback, retry_errors = load_artifact_feedback(
                artifact_dags_dir, instance_id, arm, failed_step
            )

            error_summary = ""
            if retry_errors:
                error_summary = retry_errors[-1].error_summary
            elif final_feedback:
                error_summary = extract_error_summary(final_feedback, failed_guard)

            errors.append(
                FinalError(
                    instance_id=instance_id,
                    arm=arm,
                    failed_step=failed_step,
                    failed_guard=failed_guard,
                    retry_count=retry_count,
                    final_feedback=final_feedback,
                    error_summary=error_summary,
                    all_retries=retry_errors,
                )
            )

    logger.info("Extracted %d final errors from %s", len(errors), results_path)
    return errors


def get_timestamped_path(base_path: Path, suffix: str) -> Path:
    """Generate timestamped filename: base_20260209_143022.suffix"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = base_path.stem
    return base_path.parent / f"{stem}_{ts}{suffix}"


def update_latest_symlink(timestamped_path: Path, latest_name: str) -> None:
    """Create/update symlink: final_error_analysis_latest.md -> ..._{ts}.md"""
    latest = timestamped_path.parent / latest_name
    latest.unlink(missing_ok=True)
    latest.symlink_to(timestamped_path.name)


def compute_experiment_overview(
    results_path: Path,
    all_results: list[dict] | None = None,
) -> ExperimentOverview:
    """Compute experiment overview from results.jsonl.

    Reads all results (not just failed) to compute totals.
    Optionally tries to extract model from log file.

    Args:
        results_path: Path to results.jsonl file
        all_results: Pre-loaded results (avoids re-reading file)

    Returns:
        ExperimentOverview with computed statistics
    """
    if all_results is None:
        all_results = []
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Compute basic stats
    arms = sorted(set(r.get("arm", "") for r in all_results if r.get("arm")))
    instances = set(r.get("instance_id", "") for r in all_results if r.get("instance_id"))

    successful = sum(1 for r in all_results if not r.get("failed_step"))
    failed = sum(1 for r in all_results if r.get("failed_step"))

    # Token totals (check for different field names)
    total_tokens = 0
    for r in all_results:
        # Try total_tokens first, then input+output
        if "total_tokens" in r:
            total_tokens += r.get("total_tokens", 0)
        else:
            total_tokens += r.get("input_tokens", 0)
            total_tokens += r.get("output_tokens", 0)

    # Wall time average (check for different field names)
    wall_times = []
    for r in all_results:
        wt = r.get("wall_time_seconds") or r.get("wall_time", 0)
        if wt:
            wall_times.append(wt)
    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0.0

    # Init time average (new field, may not exist in old results)
    init_times = []
    for r in all_results:
        it = r.get("init_time_seconds", 0)
        if it:
            init_times.append(it)
    avg_init_time = sum(init_times) / len(init_times) if init_times else 0.0

    # Workflow time average (new field, may not exist in old results)
    workflow_times = []
    for r in all_results:
        wft = r.get("workflow_time_seconds", 0)
        if wft:
            workflow_times.append(wft)
    avg_workflow_time = sum(workflow_times) / len(workflow_times) if workflow_times else 0.0

    # Try to extract model from experiment log
    model = None
    log_path = results_path.parent / "experiment.log"
    if log_path.exists():
        try:
            log_text = log_path.read_text()
            # Look for model in log (e.g., "model=qwen/qwen3-coder-next")
            match = re.search(r"model[=:][\s]*([^\s,\]]+)", log_text)
            if match:
                model = match.group(1)
        except OSError:
            pass

    return ExperimentOverview(
        model=model,
        arms=arms,
        instance_count=len(instances),
        total_runs=len(all_results),
        successful_runs=successful,
        failed_runs=failed,
        total_tokens=total_tokens,
        avg_wall_time=avg_wall_time,
        avg_init_time=avg_init_time,
        avg_workflow_time=avg_workflow_time,
    )


def generate_final_error_report(
    errors: list[FinalError],
    output_dir: str | Path,
    overview: ExperimentOverview | None = None,
) -> str:
    """Generate Markdown report organized by arm.

    Args:
        errors: List of FinalError objects
        output_dir: Directory to write report
        overview: Optional experiment overview to include at top

    Returns:
        Report text as string
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by arm
    by_arm: dict[str, list[FinalError]] = defaultdict(list)
    for err in errors:
        by_arm[err.arm].append(err)

    lines = [
        "# FINAL ERROR ANALYSIS",
        "=" * 60,
        "",
    ]

    # Add experiment overview if provided
    if overview:
        success_pct = (
            overview.successful_runs / overview.total_runs * 100
            if overview.total_runs > 0
            else 0
        )
        failed_pct = (
            overview.failed_runs / overview.total_runs * 100
            if overview.total_runs > 0
            else 0
        )

        lines.extend(
            [
                "## Experiment Overview",
                "",
                "| Parameter | Value |",
                "|-----------|-------|",
            ]
        )

        if overview.model:
            lines.append(f"| Model | {overview.model} |")

        arms_str = ", ".join(overview.arms) if overview.arms else "N/A"
        lines.append(f"| Arms | {arms_str} |")
        lines.append(f"| Instances | {overview.instance_count} |")
        lines.append(
            f"| Total Runs | {overview.total_runs} "
            f"({len(overview.arms)} arms × {overview.instance_count} instances) |"
        )
        lines.append(f"| Successful | {overview.successful_runs} ({success_pct:.1f}%) |")
        lines.append(f"| Failed | {overview.failed_runs} ({failed_pct:.1f}%) |")

        if overview.total_tokens > 0:
            if overview.total_tokens >= 1_000_000:
                token_str = f"{overview.total_tokens / 1_000_000:.2f}M"
            else:
                token_str = f"{overview.total_tokens:,}"
            lines.append(f"| Total Tokens | {token_str} |")

        if overview.avg_wall_time > 0:
            lines.append(f"| Avg Wall Time | {overview.avg_wall_time:.1f}s |")

        if overview.avg_init_time > 0:
            lines.append(f"| Avg Init Time | {overview.avg_init_time:.1f}s |")

        if overview.avg_workflow_time > 0:
            lines.append(f"| Avg Workflow Time | {overview.avg_workflow_time:.1f}s |")

        lines.append("")

    lines.append(f"Total failed runs: {len(errors)}")
    lines.append("")

    # Summary table first
    lines.extend(
        [
            "## Summary",
            "",
            "| Arm | Total Failures | Final Guard Breakdown |",
            "|-----|----------------|----------------------|",
        ]
    )

    for arm in sorted(by_arm.keys()):
        arm_errors = by_arm[arm]
        guard_counts: dict[str, int] = defaultdict(int)
        for err in arm_errors:
            guard_counts[err.failed_guard] += 1

        breakdown = ", ".join(
            f"{g.replace('Guard', '')}: {c}" for g, c in sorted(guard_counts.items())
        )
        lines.append(f"| {arm} | {len(arm_errors)} | {breakdown} |")

    lines.append("")

    # Detailed tables by arm
    for arm in sorted(by_arm.keys()):
        arm_errors = by_arm[arm]
        lines.extend(
            [
                f"## {arm} ({len(arm_errors)} runs ended in failure)",
                "",
            ]
        )

        # Build detailed table with all retries
        for err in arm_errors:
            # Shorten instance_id for display
            short_id = err.instance_id
            if len(short_id) > 40:
                # Extract repo and commit prefix
                parts = short_id.split("-")
                if len(parts) >= 2:
                    short_id = f"{parts[0][:20]}...{parts[-1][:8]}"

            lines.append(f"### {short_id}")
            lines.append("")
            lines.append(f"**Failed at**: {err.failed_step} | **Retries**: {err.retry_count}")
            lines.append("")
            lines.append("| Retry | Action Pair | Guard | Error |")
            lines.append("|-------|-------------|-------|-------|")

            if err.all_retries:
                for retry in err.all_retries:
                    is_final = retry.attempt == len(err.all_retries)
                    marker = " **← FINAL**" if is_final else ""
                    guard_display = f"**{retry.guard}**" if is_final else retry.guard
                    error_display = retry.error_summary.replace("|", "\\|")[:80]
                    if is_final:
                        error_display = f"**{error_display}**"
                    lines.append(
                        f"| [{retry.attempt}] | {retry.action_pair} | "
                        f"{guard_display} | {error_display}{marker} |"
                    )
            else:
                # No artifact data, just show final
                lines.append(
                    f"| [{err.retry_count}] | {err.failed_step} | **{err.failed_guard}** | "
                    f"**{err.error_summary[:80]}** ← FINAL |"
                )

            lines.append("")

    report = "\n".join(lines) + "\n"

    # Write timestamped file and update symlink
    base_path = output_dir / "final_error_analysis.md"
    timestamped_path = get_timestamped_path(base_path, ".md")
    timestamped_path.write_text(report)
    update_latest_symlink(timestamped_path, "final_error_analysis_latest.md")

    logger.info("Report written to %s", timestamped_path)
    return report


def save_final_errors_json(
    errors: list[FinalError],
    output_dir: str | Path,
) -> Path:
    """Save final errors as JSON for further analysis.

    Args:
        errors: List of FinalError objects
        output_dir: Directory to write JSON

    Returns:
        Path to the written file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = []
    for err in errors:
        entry = {
            "instance_id": err.instance_id,
            "arm": err.arm,
            "failed_step": err.failed_step,
            "failed_guard": err.failed_guard,
            "retry_count": err.retry_count,
            "error_summary": err.error_summary,
            "retries": [
                {
                    "attempt": r.attempt,
                    "guard": r.guard,
                    "error_summary": r.error_summary,
                }
                for r in err.all_retries
            ],
        }
        data.append(entry)

    base_path = output_dir / "final_errors.json"
    timestamped_path = get_timestamped_path(base_path, ".json")
    with open(timestamped_path, "w") as f:
        json.dump(data, f, indent=2)
    update_latest_symlink(timestamped_path, "final_errors_latest.json")

    logger.info("JSON written to %s", timestamped_path)
    return timestamped_path


def analyze_final_errors(
    results_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Main entry point for final error analysis.

    Args:
        results_path: Path to results.jsonl file
        output_dir: Output directory (defaults to same as results)

    Returns:
        Analysis summary dict
    """
    results_path = Path(results_path)
    out_dir = results_path.parent if output_dir is None else Path(output_dir)

    # Compute experiment overview (reads all results)
    overview = compute_experiment_overview(results_path)

    errors = extract_final_errors(results_path)

    if not errors:
        logger.warning("No failed runs found in %s", results_path)
        return {"total_failures": 0}

    # Generate report with overview
    generate_final_error_report(errors, out_dir, overview=overview)

    # Save JSON
    save_final_errors_json(errors, out_dir)

    # Compute summary stats
    by_arm: dict[str, list[FinalError]] = defaultdict(list)
    by_guard: dict[str, int] = defaultdict(int)
    for err in errors:
        by_arm[err.arm].append(err)
        by_guard[err.failed_guard] += 1

    return {
        "total_failures": len(errors),
        "by_arm": {arm: len(errs) for arm, errs in by_arm.items()},
        "by_guard": dict(by_guard),
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m examples.swe_bench_pro.final_error_analysis <results.jsonl>")
        sys.exit(1)

    results_path = sys.argv[1]
    summary = analyze_final_errors(results_path)
    print(f"\nAnalysis complete: {summary}")
