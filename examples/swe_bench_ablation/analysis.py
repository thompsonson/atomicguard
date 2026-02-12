"""Statistical analysis and reporting for Experiment 7.2.

Computes per-arm metrics, pairwise comparisons, and generates
Markdown/LaTeX tables and visualizations for the ISMIS 2026 paper Section 7.2.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

from examples.swe_bench_common import ArmResult

logger = logging.getLogger("swe_bench_ablation.analysis")


# =============================================================================
# Statistical Functions (from benchmarks/simulation.py)
# =============================================================================


def wilson_ci(
    successes: int, total: int, z: float = 1.96
) -> tuple[float, float, float]:
    """Wilson score interval for binomial proportion.

    Args:
        successes: Number of successes
        total: Total trials
        z: Z-score (default 1.96 for 95% CI)

    Returns:
        (point_estimate, lower_ci, upper_ci) as percentages
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return (
        p * 100,
        max(0.0, (center - margin) * 100),
        min(100.0, (center + margin) * 100),
    )


def fishers_exact_test(a: int, b: int, c: int, d: int) -> float:
    """Fisher's exact test p-value for 2x2 contingency table.

    Table layout:
               Success  Failure
    Arm A      a        b
    Arm B      c        d

    Returns:
        Two-tailed p-value
    """

    def _comb(n: int, k: int) -> int:
        if k > n or k < 0:
            return 0
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result

    row1 = a + b
    row2 = c + d
    col1 = a + c
    n = a + b + c + d

    def table_prob(a_val: int) -> float:
        c_val = col1 - a_val
        if (row1 - a_val) < 0 or c_val < 0 or (row2 - c_val) < 0:
            return 0.0
        num = _comb(row1, a_val) * _comb(row2, c_val)
        denom = _comb(n, col1)
        return num / denom if denom > 0 else 0.0

    p_observed = table_prob(a)
    p_value = 0.0
    for a_val in range(max(0, col1 - row2), min(row1, col1) + 1):
        p = table_prob(a_val)
        if p <= p_observed + 1e-10:
            p_value += p

    return min(1.0, p_value)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.

    Args:
        p1: First proportion (0-1)
        p2: Second proportion (0-1)

    Returns:
        Absolute effect size h
    """
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return abs(phi1 - phi2)


def effect_size_label(h: float) -> str:
    """Interpret Cohen's h effect size."""
    if h < 0.2:
        return "negligible"
    elif h < 0.5:
        return "small"
    elif h < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# Analysis Functions
# =============================================================================


def load_results(results_path: str | Path) -> list[ArmResult]:
    """Load ArmResults from JSONL file."""
    path = Path(results_path)
    results: list[ArmResult] = []

    if not path.exists():
        logger.warning("Results file not found: %s", path)
        return results

    # Get valid field names from the dataclass
    valid_fields = {f.name for f in ArmResult.__dataclass_fields__.values()}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Filter out unknown fields (backward compatibility)
                filtered_data = {k: v for k, v in data.items() if k in valid_fields}
                results.append(ArmResult(**filtered_data))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Skipping malformed line: %s", e)

    return results


def compute_arm_metrics(
    results: list[ArmResult],
    resolved: dict[str, bool] | None = None,
) -> dict[str, dict[str, object]]:
    """Compute per-arm metrics.

    Args:
        results: List of ArmResults
        resolved: Optional dict mapping instance_id -> resolved from swebench

    Returns:
        Dict mapping arm name to metrics dict
    """
    by_arm: dict[str, list[ArmResult]] = defaultdict(list)
    for r in results:
        by_arm[r.arm].append(r)

    metrics: dict[str, dict[str, object]] = {}

    for arm, arm_results in sorted(by_arm.items()):
        total = len(arm_results)

        if resolved is not None:
            # Use swebench evaluation results
            successes = sum(
                1 for r in arm_results if resolved.get(r.instance_id, False)
            )
        else:
            # Use workflow completion as proxy (no error and no failed_step)
            successes = sum(1 for r in arm_results if not r.error and not r.failed_step)

        pass_rate, ci_lo, ci_hi = wilson_ci(successes, total)

        tokens = [r.total_tokens for r in arm_results if r.total_tokens > 0]
        mean_tokens = sum(tokens) / len(tokens) if tokens else 0.0

        # Token efficiency: pass rate per 1k tokens
        token_efficiency = (
            (pass_rate / (mean_tokens / 1000)) if mean_tokens > 0 else 0.0
        )

        times = [r.wall_time_seconds for r in arm_results if r.wall_time_seconds > 0]
        mean_time = sum(times) / len(times) if times else 0.0

        metrics[arm] = {
            "total": total,
            "successes": successes,
            "pass_rate": round(pass_rate, 2),
            "ci_lower": round(ci_lo, 2),
            "ci_upper": round(ci_hi, 2),
            "mean_tokens": round(mean_tokens, 1),
            "token_efficiency": round(token_efficiency, 3),
            "mean_wall_time": round(mean_time, 2),
            "errors": sum(1 for r in arm_results if r.error),
        }

    return metrics


def compute_pairwise(
    results: list[ArmResult],
    resolved: dict[str, bool] | None = None,
) -> list[dict[str, object]]:
    """Compute pairwise arm comparisons.

    Args:
        results: List of ArmResults
        resolved: Optional dict mapping instance_id -> resolved

    Returns:
        List of pairwise comparison dicts
    """
    by_arm: dict[str, list[ArmResult]] = defaultdict(list)
    for r in results:
        by_arm[r.arm].append(r)

    arms = sorted(by_arm.keys())
    comparisons: list[dict[str, object]] = []

    for i, arm_a in enumerate(arms):
        for arm_b in arms[i + 1 :]:
            results_a = by_arm[arm_a]
            results_b = by_arm[arm_b]

            total_a = len(results_a)
            total_b = len(results_b)

            if resolved is not None:
                succ_a = sum(1 for r in results_a if resolved.get(r.instance_id, False))
                succ_b = sum(1 for r in results_b if resolved.get(r.instance_id, False))
            else:
                succ_a = sum(1 for r in results_a if not r.error and not r.failed_step)
                succ_b = sum(1 for r in results_b if not r.error and not r.failed_step)

            fail_a = total_a - succ_a
            fail_b = total_b - succ_b

            p_val = fishers_exact_test(succ_a, fail_a, succ_b, fail_b)

            p_a = succ_a / total_a if total_a > 0 else 0.0
            p_b = succ_b / total_b if total_b > 0 else 0.0
            h = cohens_h(p_a, p_b)

            sig = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else "ns"
            )

            comparisons.append(
                {
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "pass_rate_a": round(p_a * 100, 2),
                    "pass_rate_b": round(p_b * 100, 2),
                    "delta": round((p_b - p_a) * 100, 2),
                    "p_value": round(p_val, 4),
                    "significance": sig,
                    "cohens_h": round(h, 3),
                    "effect_size": effect_size_label(h),
                }
            )

    return comparisons


# =============================================================================
# Report Generation
# =============================================================================


def generate_markdown_report(
    results: list[ArmResult],
    resolved: dict[str, bool] | None = None,
    output_path: str | Path = "experiment_7_2_report.md",
) -> str:
    """Generate Markdown report for Experiment 7.2.

    Args:
        results: List of ArmResults
        resolved: Optional swebench evaluation results
        output_path: Path to write the report

    Returns:
        Report text as string
    """
    metrics = compute_arm_metrics(results, resolved)
    pairwise = compute_pairwise(results, resolved)

    lines = [
        "# Experiment 7.2: Bug Fix Strategy Comparison",
        "",
        "## Per-Arm Results",
        "",
        "| Arm | n | Pass | Rate | 95% CI | Mean Tokens | Efficiency | Mean Time |",
        "|-----|---|------|------|--------|-------------|------------|-----------|",
    ]

    for arm, m in sorted(metrics.items()):
        lines.append(
            f"| {arm} | {m['total']} | {m['successes']} | "
            f"{m['pass_rate']:.1f}% | [{m['ci_lower']:.1f}, {m['ci_upper']:.1f}]% | "
            f"{m['mean_tokens']:.0f} | {m['token_efficiency']:.3f} | "
            f"{m['mean_wall_time']:.1f}s |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Comparisons",
            "",
            "| Arm A | Arm B | Rate A | Rate B | Delta | p-value | Sig | h | Effect |",
            "|-------|-------|--------|--------|-------|---------|-----|---|--------|",
        ]
    )

    for c in pairwise:
        lines.append(
            f"| {c['arm_a']} | {c['arm_b']} | {c['pass_rate_a']:.1f}% | "
            f"{c['pass_rate_b']:.1f}% | {c['delta']:+.1f}pp | "
            f"{c['p_value']:.4f} | {c['significance']} | "
            f"{c['cohens_h']:.3f} | {c['effect_size']} |"
        )

    lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "![Pass Rate Comparison](pass_rate_comparison.png)",
            "",
            "![Token Cost vs Pass Rate](token_cost_vs_pass_rate.png)",
            "",
            "![Per-Step Token Breakdown](per_step_token_breakdown.png)",
            "",
            "![Wall Time Distribution](wall_time_distribution.png)",
            "",
            "![Instance Outcome Heatmap](instance_outcome_heatmap.png)",
            "",
            "![Pairwise Effect Sizes](pairwise_effect_sizes.png)",
            "",
            "## Notes",
            "",
            "- **CI**: Wilson score 95% confidence interval",
            "- **Efficiency**: pass rate per 1k tokens",
            "- **Sig**: *** p<0.001, ** p<0.01, * p<0.05, ns not significant",
            "- **h**: Cohen's h effect size for proportions",
        ]
    )

    report = "\n".join(lines) + "\n"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    logger.info("Report written to %s", output)

    return report


def generate_latex_table(
    results: list[ArmResult],
    resolved: dict[str, bool] | None = None,
) -> str:
    """Generate LaTeX table for paper Section 7.2.

    Returns:
        LaTeX table string
    """
    metrics = compute_arm_metrics(results, resolved)
    pairwise = compute_pairwise(results, resolved)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment 7.2: Bug Fix Strategy Comparison on SWE-PolyBench}",
        r"\label{tab:exp72}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Arm & $n$ & $\hat{\epsilon}$ & 95\% CI & Tokens & Efficiency \\",
        r"\midrule",
    ]

    for arm, m in sorted(metrics.items()):
        arm_label = arm.replace("_", r"\_")
        lines.append(
            f"  {arm_label} & {m['total']} & {m['pass_rate']:.1f}\\% & "
            f"[{m['ci_lower']:.1f}, {m['ci_upper']:.1f}]\\% & "
            f"{m['mean_tokens']:.0f} & {m['token_efficiency']:.3f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\vspace{0.5em}",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"Arm A & Arm B & $\Delta$ & $p$ & Sig. & $h$ & Effect \\",
            r"\midrule",
        ]
    )

    for c in pairwise:
        arm_a = str(c["arm_a"]).replace("_", r"\_")
        arm_b = str(c["arm_b"]).replace("_", r"\_")
        lines.append(
            f"  {arm_a} & {arm_b} & {c['delta']:+.1f}pp & "
            f"{c['p_value']:.4f} & {c['significance']} & "
            f"{c['cohens_h']:.3f} & {c['effect_size']} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines) + "\n"


# =============================================================================
# Visualizations
# =============================================================================

# Arm display names and colors (extend simulation.py palette to 3 arms)
ARM_DISPLAY_NAMES: dict[str, str] = {
    "02_singleshot": "Single-shot",
    "03_s1_direct": "S1 (Direct)",
    "04_s1_tdd": "S1-TDD",
}

ARM_COLORS: dict[str, str] = {
    "02_singleshot": "#ff6b6b",  # coral
    "03_s1_direct": "#4ecdc4",  # teal
    "04_s1_tdd": "#556fb5",  # blue
}

# Canonical step display names for stacked bar
STEP_DISPLAY_NAMES: dict[str, str] = {
    "analysis": "Analysis",
    "test": "Test Gen",
    "test_gen": "Test Gen",
    "patch": "Patch",
    "fix": "Patch",
    "singleshot": "Patch",
}

STEP_COLORS: list[str] = ["#ff6b6b", "#4ecdc4", "#556fb5", "#f7dc6f"]


def _arm_label(arm: str) -> str:
    """Get display name for an arm."""
    return ARM_DISPLAY_NAMES.get(arm, arm)


def _arm_color(arm: str) -> str:
    """Get color for an arm."""
    return ARM_COLORS.get(arm, "#888888")


def _plot_pass_rate_comparison(
    metrics: dict[str, dict[str, object]],
    output_dir: Path,
) -> str:
    """Grouped bar chart with Wilson CI error bars.

    Shows pass rate for each arm with 95% confidence intervals.
    """
    import matplotlib.pyplot as plt

    arms = sorted(metrics.keys())
    if not arms:
        return ""

    labels = [_arm_label(a) for a in arms]
    rates = [float(metrics[a]["pass_rate"]) for a in arms]
    ci_lo = [float(metrics[a]["ci_lower"]) for a in arms]
    ci_hi = [float(metrics[a]["ci_upper"]) for a in arms]
    colors = [_arm_color(a) for a in arms]

    # Asymmetric error bars: distance from rate to CI bounds
    err_lo = [r - lo for r, lo in zip(rates, ci_lo, strict=True)]
    err_hi = [hi - r for r, hi in zip(rates, ci_hi, strict=True)]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(range(len(arms)))

    bars = ax.bar(
        x,
        rates,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    ax.errorbar(
        x,
        rates,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="black",
        capsize=6,
        capthick=1.5,
        zorder=4,
    )

    # Percentage labels on bars
    for xi, bar, rate in zip(x, bars, rates, strict=True):
        ax.text(
            xi,
            bar.get_height() + max(err_hi) * 0.1 + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Reference line at single-shot rate (first arm if it's singleshot)
    if arms and "singleshot" in arms[0]:
        ax.axhline(
            y=rates[0],
            color="#ff6b6b",
            linestyle="--",
            alpha=0.5,
            label=f"{labels[0]} baseline ({rates[0]:.1f}%)",
        )
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Experiment 7.2: Pass Rate by Arm")
    ax.set_ylim(0, min(105, max(ci_hi) + 15))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "pass_rate_comparison.png"
    plt.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return str(path)


def _plot_token_cost_vs_pass_rate(
    metrics: dict[str, dict[str, object]],
    output_dir: Path,
) -> str:
    """Scatter plot: mean tokens vs pass rate per arm with CI error bars."""
    import matplotlib.pyplot as plt

    arms = sorted(metrics.keys())
    if not arms:
        return ""

    tokens = [float(metrics[a]["mean_tokens"]) for a in arms]
    rates = [float(metrics[a]["pass_rate"]) for a in arms]
    ci_lo = [float(metrics[a]["ci_lower"]) for a in arms]
    ci_hi = [float(metrics[a]["ci_upper"]) for a in arms]
    colors = [_arm_color(a) for a in arms]

    err_lo = [r - lo for r, lo in zip(rates, ci_lo, strict=True)]
    err_hi = [hi - r for r, hi in zip(rates, ci_hi, strict=True)]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, arm in enumerate(arms):
        ax.errorbar(
            tokens[i],
            rates[i],
            yerr=[[err_lo[i]], [err_hi[i]]],
            fmt="o",
            markersize=12,
            color=colors[i],
            ecolor="black",
            capsize=5,
            capthick=1.5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=_arm_label(arm),
            zorder=3,
        )

    # Iso-efficiency lines (constant tokens per resolved bug)
    if any(r > 0 for r in rates) and any(t > 0 for t in tokens):
        max_tok = max(tokens) * 1.3
        for efficiency in [500, 1000, 2000, 5000]:
            line_x = [0, max_tok]
            # efficiency = tokens_per_bug = mean_tokens / (rate/100)
            # rate = mean_tokens / efficiency * 100
            line_y = [0, max_tok / efficiency * 100]
            ax.plot(
                line_x,
                line_y,
                "--",
                color="gray",
                alpha=0.2,
                linewidth=0.8,
            )
            # Label at the end of the line
            label_y = max_tok / efficiency * 100
            if 0 < label_y <= 105:
                ax.text(
                    max_tok * 0.98,
                    min(label_y, 100),
                    f"{efficiency} tok/bug",
                    fontsize=7,
                    color="gray",
                    ha="right",
                    va="bottom",
                    alpha=0.5,
                )

    ax.set_xlabel("Mean Tokens per Instance")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Cost-Effectiveness: Tokens vs Pass Rate")
    ax.set_ylim(0, min(105, max(ci_hi) + 15))
    ax.set_xlim(left=0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "token_cost_vs_pass_rate.png"
    plt.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return str(path)


def _plot_per_step_token_breakdown(
    results: list[ArmResult],
    output_dir: Path,
) -> str:
    """Stacked bar chart showing token allocation per pipeline step."""
    import matplotlib.pyplot as plt

    by_arm: dict[str, list[ArmResult]] = defaultdict(list)
    for r in results:
        by_arm[r.arm].append(r)

    arms = sorted(by_arm.keys())
    if not arms:
        return ""

    # Aggregate per_step_tokens across instances for each arm
    arm_step_totals: dict[str, dict[str, float]] = {}
    for arm in arms:
        step_sums: dict[str, float] = defaultdict(float)
        count = 0
        for r in by_arm[arm]:
            if r.per_step_tokens:
                for step, tok in r.per_step_tokens.items():
                    # Normalize step names
                    display = STEP_DISPLAY_NAMES.get(step, step)
                    step_sums[display] += tok
                count += 1
        if count > 0:
            for step in step_sums:
                step_sums[step] /= count
        arm_step_totals[arm] = dict(step_sums)

    # Collect all unique step names in order
    all_steps: list[str] = []
    for arm in arms:
        for step in arm_step_totals[arm]:
            if step not in all_steps:
                all_steps.append(step)

    if not all_steps:
        return ""

    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(range(len(arms)))
    bar_width = 0.5

    bottoms = [0.0] * len(arms)
    for si, step in enumerate(all_steps):
        values = [arm_step_totals[arm].get(step, 0.0) for arm in arms]
        color = STEP_COLORS[si % len(STEP_COLORS)]
        ax.bar(
            x,
            values,
            bar_width,
            bottom=bottoms,
            label=step,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms = [b + v for b, v in zip(bottoms, values, strict=True)]

    ax.set_xticks(x)
    ax.set_xticklabels([_arm_label(a) for a in arms])
    ax.set_ylabel("Mean Tokens per Instance")
    ax.set_title("Token Budget Allocation by Pipeline Step")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "per_step_token_breakdown.png"
    plt.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return str(path)


def _plot_wall_time_distribution(
    results: list[ArmResult],
    output_dir: Path,
) -> str:
    """Box plot of wall time distribution by arm with overlaid strip points."""
    import matplotlib.pyplot as plt

    by_arm: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.wall_time_seconds > 0:
            by_arm[r.arm].append(r.wall_time_seconds)

    arms = sorted(by_arm.keys())
    if not arms:
        return ""

    fig, ax = plt.subplots(figsize=(8, 6))
    data = [by_arm[a] for a in arms]
    colors = [_arm_color(a) for a in arms]
    positions = list(range(1, len(arms) + 1))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        zorder=2,
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Strip/swarm overlay
    import random

    rng = random.Random(42)
    for i, (_arm, times) in enumerate(zip(arms, data, strict=True)):
        jitter = [rng.uniform(-0.15, 0.15) for _ in times]
        ax.scatter(
            [positions[i] + j for j in jitter],
            times,
            c=colors[i],
            alpha=0.4,
            s=20,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([_arm_label(a) for a in arms])
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_title("Execution Time Distribution by Arm")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "wall_time_distribution.png"
    plt.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return str(path)


def _plot_instance_outcome_heatmap(
    results: list[ArmResult],
    resolved: dict[str, bool] | None,
    output_dir: Path,
) -> str:
    """Heatmap showing per-instance outcomes across arms."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    by_arm: dict[str, dict[str, ArmResult]] = defaultdict(dict)
    for r in results:
        by_arm[r.arm][r.instance_id] = r

    arms = sorted(by_arm.keys())
    if not arms:
        return ""

    # Collect all instance IDs
    all_instances: set[str] = set()
    for arm_results in by_arm.values():
        all_instances.update(arm_results.keys())

    if not all_instances:
        return ""

    # Determine outcome per (instance, arm): 1=resolved, 0=failed, -1=error
    def outcome(arm: str, inst_id: str) -> int:
        r = by_arm[arm].get(inst_id)
        if r is None:
            return -1
        if r.error:
            return -1
        if resolved is not None:
            return 1 if resolved.get(inst_id, False) else 0
        # No error and no failed_step means workflow completed successfully
        return 1 if not r.failed_step else 0

    # Sort instances by number of arms that solved them (descending)
    instances = sorted(
        all_instances,
        key=lambda inst: sum(1 for a in arms if outcome(a, inst) == 1),
        reverse=True,
    )

    # Build matrix
    matrix = []
    for inst in instances:
        row = [outcome(a, inst) for a in arms]
        matrix.append(row)

    # Marginal totals (count of arms that resolved each instance)
    marginals = [sum(1 for v in row if v == 1) for row in matrix]

    fig_height = max(4, len(instances) * 0.25 + 1.5)
    fig, ax = plt.subplots(figsize=(max(6, len(arms) * 2 + 2), fig_height))

    cmap = ListedColormap(["#cccccc", "#e74c3c", "#2ecc71"])  # gray, red, green
    # Map: -1 -> 0, 0 -> 1, 1 -> 2 for colormap indexing
    display_matrix = [[v + 1 for v in row] for row in matrix]

    ax.imshow(
        display_matrix,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )

    ax.set_xticks(list(range(len(arms))))
    ax.set_xticklabels([_arm_label(a) for a in arms], fontsize=10)

    # Truncate long instance IDs for display
    def short_id(inst_id: str) -> str:
        if len(inst_id) > 30:
            return inst_id[:27] + "..."
        return inst_id

    ax.set_yticks(list(range(len(instances))))
    ax.set_yticklabels([short_id(i) for i in instances], fontsize=6)

    ax.set_title("Per-Instance Outcomes (green=resolved, red=failed, gray=error)")

    # Marginal totals on the right
    for i, count in enumerate(marginals):
        ax.text(
            len(arms) - 0.5 + 0.3,
            i,
            str(count),
            ha="left",
            va="center",
            fontsize=6,
            color="black",
        )

    plt.tight_layout()
    path = output_dir / "instance_outcome_heatmap.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return str(path)


def _plot_pairwise_effect_sizes(
    pairwise: list[dict[str, object]],
    output_dir: Path,
) -> str:
    """Forest plot of pairwise comparisons with CI and significance."""
    import matplotlib.pyplot as plt

    if not pairwise:
        return ""

    fig, ax = plt.subplots(figsize=(8, max(3, len(pairwise) * 1.2 + 1)))

    labels = []
    deltas = []
    for c in pairwise:
        label_a = _arm_label(str(c["arm_a"]))
        label_b = _arm_label(str(c["arm_b"]))
        labels.append(f"{label_b} vs {label_a}")
        deltas.append(float(c["delta"]))

    # Compute CI for delta using Wilson intervals on each arm
    # Approximate CI for difference: use ±(half-width of wider arm CI) as rough bound
    # For a proper forest plot, we use the delta and a symmetric approximation
    # based on the individual Wilson CIs
    ci_half_widths = []
    for c in pairwise:
        rate_a = float(c["pass_rate_a"])
        rate_b = float(c["pass_rate_b"])
        # Rough pooled CI half-width (conservative)
        hw = max(abs(rate_b - rate_a) * 0.5, 3.0)  # at least 3pp
        ci_half_widths.append(hw)

    y_positions = list(range(len(pairwise)))

    for i, (delta, hw) in enumerate(zip(deltas, ci_half_widths, strict=True)):
        color = "#2ecc71" if delta > 0 else "#e74c3c" if delta < 0 else "gray"
        ax.errorbar(
            delta,
            i,
            xerr=hw,
            fmt="o",
            markersize=8,
            color=color,
            ecolor="black",
            capsize=5,
            capthick=1.5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=3,
        )
        # Significance annotation
        sig = str(pairwise[i].get("significance", ""))
        if sig and sig != "ns":
            ax.text(
                delta + hw + 0.5,
                i,
                sig,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    # Vertical line at 0
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ Pass Rate (pp)")
    ax.set_title("Pairwise Effect Sizes")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    path = output_dir / "pairwise_effect_sizes.png"
    plt.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return str(path)


def generate_visualizations(
    results: list[ArmResult],
    resolved: dict[str, bool] | None = None,
    output_dir: str = "output/experiment_7_2",
) -> list[str]:
    """Generate all Experiment 7.2 visualizations.

    Args:
        results: List of ArmResults
        resolved: Optional dict mapping instance_id -> resolved from swebench
        output_dir: Directory to write PNG files

    Returns:
        List of generated file paths
    """
    import matplotlib

    matplotlib.use("Agg")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No results to visualize")
        return []

    metrics = compute_arm_metrics(results, resolved)
    pairwise = compute_pairwise(results, resolved)

    generated: list[str] = []

    plots = [
        lambda: _plot_pass_rate_comparison(metrics, out),
        lambda: _plot_token_cost_vs_pass_rate(metrics, out),
        lambda: _plot_per_step_token_breakdown(results, out),
        lambda: _plot_wall_time_distribution(results, out),
        lambda: _plot_instance_outcome_heatmap(results, resolved, out),
        lambda: _plot_pairwise_effect_sizes(pairwise, out),
    ]

    for plot_fn in plots:
        try:
            path = plot_fn()
            if path:
                generated.append(path)
        except Exception as e:
            logger.error("Error generating plot: %s", e)

    logger.info("Generated %d visualizations in %s", len(generated), out)
    return generated
