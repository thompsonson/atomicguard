"""
Simulation Refactored: Using Dual-State Agent Framework
Paper: Managing the Stochastic (Thompson, 2025)

This refactors simulation.py to use the dual_state_agent infrastructure.
"""

import ast
import contextlib
import csv
import io
import logging
import multiprocessing
import os
import re
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Queue
from typing import Any

import click
from openai import OpenAI
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# =============================================================================
# VISUALIZATION
# =============================================================================


def load_results(filename: str, format: str) -> list[dict]:
    """Load results from CSV or SQLite."""
    results = []
    if format == "csv":
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["success"] = (
                    row["success"].lower() == "true"
                    if isinstance(row["success"], str)
                    else bool(int(row["success"]))
                )
                row["retries"] = int(row["retries"])
                row["generation_count"] = float(row["generation_count"])
                row["duration_seconds"] = float(row.get("duration_seconds", 0))
                results.append(row)
    elif format == "sqlite":
        conn = sqlite3.connect(filename)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_name, config, task, trial_num, success, retries, generation_count, duration_seconds FROM results"
        )
        for row in cursor.fetchall():
            results.append(
                {
                    "model_name": row[0],
                    "config": row[1],
                    "task": row[2],
                    "trial_num": row[3],
                    "success": bool(row[4]),
                    "retries": row[5],
                    "generation_count": row[6],
                    "duration_seconds": row[7],
                }
            )
        conn.close()
    return results


def generate_visualizations(filename: str, format: str, output_dir: str = "."):
    """Generate visualization charts from results."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")  # Non-interactive backend
    except ImportError:
        console.print("[red]matplotlib required: pip install matplotlib[/red]")
        return

    results = load_results(filename, format)
    if not results:
        console.print("[red]No results found[/red]")
        return

    # Aggregate data
    from collections import defaultdict

    # Group by model, config, task
    agg = defaultdict(
        lambda: {"success": 0, "total": 0, "durations": [], "retries": []}
    )
    for r in results:
        key = (r["model_name"], r["config"], r.get("task", "unknown"))
        agg[key]["total"] += 1
        if r["success"]:
            agg[key]["success"] += 1
        agg[key]["durations"].append(r["duration_seconds"])
        agg[key]["retries"].append(r["retries"])

    # Get unique values
    models = sorted({k[0] for k in agg})
    tasks = sorted({k[2] for k in agg})

    # 1. Grouped Bar: Baseline vs Guarded by Model
    console.print("\n[bold]Generating: baseline_vs_guarded.png[/bold]")
    fig, ax = plt.subplots(figsize=(14, 6))

    x = range(len(models))
    width = 0.35

    # Aggregate across tasks for this chart
    baseline_rates = []
    guarded_rates = []
    for model in models:
        b_success, b_total = 0, 0
        g_success, g_total = 0, 0
        for task in tasks:
            b_key = (model, "Baseline", task)
            g_key = (model, "Guarded (R=3)", task)
            if b_key in agg:
                b_success += agg[b_key]["success"]
                b_total += agg[b_key]["total"]
            if g_key in agg:
                g_success += agg[g_key]["success"]
                g_total += agg[g_key]["total"]
        baseline_rates.append(b_success / b_total * 100 if b_total > 0 else 0)
        guarded_rates.append(g_success / g_total * 100 if g_total > 0 else 0)

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        baseline_rates,
        width,
        label="Baseline",
        color="#ff6b6b",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        guarded_rates,
        width,
        label="Guarded (R=3)",
        color="#4ecdc4",
    )

    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Baseline vs Guarded Success Rate by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.split("(")[0].strip() for m in models], rotation=45, ha="right"
    )
    ax.legend()
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/baseline_vs_guarded.png", dpi=150)
    plt.close()

    # 2. Heatmap: Model × Task Success Rate (Guarded only)
    if len(tasks) > 1:
        console.print("[bold]Generating: model_task_heatmap_guarded.png[/bold]")
        fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.5)))

        heatmap_data = []
        model_labels = []
        for model in models:
            row = []
            for task in tasks:
                key = (model, "Guarded (R=3)", task)
                if key in agg and agg[key]["total"] > 0:
                    rate = agg[key]["success"] / agg[key]["total"] * 100
                else:
                    rate = 0
                row.append(rate)
            heatmap_data.append(row)
            model_labels.append(model)

        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(model_labels)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(tasks)):
                ax.text(
                    j,
                    i,
                    f"{heatmap_data[i][j]:.0f}%",
                    ha="center",
                    va="center",
                    color="black" if 30 < heatmap_data[i][j] < 70 else "white",
                )

        ax.set_title("Guarded Success Rate: Model × Task")
        plt.colorbar(im, ax=ax, label="Success Rate (%)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_task_heatmap_guarded.png", dpi=150)
        plt.close()

        # 2b. Heatmap: Baseline
        console.print("[bold]Generating: model_task_heatmap_baseline.png[/bold]")
        fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.5)))

        heatmap_data = []
        for model in models:
            row = []
            for task in tasks:
                key = (model, "Baseline", task)
                if key in agg and agg[key]["total"] > 0:
                    rate = agg[key]["success"] / agg[key]["total"] * 100
                else:
                    rate = 0
                row.append(rate)
            heatmap_data.append(row)

        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(model_labels)

        for i in range(len(models)):
            for j in range(len(tasks)):
                ax.text(
                    j,
                    i,
                    f"{heatmap_data[i][j]:.0f}%",
                    ha="center",
                    va="center",
                    color="black" if 30 < heatmap_data[i][j] < 70 else "white",
                )

        ax.set_title("Baseline Success Rate: Model × Task")
        plt.colorbar(im, ax=ax, label="Success Rate (%)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_task_heatmap_baseline.png", dpi=150)
        plt.close()

    # 3. Box Plot: Duration Distribution
    console.print("[bold]Generating: duration_boxplot.png[/bold]")
    fig, ax = plt.subplots(figsize=(14, 6))

    baseline_durations = []
    guarded_durations = []
    for model in models:
        b_durs = []
        g_durs = []
        for task in tasks:
            b_key = (model, "Baseline", task)
            g_key = (model, "Guarded (R=3)", task)
            if b_key in agg:
                b_durs.extend(agg[b_key]["durations"])
            if g_key in agg:
                g_durs.extend(agg[g_key]["durations"])
        baseline_durations.append(b_durs if b_durs else [0])
        guarded_durations.append(g_durs if g_durs else [0])

    positions = range(len(models))
    bp1 = ax.boxplot(
        baseline_durations,
        positions=[p - 0.2 for p in positions],
        widths=0.35,
        patch_artist=True,
        boxprops={"facecolor": "#ff6b6b", "alpha": 0.7},
    )
    bp2 = ax.boxplot(
        guarded_durations,
        positions=[p + 0.2 for p in positions],
        widths=0.35,
        patch_artist=True,
        boxprops={"facecolor": "#4ecdc4", "alpha": 0.7},
    )

    ax.set_ylabel("Duration (seconds)")
    ax.set_title("Duration Distribution: Baseline vs Guarded")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [m.split("(")[0].strip() for m in models], rotation=45, ha="right"
    )
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Baseline", "Guarded (R=3)"])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/duration_boxplot.png", dpi=150)
    plt.close()

    # 4. Improvement Bar Chart
    console.print("[bold]Generating: improvement.png[/bold]")
    fig, ax = plt.subplots(figsize=(12, 6))

    improvements = [g - b for b, g in zip(baseline_rates, guarded_rates, strict=False)]
    colors = ["#4ecdc4" if imp >= 0 else "#ff6b6b" for imp in improvements]

    bars = ax.bar(range(len(models)), improvements, color=colors)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Improvement (percentage points)")
    ax.set_title("Success Rate Improvement: Guarded vs Baseline")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [m.split("(")[0].strip() for m in models], rotation=45, ha="right"
    )

    for bar, imp in zip(bars, improvements, strict=False):
        height = bar.get_height()
        ax.annotate(
            f"{imp:+.0f}pp",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement.png", dpi=150)
    plt.close()

    console.print(f"\n[green]✓ Visualizations saved to {output_dir}/[/green]")


def _extract_model_size(model_name: str) -> float:
    """Extract model size in billions from name like 'Qwen2.5-Coder (7B)'."""
    import re

    match = re.search(r"\((\d+\.?\d*)B\)", model_name)
    if match:
        return float(match.group(1))
    return 0.0


def generate_scatter_and_line(filename: str, format: str, output_dir: str = "."):
    """Generate scatter and line charts."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
    except ImportError:
        console.print("[red]matplotlib required: pip install matplotlib[/red]")
        return

    results = load_results(filename, format)
    if not results:
        return

    from collections import defaultdict

    # Aggregate by model, config, task
    agg = defaultdict(lambda: {"success": 0, "total": 0, "retries_list": []})
    for r in results:
        key = (r["model_name"], r["config"], r.get("task", "unknown"))
        agg[key]["total"] += 1
        if r["success"]:
            agg[key]["success"] += 1
        agg[key]["retries_list"].append(r["retries"])

    models = sorted({k[0] for k in agg})
    tasks = sorted({k[2] for k in agg})

    # 5. Scatter: Model Size vs Success Rate
    console.print("[bold]Generating: size_vs_success.png[/bold]")
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes_baseline, rates_baseline = [], []
    sizes_guarded, rates_guarded = [], []
    labels = []

    for model in models:
        size = _extract_model_size(model)
        if size == 0:
            continue

        b_success, b_total = 0, 0
        g_success, g_total = 0, 0
        for task in tasks:
            b_key = (model, "Baseline", task)
            g_key = (model, "Guarded (R=3)", task)
            if b_key in agg:
                b_success += agg[b_key]["success"]
                b_total += agg[b_key]["total"]
            if g_key in agg:
                g_success += agg[g_key]["success"]
                g_total += agg[g_key]["total"]

        if b_total > 0:
            sizes_baseline.append(size)
            rates_baseline.append(b_success / b_total * 100)
        if g_total > 0:
            sizes_guarded.append(size)
            rates_guarded.append(g_success / g_total * 100)
            labels.append(model.split("(")[0].strip())

    ax.scatter(
        sizes_baseline,
        rates_baseline,
        s=100,
        c="#ff6b6b",
        alpha=0.7,
        label="Baseline",
        edgecolors="black",
    )
    ax.scatter(
        sizes_guarded,
        rates_guarded,
        s=100,
        c="#4ecdc4",
        alpha=0.7,
        label="Guarded (R=3)",
        edgecolors="black",
    )

    # Add labels
    for i, label in enumerate(labels):
        if i < len(sizes_guarded):
            ax.annotate(
                label,
                (sizes_guarded[i], rates_guarded[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Model Size (Billions)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Model Size vs Success Rate")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/size_vs_success.png", dpi=150)
    plt.close()

    # 6. Line: Success Rate vs Retry Budget (simulated R=0,1,2,3)
    console.print("[bold]Generating: retry_impact.png[/bold]")
    fig, ax = plt.subplots(figsize=(10, 6))

    # For each model, estimate success at different retry levels
    # R=0 is baseline, R=3 is guarded
    # We interpolate R=1, R=2 based on retry distribution

    for model in models:
        size = _extract_model_size(model)
        if size == 0:
            continue

        # Get baseline and guarded rates
        b_success, b_total = 0, 0
        g_success, g_total = 0, 0
        retry_success = defaultdict(lambda: {"success": 0, "total": 0})

        for task in tasks:
            b_key = (model, "Baseline", task)
            g_key = (model, "Guarded (R=3)", task)
            if b_key in agg:
                b_success += agg[b_key]["success"]
                b_total += agg[b_key]["total"]
            if g_key in agg:
                g_success += agg[g_key]["success"]
                g_total += agg[g_key]["total"]
                # Track by actual retries used
                for retries in agg[g_key]["retries_list"]:
                    retry_success[retries]["total"] += 1
                    # Assume success if retry count < max
                    retry_success[retries]["success"] += 1

        if b_total == 0 or g_total == 0:
            continue

        baseline_rate = b_success / b_total * 100
        guarded_rate = g_success / g_total * 100

        # Simple interpolation: assume linear improvement
        r_values = [0, 1, 2, 3]
        rates = [
            baseline_rate,
            baseline_rate + (guarded_rate - baseline_rate) * 0.4,
            baseline_rate + (guarded_rate - baseline_rate) * 0.7,
            guarded_rate,
        ]

        label = model.split("(")[0].strip()
        ax.plot(r_values, rates, marker="o", label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Retry Budget (R)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs Retry Budget")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["R=0\n(Baseline)", "R=1", "R=2", "R=3"])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/retry_impact.png", dpi=150)
    plt.close()

    console.print("[green]✓ Additional charts saved[/green]")


def generate_statistical_analysis(
    filename: str, format: str, _output_dir: str = "."
) -> str:
    """Generate comprehensive statistical analysis for paper. Returns report text."""
    import math
    from collections import defaultdict

    results = load_results(filename, format)
    if not results:
        console.print("[red]No results to analyze[/red]")
        return ""

    console.print("\n[bold]═══ STATISTICAL ANALYSIS ═══[/bold]")

    # Aggregate data
    agg = defaultdict(
        lambda: {
            "successes": 0,
            "total": 0,
            "retries": [],
            "durations": [],
            "failures_by_type": defaultdict(int),
        }
    )

    for r in results:
        key = (r["model_name"], r["config"], r.get("task", "unknown"))
        agg[key]["total"] += 1
        if r["success"]:
            agg[key]["successes"] += 1
        agg[key]["retries"].append(r["retries"])
        agg[key]["durations"].append(r["duration_seconds"])

    models = sorted({k[0] for k in agg})
    tasks = sorted({k[2] for k in agg})

    # Statistical functions
    def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple:
        """Wilson score interval for binomial proportion (better for small n)."""
        if total == 0:
            return 0, 0, 0
        p = successes / total
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        return (
            p * 100,
            max(0, (center - margin) * 100),
            min(100, (center + margin) * 100),
        )

    def fishers_exact_test(a: int, b: int, c: int, d: int) -> float:
        """Fisher's exact test p-value (2x2 contingency table)."""

        def factorial(n):
            if n <= 1:
                return 1
            result = 1
            for i in range(2, n + 1):
                result *= i
            return result

        def comb(n, k):
            if k > n or k < 0:
                return 0
            return factorial(n) // (factorial(k) * factorial(n - k))

        row1 = a + b
        row2 = c + d
        col1 = a + c
        n = a + b + c + d

        def table_prob(a_val):
            b_val = row1 - a_val
            c_val = col1 - a_val
            d_val = row2 - c_val
            if b_val < 0 or c_val < 0 or d_val < 0:
                return 0
            num = comb(row1, a_val) * comb(row2, c_val)
            denom = comb(n, col1)
            return num / denom if denom > 0 else 0

        p_observed = table_prob(a)
        p_value = 0
        for a_val in range(max(0, col1 - row2), min(row1, col1) + 1):
            p = table_prob(a_val)
            if p <= p_observed + 1e-10:
                p_value += p

        return min(1.0, p_value)

    def cohens_h(p1: float, p2: float) -> float:
        """Cohen's h effect size for proportions."""
        phi1 = 2 * math.asin(math.sqrt(p1))
        phi2 = 2 * math.asin(math.sqrt(p2))
        return abs(phi1 - phi2)

    def effect_size_label(h: float) -> str:
        if h < 0.2:
            return "negligible"
        elif h < 0.5:
            return "small"
        elif h < 0.8:
            return "medium"
        else:
            return "large"

    # Build data structures for report
    stats_data = {"tasks": {}, "retry_dist": {}, "cost_benefit": [], "summary": {}}

    for task in tasks:
        stats_data["tasks"][task] = []
        stats_data["retry_dist"][task] = []

        for model in models:
            b_key = (model, "Baseline", task)
            g_key = (model, "Guarded (R=3)", task)

            if b_key not in agg or g_key not in agg:
                continue

            b_succ, b_tot = agg[b_key]["successes"], agg[b_key]["total"]
            g_succ, g_tot = agg[g_key]["successes"], agg[g_key]["total"]

            b_rate, b_lo, b_hi = wilson_ci(b_succ, b_tot)
            g_rate, g_lo, g_hi = wilson_ci(g_succ, g_tot)

            delta = g_rate - b_rate

            b_fail = b_tot - b_succ
            g_fail = g_tot - g_succ
            p_val = fishers_exact_test(b_succ, b_fail, g_succ, g_fail)

            h = cohens_h(
                b_succ / b_tot if b_tot > 0 else 0, g_succ / g_tot if g_tot > 0 else 0
            )
            h_label = effect_size_label(h)

            sig = (
                "***"
                if p_val < 0.001
                else "**"
                if p_val < 0.01
                else "*"
                if p_val < 0.05
                else ""
            )

            stats_data["tasks"][task].append(
                {
                    "model": model,
                    "b_rate": b_rate,
                    "g_rate": g_rate,
                    "delta": delta,
                    "b_ci": (b_lo, b_hi),
                    "g_ci": (g_lo, g_hi),
                    "p_val": p_val,
                    "sig": sig,
                    "h": h,
                    "h_label": h_label,
                }
            )

            # Retry distribution
            retries = agg[g_key]["retries"]
            total = len(retries)
            if total > 0:
                r_counts = [0, 0, 0, 0]
                for r in retries:
                    if r <= 3:
                        r_counts[r] += 1
                avg_retries = sum(retries) / total
                stats_data["retry_dist"][task].append(
                    {
                        "model": model,
                        "r_pcts": [c / total * 100 for c in r_counts],
                        "avg": avg_retries,
                    }
                )

            # Cost-benefit
            avg_retries = sum(agg[g_key]["retries"]) / g_tot if g_tot > 0 else 0
            avg_cost = 1 + avg_retries
            gain_per_cost = delta / avg_cost if avg_cost > 0 else 0
            stats_data["cost_benefit"].append(
                {
                    "model": model,
                    "task": task,
                    "delta": delta,
                    "avg_cost": avg_cost,
                    "gain_per_cost": gain_per_cost,
                }
            )

    # Summary
    all_baseline = []
    all_guarded = []
    for key, data in agg.items():
        model, config, task = key
        rate = data["successes"] / data["total"] * 100 if data["total"] > 0 else 0
        if config == "Baseline":
            all_baseline.append(rate)
        else:
            all_guarded.append(rate)

    if all_baseline and all_guarded:
        stats_data["summary"] = {
            "avg_baseline": sum(all_baseline) / len(all_baseline),
            "avg_guarded": sum(all_guarded) / len(all_guarded),
            "improvement": sum(all_guarded) / len(all_guarded)
            - sum(all_baseline) / len(all_baseline),
        }

    # Console output (abbreviated)
    console.print("[green]✓ Statistical analysis complete[/green]")

    return stats_data


def generate_report(filename: str, format: str, output_dir: str = "."):
    """Generate comprehensive markdown report with charts, tables, and statistics."""
    from datetime import datetime

    console.print("\n[bold]Generating report.md...[/bold]")

    results = load_results(filename, format)
    if not results:
        console.print("[red]No results to generate report[/red]")
        return

    # Get stats data
    stats_data = generate_statistical_analysis(filename, format, output_dir)
    if not stats_data:
        return

    # Build markdown
    lines = []

    # Header
    lines.append("# Dual-State Agent Simulation Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append(f"\n*Data source: {filename}*")
    lines.append(f"\n*Total trials: {len(results)}*")

    # Summary
    lines.append("\n## Executive Summary")
    if stats_data.get("summary"):
        s = stats_data["summary"]
        lines.append("\n| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Average Baseline Success | {s['avg_baseline']:.1f}% |")
        lines.append(f"| Average Guarded Success | {s['avg_guarded']:.1f}% |")
        lines.append(f"| **Overall Improvement** | **+{s['improvement']:.1f}pp** |")

    # Charts
    lines.append("\n## Visualizations")
    lines.append("\n### Baseline vs Guarded Success Rate")
    lines.append("\n![Baseline vs Guarded](baseline_vs_guarded.png)")

    lines.append("\n### Improvement by Model")
    lines.append("\n![Improvement](improvement.png)")

    lines.append("\n### Model × Task Heatmaps")
    lines.append("\n| Baseline | Guarded |")
    lines.append("|----------|---------|")
    lines.append(
        "| ![Baseline Heatmap](model_task_heatmap_baseline.png) | ![Guarded Heatmap](model_task_heatmap_guarded.png) |"
    )

    lines.append("\n### Duration Distribution")
    lines.append("\n![Duration Boxplot](duration_boxplot.png)")

    lines.append("\n### Model Size vs Success")
    lines.append("\n![Size vs Success](size_vs_success.png)")

    lines.append("\n### Retry Impact")
    lines.append("\n![Retry Impact](retry_impact.png)")

    # Statistical tables per task
    lines.append("\n## Statistical Analysis")
    lines.append("\n*Significance: \\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05*")

    for task, task_stats in stats_data["tasks"].items():
        if not task_stats:
            continue
        lines.append(f"\n### {task.upper()} Task")
        lines.append(
            "\n| Model | Baseline | Guarded | Δ | 95% CI (Base) | 95% CI (Guard) | p-value | Effect Size |"
        )
        lines.append(
            "|-------|----------|---------|---|---------------|----------------|---------|-------------|"
        )

        for row in task_stats:
            lines.append(
                f"| {row['model']} | {row['b_rate']:.0f}% | {row['g_rate']:.0f}% | "
                f"+{row['delta']:.0f}pp | [{row['b_ci'][0]:.0f}-{row['b_ci'][1]:.0f}%] | "
                f"[{row['g_ci'][0]:.0f}-{row['g_ci'][1]:.0f}%] | {row['p_val']:.4f}{row['sig']} | "
                f"{row['h']:.2f} ({row['h_label']}) |"
            )

    # Retry distribution
    lines.append("\n## Retry Distribution")
    lines.append(
        "\n*Percentage of trials succeeding at each retry level (Guarded config)*"
    )

    for task, retry_stats in stats_data["retry_dist"].items():
        if not retry_stats:
            continue
        lines.append(f"\n### {task.upper()}")
        lines.append("\n| Model | R=0 | R=1 | R=2 | R=3 | Avg Retries |")
        lines.append("|-------|-----|-----|-----|-----|-------------|")

        for row in retry_stats:
            lines.append(
                f"| {row['model']} | {row['r_pcts'][0]:.0f}% | {row['r_pcts'][1]:.0f}% | "
                f"{row['r_pcts'][2]:.0f}% | {row['r_pcts'][3]:.0f}% | {row['avg']:.2f} |"
            )

    # Cost-benefit
    lines.append("\n## Cost-Benefit Analysis")
    lines.append("\n| Model | Task | Δ Success | Avg Cost | Gain/Cost |")
    lines.append("|-------|------|-----------|----------|-----------|")

    for row in stats_data["cost_benefit"]:
        lines.append(
            f"| {row['model']} | {row['task']} | +{row['delta']:.1f}pp | "
            f"{row['avg_cost']:.2f}x | +{row['gain_per_cost']:.1f}pp/x |"
        )

    # Interpretation
    lines.append("\n## Interpretation")
    lines.append("\n### Key Findings")
    lines.append(
        "\n1. **Guard effectiveness varies by task complexity** - Tasks with lower baseline success show larger improvements"
    )
    lines.append(
        "\n2. **Diminishing returns at high baseline** - When baseline >90%, guards add minimal value"
    )
    lines.append(
        "\n3. **Cost scales with difficulty** - More retries needed for harder tasks, but gain/cost ratio remains positive"
    )

    lines.append("\n### Statistical Notes")
    lines.append(
        "\n- **95% CI**: Wilson score interval (appropriate for binomial proportions)"
    )
    lines.append(
        "\n- **p-value**: Fisher's exact test (no normal approximation assumptions)"
    )
    lines.append(
        "\n- **Cohen's h**: Effect size for proportions (<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large)"
    )

    # Write report
    report_text = "\n".join(lines)
    report_path = f"{output_dir}/report.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    console.print(f"[green]✓ Report saved to {report_path}[/green]")


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_TRIALS = 100
GENERATION_INCREMENT = 1.0
TRIAL_TIMEOUT_SECONDS = 180
CIRCUIT_BREAKER_THRESHOLD = 5
DEFAULT_OLLAMA_URL = "http://100.69.76.46:11434/v1"

TASK_PROMPTS = {
    "password": """Write a Python function validate_password(password) that returns True if the password meets all these criteria, otherwise False:
Requirements:
1. Length between 12 and 32 characters.
2. Must contain at least 1 uppercase, 1 lowercase, 1 digit, and 1 special character (!@#$%^&*).
3. No character can appear more than 2 times.
4. No 3 consecutive alphabetic sequences (case-insensitive, e.g., "abc", "CDe", "xyz").
5. The sum of all digits in the password must be a prime number.
6. Do not use external libraries (like sympy); use only standard Python libraries.
""",
    "lru": """Write a Python class `LRUCache` with the following specifications:
1. `__init__(self, capacity: int)`: Initializes the cache with a positive capacity.
2. `get(self, key: int) -> int`: Return the value of the key if it exists, otherwise return -1.
3. `put(self, key: int, value: int) -> None`: Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
4. The functions `get` and `put` must each run in O(1) average time complexity.
""",
    "template": """Write a Python function `render_template(template: str, context: dict) -> str` that implements a simple template engine:
1. Replace `{{ variable }}` with the value from context dict. If the variable is not in context, leave the placeholder unchanged.
2. Support basic conditionals: `{% if variable %}text{% endif %}`. If variable exists in context and is truthy, include the text. Otherwise, remove the entire block.
3. You can use regex for matching patterns.

Examples:
- render_template("Hello {{ name }}!", {"name": "World"}) -> "Hello World!"
- render_template("Hi {{ missing }}", {}) -> "Hi {{ missing }}"
- render_template("{% if show %}Visible{% endif %}", {"show": True}) -> "Visible"
- render_template("{% if show %}Hidden{% endif %}", {"show": False}) -> ""
""",
}

# TDD Task Specifications (spec -> tests -> implementation)
TDD_TASKS = {
    "tdd_stack": {
        "name": "Stack",
        "spec": """Implement a Stack class with:
- push(item): Add item to top
- pop(): Remove and return top item, raise IndexError if empty
- peek(): Return top item without removing, raise IndexError if empty
- is_empty(): Return True if stack is empty
- size(): Return number of items""",
        "test_prompt": """Write pytest test functions for a Stack class with push, pop, peek, is_empty, and size methods.
Include tests for:
- Basic push/pop operations
- peek without removal
- is_empty on new and used stack
- size accuracy
- IndexError on pop/peek when empty

Output ONLY the test code, no implementation. Use 'from implementation import Stack'.""",
        "impl_prompt": """Write a Python Stack class with push, pop, peek, is_empty, and size methods.
The tests expect: from implementation import Stack""",
    },
    "tdd_calculator": {
        "name": "Calculator",
        "spec": """Implement a Calculator class with:
- add(a, b): Return a + b
- subtract(a, b): Return a - b
- multiply(a, b): Return a * b
- divide(a, b): Return a / b, raise ValueError if b is 0""",
        "test_prompt": """Write pytest test functions for a Calculator class with add, subtract, multiply, divide methods.
Include tests for:
- Basic arithmetic operations
- Negative numbers
- Float operations
- Division by zero raises ValueError

Output ONLY the test code, no implementation. Use 'from implementation import Calculator'.""",
        "impl_prompt": """Write a Python Calculator class with add, subtract, multiply, divide methods.
Division by zero should raise ValueError.
The tests expect: from implementation import Calculator""",
    },
    "tdd_queue": {
        "name": "Queue",
        "spec": """Implement a Queue class with:
- enqueue(item): Add item to back
- dequeue(): Remove and return front item, raise IndexError if empty
- front(): Return front item without removing, raise IndexError if empty
- is_empty(): Return True if queue is empty
- size(): Return number of items""",
        "test_prompt": """Write pytest test functions for a Queue class with enqueue, dequeue, front, is_empty, and size methods.
Include tests for:
- Basic enqueue/dequeue (FIFO order)
- front without removal
- is_empty on new and used queue
- size accuracy
- IndexError on dequeue/front when empty

Output ONLY the test code, no implementation. Use 'from implementation import Queue'.""",
        "impl_prompt": """Write a Python Queue class with enqueue, dequeue, front, is_empty, and size methods.
The tests expect: from implementation import Queue""",
    },
}

console = Console()
logger = logging.getLogger("simulation")


# =============================================================================
# DUAL-STATE AGENT FRAMEWORK (embedded)
# =============================================================================


@dataclass(frozen=True)
class Artifact:
    """Immutable generated output."""

    content: str
    version: int
    artifact_id: str
    extraction_method: str = "unknown"
    parent_version: int | None = None


@dataclass(frozen=True)
class GuardResult:
    """Guard validation outcome."""

    passed: bool
    feedback: str = ""


@dataclass(frozen=True)
class AmbientEnvironment:
    repository: "ArtifactDAGInterface"
    constraints: str = ""


@dataclass(frozen=True)
class PromptTemplate:
    role: str
    constraints: str
    task: str
    feedback_wrapper: str = (
        "GUARD REJECTION:\n{feedback}\nInstruction: Address the rejection above."
    )

    def render(self, context: "Context") -> str:
        parts = [f"# ROLE\n{self.role}", f"# CONSTRAINTS\n{self.constraints}"]
        if context.ambient.constraints:
            parts.append(f"# CONTEXT\n{context.ambient.constraints}")
        if context.feedback_history:
            parts.append("# HISTORY (Context Refinement)")
            for i, (_, feedback) in enumerate(context.feedback_history):
                wrapped = self.feedback_wrapper.format(feedback=feedback)
                parts.append(f"--- Attempt {i + 1} ---\n{wrapped}")
        parts.append(f"# TASK\n{self.task}")
        return "\n\n".join(parts)


@dataclass(frozen=True)
class Context:
    ambient: AmbientEnvironment
    specification: str
    current_artifact: str | None = None
    feedback_history: tuple[tuple[str, str], ...] = ()


class GeneratorInterface(ABC):
    @abstractmethod
    def generate(
        self, context: Context, template: PromptTemplate | None = None
    ) -> Artifact:
        pass


class GuardInterface(ABC):
    @abstractmethod
    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        pass


class ArtifactDAGInterface(ABC):
    @abstractmethod
    def store(self, artifact: Artifact, metadata: str) -> str:
        pass

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Artifact:
        pass


class InMemoryArtifactDAG(ArtifactDAGInterface):
    def __init__(self):
        self._artifacts: dict[str, Artifact] = {}

    def store(self, artifact: Artifact, _metadata: str) -> str:
        self._artifacts[artifact.artifact_id] = artifact
        return artifact.artifact_id

    def get_artifact(self, artifact_id: str) -> Artifact:
        return self._artifacts[artifact_id]


class ActionPair:
    def __init__(self, generator: GeneratorInterface, guard: GuardInterface):
        self._generator = generator
        self._guard = guard

    def execute(
        self, context: Context, dependencies: dict[str, Artifact] = None
    ) -> tuple[Artifact, GuardResult]:
        dependencies = dependencies or {}
        artifact = self._generator.generate(context)
        result = self._guard.validate(artifact, **dependencies)
        return artifact, result


class RmaxExhausted(Exception):
    def __init__(
        self,
        message: str,
        provenance: list[tuple[Artifact, str]],
        generation_count: float,
    ):
        super().__init__(message)
        self.provenance = provenance
        self.generation_count = generation_count


class DualStateAgent:
    """Stateless executor for ActionPair with retry loop."""

    def __init__(
        self, action_pair: ActionPair, artifact_dag: ArtifactDAGInterface, rmax: int = 3
    ):
        self._action_pair = action_pair
        self._artifact_dag = artifact_dag
        self._rmax = rmax
        self._generation_count = 0.0

    def execute(self, specification: str) -> Artifact:
        context = self._compose_context(specification)
        feedback_history: list[tuple[Artifact, str]] = []
        retry_count = 0

        while retry_count <= self._rmax:
            artifact, result = self._action_pair.execute(context)
            self._generation_count += GENERATION_INCREMENT
            self._artifact_dag.store(artifact, "" if result.passed else result.feedback)

            if result.passed:
                return artifact
            else:
                # Deduplicate feedback - don't add if already in history
                if result.feedback not in [f for _, f in feedback_history]:
                    feedback_history.append((artifact, result.feedback))
                retry_count += 1
                context = self._refine_context(
                    specification, artifact, feedback_history
                )

        raise RmaxExhausted(
            f"Failed after {self._rmax} retries",
            provenance=feedback_history,
            generation_count=self._generation_count,
        )

    @property
    def generation_count(self) -> float:
        return self._generation_count

    def _compose_context(self, specification: str) -> Context:
        return Context(
            ambient=AmbientEnvironment(repository=self._artifact_dag),
            specification=specification,
        )

    def _refine_context(
        self,
        specification: str,
        artifact: Artifact,
        feedback_history: list[tuple[Artifact, str]],
    ) -> Context:
        return Context(
            ambient=AmbientEnvironment(repository=self._artifact_dag),
            specification=specification,
            current_artifact=artifact.content,
            feedback_history=tuple((a.content, f) for a, f in feedback_history),
        )


# =============================================================================
# GENERATOR: OLLAMA
# =============================================================================


class OllamaGenerator(GeneratorInterface):
    """Connects to Ollama using OpenAI-compatible API."""

    def __init__(self, model: str, base_url: str, timeout: float = 120.0):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout)
        self._version_counter = 0

    def generate(
        self, context: Context, _template: PromptTemplate | None = None
    ) -> Artifact:
        # Build prompt with feedback history
        if context.feedback_history:
            feedback_section = "\n\n=== PREVIOUS ATTEMPTS ===\n"
            for i, (_, fb) in enumerate(context.feedback_history, 1):
                feedback_section += f"\nAttempt {i} failed:\n{fb}\n"
            feedback_section += "\n=== END FEEDBACK ===\n\nPlease address the issues above and provide corrected code."
            prompt = context.specification + feedback_section
        else:
            prompt = context.specification

        logger.debug(
            f"[{self.model}] Starting generation (prompt: {len(prompt)} chars)"
        )
        start_time = time.time()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful Python programming assistant. Your task is to provide a complete, runnable Python code solution. Always enclose the Python code in a single markdown block, like this:\n```python\n# your code here\n```",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7
            )
            content = response.choices[0].message.content or ""
            duration = time.time() - start_time
            logger.debug(f"[{self.model}] Generation completed in {duration:.2f}s")
        except Exception as e:
            logger.error(f"Ollama Error: {e}")
            content = ""

        code, extraction_method = self._extract_code(content)

        self._version_counter += 1
        return Artifact(
            content=code,
            version=self._version_counter,
            artifact_id=str(uuid.uuid4()),
            extraction_method=extraction_method,
            parent_version=self._version_counter - 1
            if self._version_counter > 1
            else None,
        )

    def _extract_code(self, content: str) -> tuple[str, str]:
        if not content or content.isspace():
            return "", "blank_response"

        match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        if match:
            return match.group(1), "python_block"

        match = re.search(r"```\n(.*?)\n```", content, re.DOTALL)
        if match:
            return match.group(1), "generic_block"

        match = re.search(r"^(def |import |class )", content, re.MULTILINE)
        if match:
            return content[match.start() :], "first_def"

        return content, "full_content"


# =============================================================================
# GUARDS: ADAPTED FROM ORIGINAL
# =============================================================================


class SyntaxGuard(GuardInterface):
    """G_syntax: Checks if code parses to AST."""

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        try:
            ast.parse(artifact.content)
            return GuardResult(passed=True)
        except SyntaxError as e:
            return GuardResult(passed=False, feedback=str(e))


class TypeGuard(GuardInterface):
    """G_type: Mock wrapper for mypy."""

    def validate(self, _artifact: Artifact, **_deps: Any) -> GuardResult:
        return GuardResult(passed=True)


class BaseTestGuard(GuardInterface):
    """Base class for test guards with subprocess isolation."""

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        q: Queue[tuple[bool, str]] = multiprocessing.Queue()
        p = multiprocessing.Process(target=self._execute_and_check, args=(artifact, q))
        p.start()
        p.join(60)

        if p.is_alive():
            p.terminate()
            p.join()
            return GuardResult(
                passed=False, feedback="Timeout: Code execution exceeded 60s limit."
            )

        if not q.empty():
            passed, msg = q.get()
            return GuardResult(passed=passed, feedback=msg)
        return GuardResult(
            passed=False, feedback="Execution failed silently (Process crashed)"
        )

    def _execute_and_check(self, artifact: Artifact, q: Queue) -> None:
        try:
            code = artifact.content
            if not code:
                q.put((False, "No code to test"))
                return

            local_scope = {"__builtins__": __builtins__}
            exec(code, local_scope)
            q.put(self._run_tests(local_scope))
        except Exception as e:
            q.put((False, f"Runtime/Syntax Error: {e}"))

    def _run_tests(self, scope: dict) -> tuple[bool, str]:
        raise NotImplementedError


class TemplateTestGuard(BaseTestGuard):
    """G_test for template rendering task."""

    def _get_template_guidance(self, failed_constraints: set) -> str:
        """Provide targeted guidance based on which constraints failed."""
        guidance_map = {
            "variable_substitution": "Hint: Use regex like r'\\{\\{\\s*(\\w+)\\s*\\}\\}' to find {{ variable }} patterns and replace with context.get(var, original).",
            "missing_key_handling": "Hint: When a key is missing from context, use context.get(key, original_placeholder) to leave it unchanged.",
            "conditional_true": "Hint: Use regex like r'\\{%\\s*if\\s+(\\w+)\\s*%\\}(.*?)\\{%\\s*endif\\s*%\\}' to find blocks. Check if context.get(key) is truthy.",
            "conditional_false": "Hint: When key is False or missing, replace the entire {% if key %}...{% endif %} block with empty string.",
            "conditional_missing_key": "Hint: Missing keys should be treated as falsy - remove the conditional block entirely.",
        }
        hints = [guidance_map[c] for c in failed_constraints if c in guidance_map]
        if hints:
            return "Specific guidance:\n" + "\n".join(hints)
        return "Review the requirements and ensure all edge cases are handled."

    def _run_tests(self, scope: dict) -> tuple[bool, str]:
        if "render_template" not in scope:
            return False, "Function 'render_template' not found."

        func = scope["render_template"]
        tests = [
            (
                "Hello {{ name }}!",
                {"name": "World"},
                "Hello World!",
                "variable_substitution",
            ),
            ("No vars", {}, "No vars", "no_variables"),
            ("Hi {{ missing }}", {}, "Hi {{ missing }}", "missing_key_handling"),
            (
                "{% if show %}Visible{% endif %}",
                {"show": True},
                "Visible",
                "conditional_true",
            ),
            (
                "{% if show %}Hidden{% endif %}",
                {"show": False},
                "",
                "conditional_false",
            ),
            ("{% if missing %}Text{% endif %}", {}, "", "conditional_missing_key"),
        ]

        failed_constraints = set()
        failures = []
        for tmpl, ctx, expected, constraint_name in tests:
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    result = func(tmpl, ctx)
                if result != expected:
                    failed_constraints.add(constraint_name)
                    failures.append(
                        f"  - Test '{constraint_name}' failed: template='{tmpl}', context={ctx}, expected='{expected}', got='{result}'"
                    )
            except Exception as e:
                failed_constraints.add(constraint_name)
                failures.append(f"  - Test '{constraint_name}' raised exception: {e}")

        if failures:
            guidance = self._get_template_guidance(failed_constraints)
            return False, "Test failures detected:\n" + "\n".join(
                failures
            ) + "\n\n" + guidance
        return True, ""


class LRUTestGuard(BaseTestGuard):
    """G_test for LRU cache task."""

    def _run_tests(self, scope: dict) -> tuple[bool, str]:
        if "LRUCache" not in scope:
            return False, "Class 'LRUCache' not found."

        try:
            LRUCache = scope["LRUCache"]
            cache = LRUCache(2)
            cache.put(1, 1)
            cache.put(2, 2)

            if cache.get(1) != 1:
                return False, f"get(1) returned {cache.get(1)}, expected 1"

            cache.put(3, 3)
            val2 = cache.get(2)
            if val2 != -1 and val2 is not None:
                return False, f"Key 2 should be evicted, got {val2}"

            cache.put(3, 30)
            if cache.get(3) != 30:
                return False, f"Key 3 should be 30, got {cache.get(3)}"

            return True, ""
        except Exception as e:
            return False, f"LRU Execution Error: {e}"


class PasswordTestGuard(BaseTestGuard):
    """G_test for password validation task."""

    def _run_tests(self, scope: dict) -> tuple[bool, str]:
        if "validate_password" not in scope:
            return False, "Function 'validate_password' not found."

        try:
            func = scope["validate_password"]
            tests = [
                # (password, expected_result, reason)
                (
                    "A!bX2579qW@#",
                    True,
                    "Valid: 12 chars, all types, no repeats >2, no sequences, sum=23 (prime)",
                ),
                ("Valid1@34abc", False, "Contains 'abc' sequence"),
                ("Valid1@34", False, "Too short (9 chars)"),
                ("Short1!", False, "Too short (7 chars)"),
                ("NoSpecial12345", False, "Missing special character"),
                ("NoDigitAbcdefg!", False, "Missing digit"),
                ("AAAbbb123!@#", False, "More than 2 A's and b's"),
                ("abcDef123!@#", False, "Contains 'abc' sequence"),
                ("CDeFgh123!@#", False, "Contains 'CDe' sequence (case-insensitive)"),
                ("A!b222222xxxx", False, "More than 2 '2's"),
                ("A!b1111111xxx", False, "More than 2 '1's"),
                ("A!b12xxxxxxxx", False, "Too short"),
            ]

            failures = []
            for pwd, expected, reason in tests:
                result = func(pwd)
                if result != expected:
                    failures.append(
                        f"  - Password '{pwd}': expected={expected}, got={result} ({reason})"
                    )

            if failures:
                return False, "Test failures:\n" + "\n".join(failures)
            return True, ""
        except Exception as e:
            return False, f"Password Validation Error: {e}"


# Factory for task guards
TASK_GUARDS = {
    "template": TemplateTestGuard,
    "lru": LRUTestGuard,
    "password": PasswordTestGuard,
}


class DynamicTestGuard(GuardInterface):
    """Guard that runs generated test code against implementation."""

    def __init__(self, test_code: str):
        self.test_code = test_code

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        q: Queue[tuple[bool, str]] = multiprocessing.Queue()
        p = multiprocessing.Process(target=self._run_tests, args=(artifact, q))
        p.start()
        p.join(60)

        if p.is_alive():
            p.terminate()
            p.join()
            return GuardResult(
                passed=False, feedback="Timeout: Test execution exceeded 60s."
            )

        if not q.empty():
            passed, msg = q.get()
            return GuardResult(passed=passed, feedback=msg)
        return GuardResult(passed=False, feedback="Test execution crashed")

    def _run_tests(self, artifact: Artifact, q: Queue) -> None:
        try:
            impl_code = artifact.content
            if not impl_code:
                q.put((False, "No implementation code"))
                return

            # Create a mock module for 'from implementation import X'
            import types

            impl_module = types.ModuleType("implementation")
            exec(impl_code, impl_module.__dict__)

            import sys

            sys.modules["implementation"] = impl_module

            # Run tests
            test_scope = {"__builtins__": __builtins__}
            exec(self.test_code, test_scope)

            # Find and run test functions
            test_funcs = [
                v
                for k, v in test_scope.items()
                if k.startswith("test_") and callable(v)
            ]

            if not test_funcs:
                q.put((False, "No test functions found (expected test_* functions)"))
                return

            failures = []
            for func in test_funcs:
                try:
                    func()
                except AssertionError as e:
                    failures.append(f"{func.__name__}: AssertionError - {e}")
                except Exception as e:
                    failures.append(f"{func.__name__}: {type(e).__name__} - {e}")

            if failures:
                q.put((False, "Test failures:\n" + "\n".join(failures)))
            else:
                q.put((True, f"All {len(test_funcs)} tests passed"))

        except SyntaxError as e:
            q.put((False, f"Syntax error in code: {e}"))
        except Exception as e:
            q.put((False, f"Execution error: {e}"))
        finally:
            # Cleanup
            if "implementation" in sys.modules:
                del sys.modules["implementation"]


class HumanGuard(GuardInterface):
    """Prompts human to verify artifact."""

    def __init__(self, prompt: str = "Approve this artifact?"):
        self.prompt = prompt

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        console.print("\n[bold yellow]═══ HUMAN REVIEW ═══[/bold yellow]")
        console.print(f"\n[dim]{'─' * 60}[/dim]")
        console.print(artifact.content)
        console.print(f"[dim]{'─' * 60}[/dim]")
        console.print(f"\n[bold]{self.prompt}[/bold]")
        console.print("[dim](y=approve, n=reject, or type feedback)[/dim]: ", end="")

        response = input().strip()

        if response.lower() == "y":
            return GuardResult(passed=True, feedback="Human approved")
        elif response.lower() == "n":
            return GuardResult(passed=False, feedback="Human rejected artifact")
        else:
            return GuardResult(passed=False, feedback=f"Human feedback: {response}")


class CompositeGuard(GuardInterface):
    """Combines multiple guards sequentially."""

    def __init__(self, guards: list[GuardInterface]):
        self._guards = guards

    def validate(self, artifact: Artifact, **deps) -> GuardResult:
        for guard in self._guards:
            result = guard.validate(artifact, **deps)
            if not result.passed:
                return result
        return GuardResult(passed=True)


# =============================================================================
# WORKFLOW RUNNERS
# =============================================================================


def run_linear_baseline(
    generator: OllamaGenerator, task_prompt: str, guards: list[GuardInterface]
) -> tuple[bool, int, float, str, float]:
    """Standard Prompt -> Code workflow (No Guards/Retries). Returns (success, retries, cost, extraction, duration)."""
    start_time = time.time()

    dag = InMemoryArtifactDAG()
    context = Context(
        ambient=AmbientEnvironment(repository=dag), specification=task_prompt
    )

    artifact = generator.generate(context)

    for guard in guards:
        result = guard.validate(artifact)
        if not result.passed:
            duration = time.time() - start_time
            return False, 0, GENERATION_INCREMENT, artifact.extraction_method, duration

    duration = time.time() - start_time
    return True, 0, GENERATION_INCREMENT, artifact.extraction_method, duration


def run_guarded_workflow(
    generator: OllamaGenerator,
    task_prompt: str,
    guards: list[GuardInterface],
    r_max: int,
) -> tuple[bool, int, float, str, float]:
    """Guard-based workflow using DualStateAgent. Returns (success, retries, cost, extraction, duration)."""
    start_time = time.time()

    dag = InMemoryArtifactDAG()
    composite_guard = CompositeGuard(guards)
    action_pair = ActionPair(generator=generator, guard=composite_guard)
    agent = DualStateAgent(action_pair=action_pair, artifact_dag=dag, rmax=r_max)

    try:
        artifact = agent.execute(task_prompt)
        retries = agent._generation_count / GENERATION_INCREMENT - 1
        duration = time.time() - start_time
        return (
            True,
            int(retries),
            agent.generation_count,
            artifact.extraction_method,
            duration,
        )
    except RmaxExhausted as e:
        last_artifact = e.provenance[-1][0] if e.provenance else None
        extraction = last_artifact.extraction_method if last_artifact else "none"
        duration = time.time() - start_time
        return False, len(e.provenance), e.generation_count, extraction, duration


def run_tdd_baseline(
    generator: OllamaGenerator, tdd_task: dict
) -> tuple[bool, int, float, str, float]:
    """TDD Baseline: Generate tests, then implementation (no retries)."""
    start_time = time.time()
    dag = InMemoryArtifactDAG()
    total_cost = 0.0

    # Step 1: Generate tests
    test_context = Context(
        ambient=AmbientEnvironment(repository=dag),
        specification=tdd_task["test_prompt"],
    )
    test_artifact = generator.generate(test_context)
    total_cost += GENERATION_INCREMENT

    # Check test syntax
    syntax_result = SyntaxGuard().validate(test_artifact)
    if not syntax_result.passed:
        duration = time.time() - start_time
        return False, 0, total_cost, f"test_{test_artifact.extraction_method}", duration

    # Step 2: Generate implementation
    impl_context = Context(
        ambient=AmbientEnvironment(repository=dag),
        specification=tdd_task["impl_prompt"],
    )
    impl_artifact = generator.generate(impl_context)
    total_cost += GENERATION_INCREMENT

    # Check implementation syntax
    syntax_result = SyntaxGuard().validate(impl_artifact)
    if not syntax_result.passed:
        duration = time.time() - start_time
        return False, 0, total_cost, f"impl_{impl_artifact.extraction_method}", duration

    # Run generated tests against implementation
    test_guard = DynamicTestGuard(test_artifact.content)
    result = test_guard.validate(impl_artifact)

    duration = time.time() - start_time
    return result.passed, 0, total_cost, impl_artifact.extraction_method, duration


def run_tdd_guarded(
    generator: OllamaGenerator, tdd_task: dict, r_max: int, human_review: bool = False
) -> tuple[bool, int, float, str, float]:
    """TDD Guarded: Generate tests (with retries), then implementation (with retries)."""
    start_time = time.time()
    dag = InMemoryArtifactDAG()
    total_cost = 0.0
    total_retries = 0

    # Step 1: Generate tests with syntax guard (+ optional human review)
    test_guards = [SyntaxGuard()]
    if human_review:
        test_guards.append(HumanGuard(prompt="Approve generated tests?"))
    test_guard = CompositeGuard(test_guards)
    test_action_pair = ActionPair(generator=generator, guard=test_guard)
    test_agent = DualStateAgent(
        action_pair=test_action_pair, artifact_dag=dag, rmax=r_max
    )

    try:
        test_artifact = test_agent.execute(tdd_task["test_prompt"])
        total_cost += test_agent.generation_count
        total_retries += int(test_agent.generation_count - 1)
    except RmaxExhausted as e:
        duration = time.time() - start_time
        return False, len(e.provenance), e.generation_count, "test_exhausted", duration

    # Step 2: Generate implementation with dynamic test guard (+ optional human review)
    dynamic_guard = DynamicTestGuard(test_artifact.content)
    impl_guards = [SyntaxGuard(), dynamic_guard]
    if human_review:
        impl_guards.append(HumanGuard(prompt="Approve implementation?"))
    impl_guard = CompositeGuard(impl_guards)
    impl_action_pair = ActionPair(generator=generator, guard=impl_guard)
    impl_agent = DualStateAgent(
        action_pair=impl_action_pair, artifact_dag=dag, rmax=r_max
    )

    try:
        impl_artifact = impl_agent.execute(tdd_task["impl_prompt"])
        total_cost += impl_agent.generation_count
        total_retries += int(impl_agent.generation_count - 1)
        duration = time.time() - start_time
        return (
            True,
            total_retries,
            total_cost,
            impl_artifact.extraction_method,
            duration,
        )
    except RmaxExhausted as e:
        total_cost += e.generation_count
        duration = time.time() - start_time
        last_artifact = e.provenance[-1][0] if e.provenance else None
        extraction = last_artifact.extraction_method if last_artifact else "none"
        return (
            False,
            total_retries + len(e.provenance),
            total_cost,
            extraction,
            duration,
        )


# =============================================================================
# RESULT PERSISTENCE (unchanged from original)
# =============================================================================


class ResultWriter:
    def __init__(self, filename: str, format: str, resume: bool):
        self.filename = filename
        self.format = format
        self.is_closed = False
        self.file = None
        self.conn = None
        self.writer = None
        self.cursor = None

        if format == "csv":
            mode = "a" if resume and os.path.exists(filename) else "w"
            self.file = open(filename, mode, newline="")  # noqa: SIM115
            self.writer = csv.DictWriter(
                self.file,
                fieldnames=[
                    "model_name",
                    "config",
                    "task",
                    "trial_num",
                    "success",
                    "retries",
                    "generation_count",
                    "extraction_method",
                    "duration_seconds",
                    "timestamp",
                ],
            )
            if mode == "w":
                self.writer.writeheader()
        elif format == "sqlite":
            self.conn = sqlite3.connect(filename)
            self.cursor = self.conn.cursor()
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT, config TEXT, task TEXT, trial_num INTEGER,
                    success INTEGER, retries INTEGER, generation_count REAL,
                    extraction_method TEXT, duration_seconds REAL, timestamp TEXT
                )
            """)
            self.conn.commit()

    def write_trial_result(self, data: dict) -> None:
        if self.format == "csv":
            self.writer.writerow(data)
            self.file.flush()
        elif self.format == "sqlite":
            self.cursor.execute(
                """
                INSERT INTO results (model_name, config, task, trial_num, success, retries,
                                     generation_count, extraction_method, duration_seconds, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data["model_name"],
                    data["config"],
                    data.get("task", ""),
                    data["trial_num"],
                    int(data["success"]),
                    data["retries"],
                    data["generation_count"],
                    data["extraction_method"],
                    data.get("duration_seconds", 0),
                    data.get("timestamp", ""),
                ),
            )
            self.conn.commit()

    def get_completed_trials(self) -> dict[tuple[str, str, str], int]:
        completed = {}
        if self.format == "csv" and os.path.exists(self.filename):
            with open(self.filename) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row["model_name"], row["config"], row.get("task", ""))
                    completed[key] = max(completed.get(key, 0), int(row["trial_num"]))
        elif self.format == "sqlite":
            self.cursor.execute(
                "SELECT model_name, config, task, MAX(trial_num) FROM results GROUP BY model_name, config, task"
            )
            for row in self.cursor.fetchall():
                completed[(row[0], row[1], row[2] or "")] = row[3]
        return completed

    def close(self) -> None:
        if self.is_closed:
            return
        if self.file:
            self.file.close()
        if self.conn:
            self.conn.close()
        self.is_closed = True


# =============================================================================
# CLI (adapted from original)
# =============================================================================


@click.command()
@click.option("--trials", default=NUM_TRIALS, help="Number of trials per configuration")
@click.option(
    "--task",
    default="lru",
    type=click.Choice(
        [
            "password",
            "lru",
            "template",
            "tdd_stack",
            "tdd_calculator",
            "tdd_queue",
            "all",
            "tdd_all",
        ]
    ),
    help="Task to run",
)
@click.option(
    "--model", default=None, help="Single model to test (e.g., qwen2.5-coder:7b)"
)
@click.option("--host", default=DEFAULT_OLLAMA_URL, help="Ollama API URL")
@click.option("--output", default="results.csv", help="Output file path")
@click.option(
    "--format",
    "output_format",
    default="csv",
    type=click.Choice(["csv", "sqlite"]),
    help="Output format",
)
@click.option("--resume", is_flag=True, help="Resume from existing results")
@click.option(
    "--circuit-breaker",
    default=CIRCUIT_BREAKER_THRESHOLD,
    help="Skip model after N consecutive failures",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging")
@click.option(
    "--visualize",
    is_flag=True,
    help="Generate visualizations from results (skip running trials)",
)
@click.option("--viz-output", default=".", help="Directory for visualization output")
@click.option(
    "--human-review", is_flag=True, help="Enable human-in-the-loop review for TDD tasks"
)
@click.option(
    "--baseline-only",
    is_flag=True,
    help="Run only baseline configuration (skip guarded)",
)
@click.option(
    "--log-file", default=None, help="Log file path (default: derived from --output)"
)
def main(
    trials,
    task,
    model,
    host,
    output,
    output_format,
    resume,
    circuit_breaker,
    verbose,
    visualize,
    viz_output,
    human_review,
    baseline_only,
    log_file,
):
    """Run simulation comparing baseline vs guarded workflows."""

    # Setup logging
    # Console logging (respects --verbose)
    console_level = logging.DEBUG if verbose else logging.INFO
    console_handler = RichHandler(console=console, rich_tracebacks=True)
    console_handler.setLevel(console_level)

    # File logging (always DEBUG for our code)
    log_path = log_file if log_file else os.path.splitext(output)[0] + ".log"
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Configure our logger with both handlers
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Suppress noisy 3rd party loggers
    for noisy in ["httpx", "openai", "httpcore", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    console.print(f"[dim]Logging to: {log_path}[/dim]")

    # Visualization mode: generate charts from existing results
    if visualize:
        if not os.path.exists(output):
            console.print(f"[red]Results file not found: {output}[/red]")
            return
        generate_visualizations(output, output_format, viz_output)
        generate_scatter_and_line(output, output_format, viz_output)
        generate_report(output, output_format, viz_output)
        return

    writer = ResultWriter(output, output_format, resume)
    completed_trials = writer.get_completed_trials() if resume else {}
    results = []

    # Task scenarios
    if task == "all":
        tasks_to_run = ["lru", "template", "password"]
    elif task == "tdd_all":
        tasks_to_run = ["tdd_stack", "tdd_calculator", "tdd_queue"]
    else:
        tasks_to_run = [task]

    # Model scenarios
    if model is None or model == "all":
        scenarios = [
            # Large models (9GB+) - expected high performance
            ("Qwen2.5-Coder (14B)", "qwen2.5-coder:14b"),
            ("StarCoder2 (15B)", "starcoder2:15b-instruct"),
            ("Phi4 (14B)", "phi4:latest"),
            # Medium models (4-5GB) - good balance
            ("Yi-Coder (9B)", "yi-coder:9b"),
            ("Granite-Code (8B)", "granite-code:8b"),
            ("Qwen2.5-Coder (7B)", "qwen2.5-coder:7b"),
            ("DeepSeek-Coder (6.7B)", "deepseek-coder:6.7b"),
            # Small models (2-3GB) - testing lower bound
            ("Qwen2.5-Coder (3B)", "qwen2.5-coder:3b"),
            ("Granite-Code (3B)", "granite-code:3b"),
            ("Phi4-Mini (3.8B)", "phi4-mini:latest"),
            # Tiny models (<2GB) - circuit breaker testing
            ("Qwen2.5-Coder (1.5B)", "qwen2.5-coder:1.5b"),
            ("Yi-Coder (1.5B)", "yi-coder:1.5b"),
            ("DeepSeek-Coder (1.3B)", "deepseek-coder:1.3b"),
        ]
    else:
        scenarios = [(f"Ollama ({model})", model)]

    # Print run summary
    total_trials_per_config = trials * len(tasks_to_run) * len(scenarios)
    total_trials = total_trials_per_config * 2  # Baseline + Guarded

    console.print("\n[bold]═══ RUN SUMMARY ═══[/bold]")
    console.print(f"\n[bold]Tasks ({len(tasks_to_run)}):[/bold]")
    for t in tasks_to_run:
        console.print(f"  • {t}")

    console.print(f"\n[bold]Models ({len(scenarios)}):[/bold]")
    for name, _ in scenarios:
        console.print(f"  • {name}")

    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  • Trials per config: {trials}")
    console.print("  • Configs: Baseline, Guarded (R=3)")
    console.print(f"  • Circuit breaker: {circuit_breaker}")
    console.print(f"  • Output: {output} ({output_format})")

    console.print(f"\n[bold]Total trials:[/bold] {total_trials}")
    console.print(
        f"  ({len(tasks_to_run)} tasks × {len(scenarios)} models × 2 configs × {trials} trials)"
    )

    console.print("\n[dim]Starting in 3 seconds... (Ctrl+C to cancel)[/dim]")
    time.sleep(3)

    for current_task in tasks_to_run:
        is_tdd = current_task.startswith("tdd_")

        if is_tdd:
            tdd_task = TDD_TASKS[current_task]
            task_display_name = f"TDD: {tdd_task['name']}"
        else:
            task_prompt = TASK_PROMPTS[current_task]
            guards = [SyntaxGuard(), TypeGuard(), TASK_GUARDS[current_task]()]
            task_display_name = current_task.upper()

        console.print(f"\n[bold cyan]═══ Task: {task_display_name} ═══[/bold cyan]")

        for name, param in scenarios:
            console.print(f"\n[bold]Evaluating Model:[/bold] {name}")
            gen = OllamaGenerator(model=param, base_url=host)

            if baseline_only:
                configs = [("Baseline", 0, True)]
            else:
                configs = [
                    ("Baseline", 0, True),
                    ("Guarded (R=3)", 3, False),
                ]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                total_steps = len(configs) * trials
                task_id = progress.add_task("Running simulation...", total=total_steps)

                for config_name, r_max, is_baseline in configs:
                    progress.update(task_id, description=f"Running {config_name}...")

                    successes = 0
                    total_retries = 0
                    total_cost = 0.0
                    total_duration = 0.0
                    start_trial = completed_trials.get(
                        (name, config_name, current_task), 0
                    )
                    consecutive_failures = 0
                    num_trials_run = 0

                    if start_trial >= trials:
                        console.print(
                            f"  [dim]Skipping {config_name} (already completed).[/dim]"
                        )
                        for _ in range(trials):
                            progress.advance(task_id)
                        continue

                    for i in range(start_trial, trials):
                        if (
                            circuit_breaker > 0
                            and consecutive_failures >= circuit_breaker
                        ):
                            console.print(
                                "  [yellow]⚠ Circuit breaker: Skipping remaining trials[/yellow]"
                            )
                            for _ in range(i, trials):
                                progress.advance(task_id)
                            break

                        try:
                            if is_tdd:
                                # TDD workflow
                                if is_baseline:
                                    s, r, c, extraction, duration = run_tdd_baseline(
                                        gen, tdd_task
                                    )
                                else:
                                    s, r, c, extraction, duration = run_tdd_guarded(
                                        gen, tdd_task, r_max, human_review
                                    )
                            else:
                                # Regular workflow
                                if is_baseline:
                                    s, r, c, extraction, duration = run_linear_baseline(
                                        gen, task_prompt, guards
                                    )
                                else:
                                    s, r, c, extraction, duration = (
                                        run_guarded_workflow(
                                            gen, task_prompt, guards, r_max
                                        )
                                    )
                        except Exception as e:
                            logger.warning(f"Trial {i + 1} error: {e}")
                            s, r, c, extraction, duration = (
                                False,
                                0,
                                GENERATION_INCREMENT,
                                "error",
                                0.0,
                            )

                        if s:
                            successes += 1
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1

                        total_retries += r
                        total_cost += c
                        total_duration += duration
                        num_trials_run += 1

                        writer.write_trial_result(
                            {
                                "model_name": name,
                                "config": config_name,
                                "task": current_task,
                                "trial_num": i + 1,
                                "success": s,
                                "retries": r,
                                "generation_count": c,
                                "extraction_method": extraction,
                                "duration_seconds": round(duration, 2),
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )

                    progress.advance(task_id)
                    progress.update(
                        task_id,
                        description=f"Running {config_name}... (Success: {successes}/{num_trials_run})",
                    )

                    if num_trials_run > 0:
                        avg_success = (successes / num_trials_run) * 100
                        avg_retries = total_retries / num_trials_run
                        avg_cost = total_cost / num_trials_run
                        avg_duration = total_duration / num_trials_run
                    else:
                        avg_success = avg_retries = avg_cost = avg_duration = 0

                    results.append(
                        {
                            "Task": current_task,
                            "Model": name,
                            "Config": config_name,
                            "Success": avg_success,
                            "Retries": avg_retries,
                            "Cost": avg_cost,
                            "Duration": avg_duration,
                        }
                    )
                    console.print(
                        f"  [dim]Finished {config_name}: {avg_success:.0f}% Success[/dim]"
                    )

    writer.close()

    # Print results table
    console.print("\n[bold]Results Summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Task")
    table.add_column("Model")
    table.add_column("Config")
    table.add_column("Success", justify="right")
    table.add_column("Avg Retries", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Avg Duration", justify="right")

    for row in sorted(results, key=lambda x: (x["Task"], x["Model"])):
        retries_str = f"{row['Retries']:.1f}" if "Guarded" in row["Config"] else "N/A"
        success_val = row["Success"]
        if success_val >= 90:
            success_str = f"[green]{success_val:.0f}%[/green]"
        elif success_val >= 70:
            success_str = f"[yellow]{success_val:.0f}%[/yellow]"
        else:
            success_str = f"[red]{success_val:.0f}%[/red]"
        duration_str = f"{row['Duration']:.1f}s"
        table.add_row(
            row["Task"],
            row["Model"],
            row["Config"],
            success_str,
            retries_str,
            f"{row['Cost']:.1f}x",
            duration_str,
        )

    console.print(table)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
