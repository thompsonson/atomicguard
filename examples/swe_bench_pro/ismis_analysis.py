"""ISMIS 2026 experiment analysis.

Produces per-model resolution rate tables, cross-model comparisons,
dose-response curves, and statistical tests for the 13-model evaluation.

Usage::

    from examples.swe_bench_pro.ismis_analysis import ISMISAnalyzer

    analyzer = ISMISAnalyzer("output/ismis_2026")
    analyzer.generate_all()
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from examples.swe_bench_ablation.analysis import (
    cohens_h,
    effect_size_label,
    fishers_exact_test,
    load_results,
    wilson_ci,
)
from examples.swe_bench_common import ArmResult

logger = logging.getLogger("swe_bench_pro.ismis_analysis")

# Arm ordering for dose-response analysis (increasing decomposition)
ARM_ORDER = ["02_singleshot", "04_s1_tdd", "08_s1_decomposed_lite"]
ARM_LABELS = {
    "02_singleshot": "Baseline",
    "04_s1_tdd": "S1-TDD",
    "08_s1_decomposed_lite": "S1-Decomposed",
}
ARM_STEPS = {
    "02_singleshot": 1,
    "04_s1_tdd": 3,
    "08_s1_decomposed_lite": 5,
}
ARM_COLORS = {
    "02_singleshot": "#ff6b6b",
    "04_s1_tdd": "#4ecdc4",
    "08_s1_decomposed_lite": "#556fb5",
}


class ISMISAnalyzer:
    """Analyzes multi-model experiment results for the ISMIS paper."""

    def __init__(self, experiment_dir: str | Path):
        self._dir = Path(experiment_dir)
        self._models: list[dict[str, Any]] = []
        self._results_by_model: dict[str, list[ArmResult]] = {}

    def load_all(self) -> None:
        """Load results from all model subdirectories."""
        if not self._dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self._dir}")

        for model_dir in sorted(self._dir.iterdir()):
            if not model_dir.is_dir():
                continue

            config_path = model_dir / "model_config.json"
            results_path = model_dir / "results.jsonl"

            if not results_path.exists():
                logger.warning("No results.jsonl in %s, skipping", model_dir)
                continue

            config = {}
            if config_path.exists():
                config = json.loads(config_path.read_text())

            label = config.get("label", model_dir.name)
            results = load_results(str(results_path))

            if results:
                self._models.append(config)
                self._results_by_model[label] = results
                logger.info("Loaded %d results for %s", len(results), label)

        logger.info("Loaded %d models with results", len(self._results_by_model))

    def compute_resolution_table(self) -> list[dict[str, Any]]:
        """Compute per-model, per-arm resolution rate table.

        Returns:
            List of dicts, one per (model, arm) pair, with fields:
            model, arm, n, resolved, rate, ci_lower, ci_upper,
            mean_tokens, delta_vs_baseline, cohens_h
        """
        rows: list[dict[str, Any]] = []

        for model_label, results in self._results_by_model.items():
            by_arm: dict[str, list[ArmResult]] = defaultdict(list)
            for r in results:
                by_arm[r.arm].append(r)

            # Compute baseline rate for delta calculation
            baseline_rate = 0.0
            baseline_arm = ARM_ORDER[0] if ARM_ORDER[0] in by_arm else None
            if baseline_arm:
                bl = by_arm[baseline_arm]
                baseline_resolved = sum(1 for r in bl if r.resolved is True)
                baseline_rate = baseline_resolved / len(bl) if bl else 0.0

            for arm in ARM_ORDER:
                if arm not in by_arm:
                    continue

                arm_results = by_arm[arm]
                n = len(arm_results)
                resolved = sum(1 for r in arm_results if r.resolved is True)
                rate, ci_lo, ci_hi = wilson_ci(resolved, n)

                total_tokens = sum(r.total_tokens for r in arm_results)
                mean_tokens = total_tokens / n if n else 0

                arm_rate = resolved / n if n else 0.0
                delta = (arm_rate - baseline_rate) * 100
                h = cohens_h(arm_rate, baseline_rate) if baseline_rate > 0 else 0.0

                rows.append(
                    {
                        "model": model_label,
                        "arm": arm,
                        "arm_label": ARM_LABELS.get(arm, arm),
                        "n": n,
                        "resolved": resolved,
                        "rate": round(rate, 1),
                        "ci_lower": round(ci_lo, 1),
                        "ci_upper": round(ci_hi, 1),
                        "mean_tokens": round(mean_tokens),
                        "delta_vs_baseline": round(delta, 1),
                        "cohens_h": round(h, 3),
                        "effect_size": effect_size_label(h),
                    }
                )

        return rows

    def compute_pairwise_per_model(self) -> list[dict[str, Any]]:
        """Compute Fisher's exact test for pairwise arm comparisons per model.

        Returns:
            List of comparison dicts with model, arm_a, arm_b, p_value, etc.
        """
        comparisons: list[dict[str, Any]] = []

        for model_label, results in self._results_by_model.items():
            by_arm: dict[str, list[ArmResult]] = defaultdict(list)
            for r in results:
                by_arm[r.arm].append(r)

            arms_present = [a for a in ARM_ORDER if a in by_arm]

            for i, arm_a in enumerate(arms_present):
                for arm_b in arms_present[i + 1 :]:
                    ra = by_arm[arm_a]
                    rb = by_arm[arm_b]
                    na = len(ra)
                    nb = len(rb)

                    sa = sum(1 for r in ra if r.resolved is True)
                    sb = sum(1 for r in rb if r.resolved is True)
                    fa = na - sa
                    fb = nb - sb

                    p_val = fishers_exact_test(sa, fa, sb, fb)
                    pa = sa / na if na else 0.0
                    pb = sb / nb if nb else 0.0
                    h = cohens_h(pa, pb)
                    delta = (pb - pa) * 100

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
                            "model": model_label,
                            "arm_a": ARM_LABELS.get(arm_a, arm_a),
                            "arm_b": ARM_LABELS.get(arm_b, arm_b),
                            "rate_a": round(pa * 100, 1),
                            "rate_b": round(pb * 100, 1),
                            "delta": round(delta, 1),
                            "p_value": round(p_val, 4),
                            "significance": sig,
                            "cohens_h": round(h, 3),
                            "effect_size": effect_size_label(h),
                        }
                    )

        return comparisons

    def compute_aggregate_stats(self) -> dict[str, Any]:
        """Compute experiment-wide aggregate statistics.

        Returns:
            Dict with aggregate metrics: mean improvement, range,
            models where decomposition helps, max improvement, etc.
        """
        table = self.compute_resolution_table()

        # Group by arm across models
        by_arm: dict[str, list[dict]] = defaultdict(list)
        for row in table:
            by_arm[row["arm"]].append(row)

        # Delta analysis (vs baseline)
        deltas: list[float] = []
        models_helped: list[str] = []
        max_delta = 0.0
        max_delta_model = ""

        for row in table:
            if row["arm"] != ARM_ORDER[0]:  # skip baseline
                d = row["delta_vs_baseline"]
                deltas.append(d)
                if d > 0:
                    models_helped.append(f"{row['model']}:{row['arm_label']}")
                if d > max_delta:
                    max_delta = d
                    max_delta_model = f"{row['model']}:{row['arm_label']}"

        return {
            "total_models": len(self._results_by_model),
            "arms": list(ARM_LABELS.values()),
            "per_arm_mean_rate": {
                ARM_LABELS.get(arm, arm): round(
                    sum(r["rate"] for r in rows) / len(rows), 1
                )
                if rows
                else 0.0
                for arm, rows in by_arm.items()
            },
            "mean_improvement_pp": round(sum(deltas) / len(deltas), 1)
            if deltas
            else 0.0,
            "max_improvement_pp": round(max_delta, 1),
            "max_improvement_model": max_delta_model,
            "models_where_decomposition_helps": len([d for d in deltas if d > 0]),
            "total_comparisons": len(deltas),
            "improvement_range_pp": (
                round(min(deltas), 1) if deltas else 0.0,
                round(max(deltas), 1) if deltas else 0.0,
            ),
        }

    # =====================================================================
    # Report generation
    # =====================================================================

    def generate_markdown_report(
        self,
        output_path: str | Path | None = None,
    ) -> str:
        """Generate comprehensive Markdown report."""
        if output_path is None:
            output_path = self._dir / "ismis_report.md"

        table = self.compute_resolution_table()
        pairwise = self.compute_pairwise_per_model()
        aggregate = self.compute_aggregate_stats()

        lines = [
            "# ISMIS 2026: Multi-Model Decomposition Evaluation",
            "",
            "## Aggregate Statistics",
            "",
            f"- **Models evaluated**: {aggregate['total_models']}",
            f"- **Arms**: {', '.join(aggregate['arms'])}",
            f"- **Mean improvement**: {aggregate['mean_improvement_pp']:+.1f}pp",
            f"- **Max improvement**: {aggregate['max_improvement_pp']:+.1f}pp ({aggregate['max_improvement_model']})",
            f"- **Models where decomposition helps**: {aggregate['models_where_decomposition_helps']}/{aggregate['total_comparisons']}",
            "",
            "## Per-Model Resolution Rate",
            "",
            "| Model | Arm | n | Resolved | Rate | 95% CI | Tokens | Delta | h | Effect |",
            "|-------|-----|---|----------|------|--------|--------|-------|---|--------|",
        ]

        for row in table:
            lines.append(
                f"| {row['model']} | {row['arm_label']} | {row['n']} | "
                f"{row['resolved']} | {row['rate']:.1f}% | "
                f"[{row['ci_lower']:.1f}, {row['ci_upper']:.1f}]% | "
                f"{row['mean_tokens']} | {row['delta_vs_baseline']:+.1f}pp | "
                f"{row['cohens_h']:.3f} | {row['effect_size']} |"
            )

        lines.extend(
            [
                "",
                "## Pairwise Comparisons (Fisher's Exact Test)",
                "",
                "| Model | Arm A | Arm B | Rate A | Rate B | Delta | p-value | Sig | h |",
                "|-------|-------|-------|--------|--------|-------|---------|-----|---|",
            ]
        )

        for c in pairwise:
            lines.append(
                f"| {c['model']} | {c['arm_a']} | {c['arm_b']} | "
                f"{c['rate_a']:.1f}% | {c['rate_b']:.1f}% | "
                f"{c['delta']:+.1f}pp | {c['p_value']:.4f} | "
                f"{c['significance']} | {c['cohens_h']:.3f} |"
            )

        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- **Rate**: Resolution rate (patches passing FAIL_TO_PASS tests)",
                "- **CI**: Wilson score 95% confidence interval",
                "- **Delta**: Improvement over baseline (Singleshot)",
                "- **h**: Cohen's h effect size",
                "- **Sig**: *** p<0.001, ** p<0.01, * p<0.05, ns not significant",
            ]
        )

        report = "\n".join(lines) + "\n"
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
        logger.info("Report written to %s", output)
        return report

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for the ISMIS paper."""
        table = self.compute_resolution_table()

        # Group by model for multi-row display
        by_model: dict[str, list[dict]] = defaultdict(list)
        for row in table:
            by_model[row["model"]].append(row)

        lines = [
            r"\begin{table*}[htbp]",
            r"\centering",
            r"\caption{Resolution rate across 13 models and three decomposition levels on SWE-Bench Pro (Python, $n=30$).}",
            r"\label{tab:ismis-results}",
            r"\begin{tabular}{llccccccc}",
            r"\toprule",
            r"Model & Size & \multicolumn{2}{c}{Baseline} & \multicolumn{2}{c}{S1-TDD} & \multicolumn{2}{c}{S1-Decomposed} & $\Delta_{\max}$ \\",
            r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
            r" & & $\hat{\epsilon}$ & CI & $\hat{\epsilon}$ & CI & $\hat{\epsilon}$ & CI & (pp) \\",
            r"\midrule",
        ]

        for model, rows in by_model.items():
            arm_data = {r["arm"]: r for r in rows}
            model_label = model.replace("_", r"\_")

            # Get size tier from config
            config = next(
                (c for c in self._models if c.get("label") == model),
                {},
            )
            size = config.get("size_tier", "")

            cols = []
            rates = []
            for arm in ARM_ORDER:
                d = arm_data.get(arm)
                if d:
                    cols.append(f"{d['rate']:.1f}\\%")
                    cols.append(f"[{d['ci_lower']:.0f},{d['ci_upper']:.0f}]")
                    rates.append(d["rate"])
                else:
                    cols.append("--")
                    cols.append("--")
                    rates.append(0.0)

            max_delta = max(rates) - rates[0] if rates else 0.0

            lines.append(
                f"  {model_label} & {size} & "
                + " & ".join(cols)
                + f" & {max_delta:+.1f} \\\\"
            )

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
            ]
        )

        return "\n".join(lines) + "\n"

    # =====================================================================
    # Visualizations
    # =====================================================================

    def plot_cross_model_comparison(self, output_dir: str | Path | None = None) -> str:
        """Grouped bar chart: resolution rate by arm for each model."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        if output_dir is None:
            output_dir = self._dir / "plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        table = self.compute_resolution_table()
        models = list(dict.fromkeys(r["model"] for r in table))
        arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in table)]

        if not models or not arms:
            return ""

        x = np.arange(len(models))
        width = 0.25
        n_arms = len(arms)

        fig, ax = plt.subplots(figsize=(max(12, len(models) * 1.5), 7))

        for i, arm in enumerate(arms):
            rates = []
            for model in models:
                row = next(
                    (r for r in table if r["model"] == model and r["arm"] == arm),
                    None,
                )
                rates.append(row["rate"] if row else 0.0)

            offset = (i - (n_arms - 1) / 2) * width
            ax.bar(
                x + offset,
                rates,
                width * 0.9,
                label=ARM_LABELS.get(arm, arm),
                color=ARM_COLORS.get(arm, "#888888"),
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Resolution Rate (%)")
        ax.set_title("Resolution Rate by Model and Arm")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = output_dir / "cross_model_comparison.png"
        plt.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.info("Saved %s", path)
        return str(path)

    def plot_dose_response(self, output_dir: str | Path | None = None) -> str:
        """Dose-response curve: resolution rate vs decomposition steps."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if output_dir is None:
            output_dir = self._dir / "plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        table = self.compute_resolution_table()
        if not table:
            return ""

        fig, ax = plt.subplots(figsize=(10, 7))

        # Group models by size tier for distinct line styles
        size_markers = {
            "7B": "o",
            "8B": "o",
            "14B": "s",
            "22B": "s",
            "32B": "D",
            "33B": "D",
            "70B": "^",
            "72B": "^",
            "123B": "v",
            "671B_MoE": "*",
            "proprietary": "P",
        }

        by_model: dict[str, list[dict]] = defaultdict(list)
        for row in table:
            by_model[row["model"]].append(row)

        for model, rows in by_model.items():
            config = next(
                (c for c in self._models if c.get("label") == model),
                {},
            )
            size = config.get("size_tier", "")
            marker = size_markers.get(size, "o")

            steps = [ARM_STEPS.get(r["arm"], 0) for r in rows]
            rates = [r["rate"] for r in rows]

            ax.plot(
                steps,
                rates,
                marker=marker,
                markersize=8,
                linewidth=1.5,
                label=f"{model} ({size})",
                alpha=0.8,
            )

        ax.set_xlabel("Decomposition Steps")
        ax.set_ylabel("Resolution Rate (%)")
        ax.set_title("Dose-Response: Resolution Rate vs Decomposition Depth")
        ax.set_xticks([ARM_STEPS[a] for a in ARM_ORDER])
        ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER])
        ax.set_ylim(0, 105)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            ncol=1,
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / "dose_response.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)
        return str(path)

    def generate_all(self, output_dir: str | Path | None = None) -> None:
        """Load data and generate all analysis outputs."""
        self.load_all()

        if not self._results_by_model:
            logger.warning("No results to analyze")
            return

        if output_dir is None:
            output_dir = self._dir

        output_dir = Path(output_dir)
        plots_dir = output_dir / "plots"

        # Tables and reports
        self.generate_markdown_report(output_dir / "ismis_report.md")

        latex = self.generate_latex_table()
        (output_dir / "ismis_table.tex").write_text(latex)
        logger.info("LaTeX table written to %s", output_dir / "ismis_table.tex")

        # Statistical data as JSON
        table = self.compute_resolution_table()
        pairwise = self.compute_pairwise_per_model()
        aggregate = self.compute_aggregate_stats()

        stats = {
            "resolution_table": table,
            "pairwise_comparisons": pairwise,
            "aggregate_statistics": aggregate,
        }
        stats_path = output_dir / "ismis_statistics.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        logger.info("Statistics written to %s", stats_path)

        # Visualizations
        self.plot_cross_model_comparison(plots_dir)
        self.plot_dose_response(plots_dir)

        logger.info("All ISMIS analysis outputs generated in %s", output_dir)
