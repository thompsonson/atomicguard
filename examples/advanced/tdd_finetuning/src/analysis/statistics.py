"""Statistical analysis tools for comparing benchmark results."""

from typing import Any
from scipy import stats
import numpy as np


class StatisticalAnalyzer:
    """Performs statistical analysis on benchmark results."""

    def compare_success_rates(
        self,
        baseline_successes: int,
        baseline_trials: int,
        finetuned_successes: int,
        finetuned_trials: int,
    ) -> dict[str, Any]:
        """Compare success rates using Fisher's exact test.

        Args:
            baseline_successes: Number of successful trials for baseline
            baseline_trials: Total trials for baseline
            finetuned_successes: Number of successful trials for fine-tuned
            finetuned_trials: Total trials for fine-tuned

        Returns:
            Dictionary with statistical test results
        """
        # Create contingency table
        # [[baseline_success, baseline_fail],
        #  [finetuned_success, finetuned_fail]]
        table = [
            [baseline_successes, baseline_trials - baseline_successes],
            [finetuned_successes, finetuned_trials - finetuned_successes],
        ]

        # Fisher's exact test (two-tailed)
        odds_ratio, p_value = stats.fisher_exact(table, alternative="two-sided")

        # Calculate success rates
        baseline_rate = baseline_successes / baseline_trials
        finetuned_rate = finetuned_successes / finetuned_trials

        # Calculate absolute difference
        abs_diff = finetuned_rate - baseline_rate

        # Calculate Cohen's h (effect size for proportions)
        cohens_h = self._calculate_cohens_h(baseline_rate, finetuned_rate)

        # Interpret effect size
        effect_size_interpretation = self._interpret_effect_size(cohens_h)

        return {
            "baseline_success_rate": baseline_rate * 100,
            "finetuned_success_rate": finetuned_rate * 100,
            "absolute_difference_pp": abs_diff * 100,  # percentage points
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
            "cohens_h": cohens_h,
            "effect_size": effect_size_interpretation,
            "baseline_n": baseline_trials,
            "finetuned_n": finetuned_trials,
        }

    def _calculate_cohens_h(self, p1: float, p2: float) -> float:
        """Calculate Cohen's h effect size for two proportions.

        Args:
            p1: First proportion
            p2: Second proportion

        Returns:
            Cohen's h value
        """
        # Arcsine transformation
        phi1 = 2 * np.arcsin(np.sqrt(p1))
        phi2 = 2 * np.arcsin(np.sqrt(p2))
        return phi2 - phi1

    def _interpret_effect_size(self, cohens_h: float) -> str:
        """Interpret Cohen's h effect size.

        Args:
            cohens_h: Cohen's h value

        Returns:
            Human-readable interpretation
        """
        abs_h = abs(cohens_h)
        if abs_h < 0.2:
            return "negligible"
        elif abs_h < 0.5:
            return "small"
        elif abs_h < 0.8:
            return "medium"
        else:
            return "large"

    def compare_multiple_tasks(
        self,
        baseline_results: dict[str, dict[str, int]],
        finetuned_results: dict[str, dict[str, int]],
    ) -> dict[str, Any]:
        """Compare results across multiple tasks.

        Args:
            baseline_results: Dict mapping task_id to {successes, trials}
            finetuned_results: Dict mapping task_id to {successes, trials}

        Returns:
            Dictionary with per-task and aggregate statistics
        """
        task_comparisons = {}
        overall_baseline_successes = 0
        overall_baseline_trials = 0
        overall_finetuned_successes = 0
        overall_finetuned_trials = 0

        # Compare each task
        for task_id in baseline_results.keys():
            if task_id not in finetuned_results:
                continue

            baseline = baseline_results[task_id]
            finetuned = finetuned_results[task_id]

            comparison = self.compare_success_rates(
                baseline["successes"],
                baseline["trials"],
                finetuned["successes"],
                finetuned["trials"],
            )

            task_comparisons[task_id] = comparison

            # Aggregate
            overall_baseline_successes += baseline["successes"]
            overall_baseline_trials += baseline["trials"]
            overall_finetuned_successes += finetuned["successes"]
            overall_finetuned_trials += finetuned["trials"]

        # Overall comparison
        overall_comparison = self.compare_success_rates(
            overall_baseline_successes,
            overall_baseline_trials,
            overall_finetuned_successes,
            overall_finetuned_trials,
        )

        # Count tasks with improvement
        tasks_improved = sum(
            1 for comp in task_comparisons.values()
            if comp["absolute_difference_pp"] > 0
        )

        tasks_significant = sum(
            1 for comp in task_comparisons.values()
            if comp["significant_at_0.05"] and comp["absolute_difference_pp"] > 0
        )

        return {
            "overall": overall_comparison,
            "per_task": task_comparisons,
            "summary": {
                "num_tasks": len(task_comparisons),
                "tasks_improved": tasks_improved,
                "tasks_with_significant_improvement": tasks_significant,
            },
        }

    def calculate_confidence_interval(
        self, successes: int, trials: int, confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for success rate.

        Args:
            successes: Number of successes
            trials: Total trials
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound) as percentages
        """
        if trials == 0:
            return (0.0, 0.0)

        # Wilson score interval
        p = successes / trials
        z = stats.norm.ppf((1 + confidence) / 2)

        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) / trials + z**2 / (4 * trials**2))) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return (lower * 100, upper * 100)

    def print_comparison_report(
        self, comparison: dict[str, Any], verbose: bool = True
    ) -> None:
        """Print a formatted comparison report.

        Args:
            comparison: Comparison results from compare_multiple_tasks
            verbose: Print detailed per-task results
        """
        print("\n" + "=" * 70)
        print("STATISTICAL COMPARISON REPORT")
        print("=" * 70)

        # Overall results
        overall = comparison["overall"]
        print("\nOVERALL RESULTS:")
        print(f"  Baseline:    {overall['baseline_success_rate']:.1f}% "
              f"({int(overall['baseline_n'] * overall['baseline_success_rate']/100)}/{overall['baseline_n']} trials)")
        print(f"  Fine-tuned:  {overall['finetuned_success_rate']:.1f}% "
              f"({int(overall['finetuned_n'] * overall['finetuned_success_rate']/100)}/{overall['finetuned_n']} trials)")
        print(f"  Improvement: {overall['absolute_difference_pp']:+.1f} percentage points")
        print(f"  Effect size: {overall['cohens_h']:.3f} ({overall['effect_size']})")
        print(f"  P-value:     {overall['p_value']:.4f} "
              f"{'***' if overall['p_value'] < 0.001 else '**' if overall['p_value'] < 0.01 else '*' if overall['p_value'] < 0.05 else 'ns'}")

        if overall["significant_at_0.05"]:
            print(f"  → Statistically significant at α=0.05")
        else:
            print(f"  → Not statistically significant at α=0.05")

        # Summary
        summary = comparison["summary"]
        print(f"\nSUMMARY:")
        print(f"  Tasks analyzed: {summary['num_tasks']}")
        print(f"  Tasks improved: {summary['tasks_improved']}/{summary['num_tasks']}")
        print(f"  Tasks with significant improvement: {summary['tasks_with_significant_improvement']}/{summary['num_tasks']}")

        # Per-task details
        if verbose and "per_task" in comparison:
            print("\nPER-TASK RESULTS:")
            print(f"{'Task':<25} {'Baseline':<12} {'Fine-tuned':<12} {'Δ':<8} {'p-value':<10} {'Sig':<5}")
            print("-" * 70)

            for task_id, task_comp in sorted(comparison["per_task"].items()):
                baseline_pct = task_comp["baseline_success_rate"]
                finetuned_pct = task_comp["finetuned_success_rate"]
                diff = task_comp["absolute_difference_pp"]
                p_val = task_comp["p_value"]
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

                print(f"{task_id:<25} {baseline_pct:>6.1f}%     {finetuned_pct:>6.1f}%     "
                      f"{diff:>+5.1f}pp  {p_val:>8.4f}  {sig:<5}")

        print("=" * 70)
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("=" * 70)
