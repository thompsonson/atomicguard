"""Evaluation tools for comparing baseline and fine-tuned models."""

import json
from pathlib import Path
from typing import Any

from ..benchmark.trial import TrialResult, BenchmarkSummary
from .statistics import StatisticalAnalyzer


class ResultsEvaluator:
    """Evaluates and compares benchmark results."""

    def __init__(self):
        """Initialize results evaluator."""
        self.analyzer = StatisticalAnalyzer()

    def load_summary(self, summary_file: Path) -> BenchmarkSummary:
        """Load benchmark summary from JSON file.

        Args:
            summary_file: Path to summary.json file

        Returns:
            BenchmarkSummary object
        """
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct BenchmarkSummary
        return BenchmarkSummary(**data)

    def load_trial_results(self, trials_dir: Path) -> list[TrialResult]:
        """Load all trial results from directory.

        Args:
            trials_dir: Directory containing *_trials.jsonl files

        Returns:
            List of TrialResult objects
        """
        all_results = []

        for trials_file in trials_dir.glob("*_trials.jsonl"):
            with open(trials_file, "r", encoding="utf-8") as f:
                for line in f:
                    trial_data = json.loads(line)
                    trial_result = TrialResult.from_dict(trial_data)
                    all_results.append(trial_result)

        return all_results

    def extract_task_statistics(
        self, summary: BenchmarkSummary
    ) -> dict[str, dict[str, int]]:
        """Extract per-task statistics from benchmark summary.

        Args:
            summary: BenchmarkSummary object

        Returns:
            Dict mapping task_id to {successes, trials}
        """
        task_stats = {}

        for task_id, task_result in summary.task_results.items():
            task_stats[task_id] = {
                "successes": task_result["workflow_successes"],
                "trials": task_result["num_trials"],
            }

        return task_stats

    def compare_models(
        self,
        baseline_summary: BenchmarkSummary,
        finetuned_summary: BenchmarkSummary,
    ) -> dict[str, Any]:
        """Compare baseline and fine-tuned model results.

        Args:
            baseline_summary: Baseline model benchmark summary
            finetuned_summary: Fine-tuned model benchmark summary

        Returns:
            Comparison results with statistical analysis
        """
        # Extract task statistics
        baseline_stats = self.extract_task_statistics(baseline_summary)
        finetuned_stats = self.extract_task_statistics(finetuned_summary)

        # Perform statistical comparison
        comparison = self.analyzer.compare_multiple_tasks(
            baseline_stats, finetuned_stats
        )

        # Add model names
        comparison["baseline_model"] = baseline_summary.model_name
        comparison["finetuned_model"] = finetuned_summary.model_name

        # Add execution time comparison
        comparison["execution_time"] = {
            "baseline_total": baseline_summary.total_execution_time,
            "finetuned_total": finetuned_summary.total_execution_time,
            "baseline_avg": baseline_summary.avg_time_per_trial,
            "finetuned_avg": finetuned_summary.avg_time_per_trial,
        }

        # Add attempt statistics comparison
        comparison["attempts"] = {
            "baseline_avg": baseline_summary.avg_attempts_per_trial,
            "finetuned_avg": finetuned_summary.avg_attempts_per_trial,
            "baseline_g_test_avg": baseline_summary.avg_g_test_attempts,
            "finetuned_g_test_avg": finetuned_summary.avg_g_test_attempts,
        }

        return comparison

    def evaluate_hypothesis(
        self, comparison: dict[str, Any], reference_model_rate: float = 70.0
    ) -> dict[str, bool]:
        """Evaluate experimental hypotheses.

        Args:
            comparison: Comparison results from compare_models
            reference_model_rate: Reference model success rate (e.g., 14B model at 70%)

        Returns:
            Dictionary with hypothesis evaluation results
        """
        overall = comparison["overall"]
        finetuned_rate = overall["finetuned_success_rate"]
        baseline_rate = overall["baseline_success_rate"]
        improvement = overall["absolute_difference_pp"]

        # H1: Fine-tuned 7B improves by ≥5pp over baseline 7B
        h1_met = improvement >= 5.0 and overall["significant_at_0.05"]

        # H2: Fine-tuned 7B matches 14B baseline on ≥3 tasks
        tasks_matching_reference = 0
        if "per_task" in comparison:
            for task_comp in comparison["per_task"].values():
                if task_comp["finetuned_success_rate"] >= reference_model_rate:
                    tasks_matching_reference += 1

        h2_met = tasks_matching_reference >= 3

        return {
            "H1_improvement_5pp": h1_met,
            "H1_details": {
                "improvement_pp": improvement,
                "significant": overall["significant_at_0.05"],
                "threshold": 5.0,
            },
            "H2_match_14B_on_3_tasks": h2_met,
            "H2_details": {
                "tasks_matching_reference": tasks_matching_reference,
                "threshold_tasks": 3,
                "reference_rate": reference_model_rate,
            },
            "overall_success": h1_met or h2_met,
        }

    def generate_report(
        self,
        baseline_summary_file: Path,
        finetuned_summary_file: Path,
        output_file: Path | None = None,
    ) -> None:
        """Generate comprehensive evaluation report.

        Args:
            baseline_summary_file: Path to baseline summary.json
            finetuned_summary_file: Path to fine-tuned summary.json
            output_file: Optional path to save report JSON
        """
        # Load summaries
        baseline_summary = self.load_summary(baseline_summary_file)
        finetuned_summary = self.load_summary(finetuned_summary_file)

        # Compare models
        comparison = self.compare_models(baseline_summary, finetuned_summary)

        # Evaluate hypotheses
        hypothesis_results = self.evaluate_hypothesis(comparison)

        # Create full report
        report = {
            "comparison": comparison,
            "hypothesis_evaluation": hypothesis_results,
            "summaries": {
                "baseline": baseline_summary.to_dict(),
                "finetuned": finetuned_summary.to_dict(),
            },
        }

        # Save report if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {output_file}")

        # Print statistical comparison
        self.analyzer.print_comparison_report(comparison, verbose=True)

        # Print hypothesis evaluation
        self.print_hypothesis_results(hypothesis_results)

        return report

    def print_hypothesis_results(self, results: dict[str, Any]) -> None:
        """Print hypothesis evaluation results.

        Args:
            results: Hypothesis evaluation results
        """
        print("\n" + "=" * 70)
        print("HYPOTHESIS EVALUATION")
        print("=" * 70)

        h1 = results["H1_improvement_5pp"]
        h1_details = results["H1_details"]
        print(f"\nH1: Fine-tuned 7B improves by ≥5pp over baseline 7B")
        print(f"  Result: {'✓ MET' if h1 else '✗ NOT MET'}")
        print(f"  Improvement: {h1_details['improvement_pp']:.1f}pp (threshold: {h1_details['threshold']}pp)")
        print(f"  Significant: {h1_details['significant']}")

        h2 = results["H2_match_14B_on_3_tasks"]
        h2_details = results["H2_details"]
        print(f"\nH2: Fine-tuned 7B matches 14B baseline (70%) on ≥3 tasks")
        print(f"  Result: {'✓ MET' if h2 else '✗ NOT MET'}")
        print(f"  Tasks matching: {h2_details['tasks_matching_reference']}/{h2_details['threshold_tasks']}")
        print(f"  Reference rate: {h2_details['reference_rate']:.1f}%")

        print(f"\nOVERALL: {'✓ EXPERIMENT SUCCESS' if results['overall_success'] else '✗ EXPERIMENT FAILED'}")
        print("=" * 70)


def main():
    """Example usage of results evaluator."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline summary.json"
    )
    parser.add_argument(
        "--finetuned",
        type=Path,
        required=True,
        help="Path to fine-tuned summary.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save evaluation report"
    )

    args = parser.parse_args()

    evaluator = ResultsEvaluator()
    evaluator.generate_report(
        args.baseline,
        args.finetuned,
        args.output,
    )


if __name__ == "__main__":
    main()
