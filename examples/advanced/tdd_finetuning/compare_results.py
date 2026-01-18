#!/usr/bin/env python3
"""Compare baseline and fine-tuned model results."""

import argparse
from pathlib import Path

from src.analysis.evaluate import ResultsEvaluator
from src.config import RESULTS_DIR


def main():
    """Compare benchmark results."""
    parser = argparse.ArgumentParser(
        description="Compare baseline and fine-tuned model results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare with default paths
  python compare_results.py

  # Compare with custom paths
  python compare_results.py \\
    --baseline results/baseline/qwen2.5-coder_7b/summary.json \\
    --finetuned results/finetuned/qwen2.5-coder_7b-tdd-finetuned/summary.json

  # Save detailed report
  python compare_results.py --output results/analysis/comparison_report.json
        """
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=RESULTS_DIR / "baseline" / "qwen2.5-coder_7b" / "summary.json",
        help="Path to baseline summary.json"
    )
    parser.add_argument(
        "--finetuned",
        type=Path,
        default=RESULTS_DIR / "finetuned" / "qwen2.5-coder_7b-tdd-finetuned" / "summary.json",
        help="Path to fine-tuned summary.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save evaluation report JSON (optional)"
    )
    parser.add_argument(
        "--reference-rate",
        type=float,
        default=70.0,
        help="Reference model success rate for hypothesis testing (default: 70.0)"
    )

    args = parser.parse_args()

    # Verify input files exist
    if not args.baseline.exists():
        print(f"Error: Baseline summary not found: {args.baseline}")
        print("\nRun baseline benchmark first:")
        print("  python run_benchmark.py --model baseline --trials 50")
        return 1

    if not args.finetuned.exists():
        print(f"Error: Fine-tuned summary not found: {args.finetuned}")
        print("\nRun fine-tuned benchmark first:")
        print("  python run_benchmark.py --model finetuned --trials 50")
        return 1

    print("Comparing benchmark results...")
    print(f"Baseline:   {args.baseline}")
    print(f"Fine-tuned: {args.finetuned}")

    # Create evaluator and generate report
    evaluator = ResultsEvaluator()
    report = evaluator.generate_report(
        args.baseline,
        args.finetuned,
        args.output,
    )

    return 0


if __name__ == "__main__":
    exit(main())
