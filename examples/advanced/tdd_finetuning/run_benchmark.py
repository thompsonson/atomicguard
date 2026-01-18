#!/usr/bin/env python3
"""Run TDD workflow benchmark."""

import argparse
import json
from pathlib import Path

from src.benchmark.runner import TDDWorkflowRunner
from src.config import BASELINE_MODEL, FINETUNED_MODEL, EXPERIMENT, TASKS_DIR


def load_tasks(tasks_dir: Path) -> list[dict]:
    """Load task specifications from directory.

    Args:
        tasks_dir: Directory containing task JSON files

    Returns:
        List of task specification dictionaries
    """
    tasks = []
    for task_file in sorted(tasks_dir.glob("*.json")):
        with open(task_file, "r", encoding="utf-8") as f:
            task = json.load(f)
            tasks.append(task)

    return tasks


def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser(
        description="Run TDD workflow benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline benchmark
  python run_benchmark.py --model baseline --trials 50

  # Run fine-tuned model benchmark
  python run_benchmark.py --model finetuned --trials 50

  # Run with custom model
  python run_benchmark.py --model custom --model-name qwen2.5-coder:14b --trials 50
        """
    )

    parser.add_argument(
        "--model",
        choices=["baseline", "finetuned", "custom"],
        default="baseline",
        help="Model to benchmark (default: baseline)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Custom model name (for --model custom)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials per task (default: 50)"
    )
    parser.add_argument(
        "--retry-budget",
        type=int,
        default=3,
        help="Maximum retry attempts per phase (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (default: results/baseline or results/finetuned)"
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=TASKS_DIR,
        help="Directory containing task specifications"
    )

    args = parser.parse_args()

    # Select model configuration
    if args.model == "baseline":
        model_config = BASELINE_MODEL.copy()
        output_subdir = "baseline"
    elif args.model == "finetuned":
        model_config = FINETUNED_MODEL.copy()
        output_subdir = "finetuned"
    else:  # custom
        if not args.model_name:
            parser.error("--model-name required when using --model custom")
        model_config = BASELINE_MODEL.copy()
        model_config["name"] = args.model_name
        output_subdir = args.model_name.replace(":", "_")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = EXPERIMENT["output_dir"] / output_subdir

    print(f"Starting TDD Workflow Benchmark")
    print(f"Model: {model_config['name']}")
    print(f"Trials per task: {args.trials}")
    print(f"Retry budget: {args.retry_budget}")
    print(f"Output directory: {output_dir}")

    # Load tasks
    tasks = load_tasks(args.tasks_dir)
    print(f"Loaded {len(tasks)} tasks: {[t['task_id'] for t in tasks]}")

    # Create runner
    runner = TDDWorkflowRunner(
        model_config=model_config,
        retry_budget=args.retry_budget,
        output_dir=output_dir,
    )

    # Run benchmark
    summary = runner.run_benchmark(tasks, num_trials=args.trials)

    print(f"\nBenchmark complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
