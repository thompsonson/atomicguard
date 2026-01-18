"""Benchmark runner for TDD workflow evaluation."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import EXPERIMENT
from ..generators.test_generator import TestGenerator
from ..generators.impl_generator import ImplGenerator
from ..guards.syntax_guard import SyntaxGuard
from ..guards.test_guard import TestGuard
from .trial import (
    AttemptResult,
    PhaseResult,
    TrialResult,
    BenchmarkSummary,
)


class TDDWorkflowRunner:
    """Runs TDD workflow benchmark trials."""

    def __init__(
        self,
        model_config: dict[str, Any],
        retry_budget: int = 3,
        output_dir: Path | None = None,
    ):
        """Initialize TDD workflow runner.

        Args:
            model_config: Model configuration dictionary
            retry_budget: Maximum retry attempts per phase
            output_dir: Directory to save results
        """
        self.model_config = model_config
        self.retry_budget = retry_budget
        self.output_dir = output_dir or EXPERIMENT["output_dir"]

        # Initialize generators
        self.test_gen = TestGenerator(model_config)
        self.impl_gen = ImplGenerator(model_config)

        # Initialize guards
        self.syntax_guard = SyntaxGuard()
        self.test_guard = TestGuard()

    def run_trial(self, task: dict[str, Any], trial_num: int) -> TrialResult:
        """Run a single TDD workflow trial.

        Args:
            task: Task specification dictionary
            trial_num: Trial number (1-indexed)

        Returns:
            TrialResult containing complete trial information
        """
        start_time = time.time()
        task_id = task["task_id"]

        # Phase 1: Generate tests (g_test)
        g_test_result = self._run_phase_with_retry(
            phase_name="g_test",
            generator_func=lambda: self.test_gen.generate(task),
            guard_func=self.syntax_guard.validate,
        )

        # Initialize trial result
        trial_result = TrialResult(
            task_id=task_id,
            trial_number=trial_num,
            workflow_success=False,
            g_test_result=g_test_result,
        )

        # Check if g_test succeeded
        if not g_test_result.success:
            trial_result.total_attempts = g_test_result.total_attempts
            trial_result.execution_time_seconds = time.time() - start_time
            return trial_result

        # Check if first attempt succeeded (quality flag)
        if g_test_result.total_attempts == 1 and g_test_result.success:
            trial_result.first_attempt_test_success = True

        # Phase 2: Generate implementation (g_impl)
        test_code = g_test_result.final_code
        g_impl_result = self._run_phase_with_retry(
            phase_name="g_impl",
            generator_func=lambda: self.impl_gen.generate(task, test_code),
            guard_func=lambda code: self.test_guard.validate(code, test_code),
        )

        trial_result.g_impl_result = g_impl_result

        # Calculate final success flags
        trial_result.workflow_success = g_test_result.success and g_impl_result.success
        trial_result.validated_success = trial_result.workflow_success
        trial_result.total_attempts = (
            g_test_result.total_attempts + g_impl_result.total_attempts
        )
        trial_result.execution_time_seconds = time.time() - start_time

        return trial_result

    def _run_phase_with_retry(
        self,
        phase_name: str,
        generator_func: callable,
        guard_func: callable,
    ) -> PhaseResult:
        """Run a single phase with retry mechanism.

        Args:
            phase_name: Name of the phase ("g_test" or "g_impl")
            generator_func: Function that generates code
            guard_func: Function that validates code (returns tuple of (bool, str))

        Returns:
            PhaseResult with all attempts tracked
        """
        phase_result = PhaseResult(phase_name=phase_name, success=False)
        attempts = []

        for attempt_num in range(1, self.retry_budget + 1):
            # Generate code
            generated_code = generator_func()

            # Validate with guard
            guard_passed, guard_feedback = guard_func(generated_code)

            # Record attempt
            attempt = AttemptResult(
                attempt_number=attempt_num,
                generated_code=generated_code,
                guard_passed=guard_passed,
                guard_feedback=guard_feedback,
                timestamp=datetime.now().isoformat(),
            )
            attempts.append(attempt)

            if guard_passed:
                # Success! Record and exit
                phase_result.success = True
                phase_result.final_code = generated_code
                phase_result.attempts = attempts
                phase_result.total_attempts = attempt_num
                return phase_result

            # If not passed and not last attempt, continue to retry

        # All attempts exhausted without success
        phase_result.success = False
        phase_result.attempts = attempts
        phase_result.total_attempts = len(attempts)
        return phase_result

    def run_benchmark(
        self, tasks: list[dict[str, Any]], num_trials: int = 50
    ) -> BenchmarkSummary:
        """Run complete benchmark over all tasks.

        Args:
            tasks: List of task specification dictionaries
            num_trials: Number of trials per task

        Returns:
            BenchmarkSummary with aggregated statistics
        """
        print(f"Running TDD benchmark: {len(tasks)} tasks × {num_trials} trials")
        print(f"Model: {self.model_config['name']}")
        print(f"Retry budget: {self.retry_budget}")
        print("-" * 60)

        all_results = []
        task_summaries = {}
        start_time = time.time()

        for task in tasks:
            task_id = task["task_id"]
            print(f"\nTask: {task['name']} ({task_id})")

            task_results = []
            for trial_num in range(1, num_trials + 1):
                trial_result = self.run_trial(task, trial_num)
                task_results.append(trial_result)

                # Save individual trial result
                self._save_trial_result(trial_result)

                # Print progress
                if trial_num % 10 == 0:
                    success_count = sum(1 for r in task_results if r.workflow_success)
                    success_rate = success_count / len(task_results) * 100
                    print(f"  Trial {trial_num}/{num_trials}: {success_rate:.1f}% success")

            # Calculate task-level statistics
            task_summary = self._calculate_task_summary(task_id, task_results)
            task_summaries[task_id] = task_summary
            all_results.extend(task_results)

            print(
                f"  Final: {task_summary['workflow_success_rate']:.1f}% success "
                f"({task_summary['workflow_successes']}/{num_trials} trials)"
            )

        # Calculate overall summary
        total_time = time.time() - start_time
        summary = self._calculate_benchmark_summary(
            all_results, task_summaries, total_time
        )

        # Save summary
        self._save_summary(summary)

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print(f"Overall workflow success rate: {summary.workflow_success_rate:.1f}%")
        print(f"Training samples (first-attempt): {summary.first_attempt_test_successes}")
        print(f"Training samples (validated): {summary.validated_successes}")
        print(f"Total execution time: {total_time/60:.1f} minutes")
        print("=" * 60)

        return summary

    def _calculate_task_summary(
        self, task_id: str, results: list[TrialResult]
    ) -> dict[str, Any]:
        """Calculate summary statistics for a single task."""
        workflow_successes = sum(1 for r in results if r.workflow_success)
        g_test_successes = sum(
            1 for r in results if r.g_test_result and r.g_test_result.success
        )
        g_impl_successes = sum(
            1 for r in results if r.g_impl_result and r.g_impl_result.success
        )
        first_attempt_successes = sum(1 for r in results if r.first_attempt_test_success)
        validated_successes = sum(1 for r in results if r.validated_success)

        num_trials = len(results)

        return {
            "task_id": task_id,
            "num_trials": num_trials,
            "workflow_successes": workflow_successes,
            "workflow_success_rate": workflow_successes / num_trials * 100,
            "g_test_successes": g_test_successes,
            "g_test_success_rate": g_test_successes / num_trials * 100,
            "g_impl_successes": g_impl_successes,
            "g_impl_success_rate": g_impl_successes / num_trials * 100,
            "first_attempt_test_successes": first_attempt_successes,
            "validated_successes": validated_successes,
        }

    def _calculate_benchmark_summary(
        self,
        all_results: list[TrialResult],
        task_summaries: dict[str, dict[str, Any]],
        total_time: float,
    ) -> BenchmarkSummary:
        """Calculate overall benchmark summary statistics."""
        total_trials = len(all_results)
        workflow_successes = sum(1 for r in all_results if r.workflow_success)
        g_test_successes = sum(
            1 for r in all_results if r.g_test_result and r.g_test_result.success
        )
        g_impl_successes = sum(
            1 for r in all_results if r.g_impl_result and r.g_impl_result.success
        )

        first_attempt_successes = sum(
            1 for r in all_results if r.first_attempt_test_success
        )
        validated_successes = sum(1 for r in all_results if r.validated_success)

        total_attempts = sum(r.total_attempts for r in all_results)

        # Calculate average attempts per phase
        g_test_attempts = [
            r.g_test_result.total_attempts
            for r in all_results
            if r.g_test_result
        ]
        g_impl_attempts = [
            r.g_impl_result.total_attempts
            for r in all_results
            if r.g_impl_result
        ]

        return BenchmarkSummary(
            model_name=self.model_config["name"],
            num_tasks=len(task_summaries),
            num_trials_per_task=total_trials // len(task_summaries),
            total_trials=total_trials,
            workflow_success_rate=workflow_successes / total_trials * 100,
            g_test_success_rate=g_test_successes / total_trials * 100,
            g_impl_success_rate=g_impl_successes / total_trials * 100,
            avg_attempts_per_trial=total_attempts / total_trials,
            avg_g_test_attempts=sum(g_test_attempts) / len(g_test_attempts),
            avg_g_impl_attempts=sum(g_impl_attempts) / len(g_impl_attempts) if g_impl_attempts else 0,
            first_attempt_test_successes=first_attempt_successes,
            validated_successes=validated_successes,
            total_execution_time=total_time,
            avg_time_per_trial=total_time / total_trials,
            task_results=task_summaries,
        )

    def _save_trial_result(self, result: TrialResult) -> None:
        """Save individual trial result to file."""
        if not self.output_dir:
            return

        # Create output directory structure
        model_dir = self.output_dir / self.model_config["name"].replace(":", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save to task-specific file
        task_file = model_dir / f"{result.task_id}_trials.jsonl"
        with open(task_file, "a", encoding="utf-8") as f:
            f.write(result.to_json(indent=None) + "\n")

    def _save_summary(self, summary: BenchmarkSummary) -> None:
        """Save benchmark summary to file."""
        if not self.output_dir:
            return

        model_dir = self.output_dir / self.model_config["name"].replace(":", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        summary_file = model_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary.to_json())

        print(f"\nResults saved to: {model_dir}")
