"""Extract training data from benchmark results."""

import json
from pathlib import Path
from typing import Any

from ..benchmark.trial import TrialResult
from ..config import QUALITY_STANDARDS


class TrainingDataExtractor:
    """Extracts high-quality training samples from benchmark results."""

    def __init__(
        self,
        first_attempt_success: bool = True,
        validated_success: bool = True,
    ):
        """Initialize data extractor with quality standards.

        Args:
            first_attempt_success: Require g_test to pass on first attempt
            validated_success: Require both g_test and g_impl to succeed
        """
        self.first_attempt_success = first_attempt_success
        self.validated_success = validated_success

    def extract_from_results_dir(self, results_dir: Path) -> list[dict[str, Any]]:
        """Extract training samples from a results directory.

        Args:
            results_dir: Directory containing trial result files (*.jsonl)

        Returns:
            List of training samples (prompt-completion pairs)
        """
        training_samples = []

        # Find all trial result files
        for trial_file in results_dir.glob("*_trials.jsonl"):
            with open(trial_file, "r", encoding="utf-8") as f:
                for line in f:
                    trial_data = json.loads(line)
                    trial_result = TrialResult.from_dict(trial_data)

                    # Check if trial meets quality standards
                    if self._meets_quality_standards(trial_result):
                        sample = self._create_training_sample(trial_result)
                        if sample:
                            training_samples.append(sample)

        return training_samples

    def _meets_quality_standards(self, trial: TrialResult) -> bool:
        """Check if trial meets quality standards for training data.

        Args:
            trial: TrialResult to evaluate

        Returns:
            True if trial meets quality standards
        """
        # Check first attempt success requirement
        if self.first_attempt_success and not trial.first_attempt_test_success:
            return False

        # Check validated success requirement
        if self.validated_success and not trial.validated_success:
            return False

        return True

    def _create_training_sample(self, trial: TrialResult) -> dict[str, Any] | None:
        """Create training sample from trial result.

        Args:
            trial: TrialResult to convert

        Returns:
            Training sample dictionary or None if invalid
        """
        if not trial.g_test_result or not trial.g_test_result.success:
            return None

        # Get the successful test code (from first attempt)
        if not trial.g_test_result.attempts:
            return None

        first_attempt = trial.g_test_result.attempts[0]
        if not first_attempt.guard_passed:
            return None

        # Get task specification from metadata or reconstruct
        task_spec = trial.metadata.get("task_spec", {})

        # Create prompt-completion pair
        return {
            "task_id": trial.task_id,
            "trial_number": trial.trial_number,
            "prompt": self._build_training_prompt(task_spec),
            "completion": first_attempt.generated_code,
            "metadata": {
                "workflow_success": trial.workflow_success,
                "first_attempt": True,
                "validated": trial.validated_success,
            },
        }

    def _build_training_prompt(self, task_spec: dict[str, Any]) -> str:
        """Build training prompt from task specification.

        Args:
            task_spec: Task specification dictionary

        Returns:
            Formatted training prompt
        """
        specification = task_spec.get("specification", "")
        task_name = task_spec.get("name", "Unknown Task")

        # Match the format used by TestGenerator
        prompt = f"""Generate pytest test functions for the following task:

Task: {task_name}

Specification:
{specification}

Requirements:
1. Write comprehensive pytest test functions that cover the examples and edge cases
2. Include at least 5-7 test cases covering normal cases, edge cases, and error conditions
3. Use descriptive test function names (test_<scenario>)
4. Include docstrings explaining what each test validates
5. Make tests independent and self-contained
6. Do NOT include the implementation - only test functions
7. Import pytest at the top
8. Assume the function being tested will be imported from a module called 'solution'

Output ONLY the Python test code, no explanations."""

        return prompt

    def save_training_data(
        self, samples: list[dict[str, Any]], output_file: Path
    ) -> None:
        """Save training samples to JSONL file.

        Args:
            samples: List of training samples
            output_file: Output file path
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        print(f"Saved {len(samples)} training samples to {output_file}")

    def print_statistics(self, samples: list[dict[str, Any]]) -> None:
        """Print statistics about extracted training data.

        Args:
            samples: List of training samples
        """
        print("\n" + "=" * 60)
        print("TRAINING DATA EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total samples extracted: {len(samples)}")

        if not samples:
            print("No samples found meeting quality standards")
            return

        # Count by task
        task_counts = {}
        for sample in samples:
            task_id = sample["task_id"]
            task_counts[task_id] = task_counts.get(task_id, 0) + 1

        print(f"\nSamples per task:")
        for task_id, count in sorted(task_counts.items()):
            print(f"  {task_id}: {count}")

        # Count validated vs non-validated
        validated = sum(1 for s in samples if s["metadata"].get("validated", False))
        print(f"\nValidated samples (g_impl also succeeded): {validated}/{len(samples)}")

        # Calculate average lengths
        avg_prompt_len = sum(len(s["prompt"]) for s in samples) / len(samples)
        avg_completion_len = sum(len(s["completion"]) for s in samples) / len(samples)

        print(f"\nAverage prompt length: {avg_prompt_len:.0f} chars")
        print(f"Average completion length: {avg_completion_len:.0f} chars")
        print("=" * 60)
