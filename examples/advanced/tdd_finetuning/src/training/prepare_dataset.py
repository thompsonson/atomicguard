"""Prepare training dataset for fine-tuning."""

import json
from pathlib import Path
from typing import Any


class DatasetPreparer:
    """Prepares training data for LoRA fine-tuning."""

    def __init__(self, train_split: float = 0.9):
        """Initialize dataset preparer.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        self.train_split = train_split

    def prepare_from_jsonl(
        self, input_file: Path, output_dir: Path
    ) -> tuple[Path, Path]:
        """Prepare training and validation datasets from JSONL file.

        Args:
            input_file: Path to JSONL file with training samples
            output_dir: Directory to save prepared datasets

        Returns:
            Tuple of (train_file_path, val_file_path)
        """
        # Load all samples
        samples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))

        if not samples:
            raise ValueError(f"No samples found in {input_file}")

        # Split into train and validation
        split_idx = int(len(samples) * self.train_split)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        print(f"Total samples: {len(samples)}")
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")

        # Prepare output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save datasets
        train_file = output_dir / "train.jsonl"
        val_file = output_dir / "val.jsonl"

        self._save_dataset(train_samples, train_file)
        if val_samples:
            self._save_dataset(val_samples, val_file)

        return train_file, val_file

    def _save_dataset(self, samples: list[dict[str, Any]], output_file: Path) -> None:
        """Save dataset in format suitable for fine-tuning.

        Args:
            samples: List of training samples
            output_file: Output file path
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                # Convert to instruction-following format
                formatted_sample = self._format_for_training(sample)
                f.write(json.dumps(formatted_sample) + "\n")

        print(f"Saved {len(samples)} samples to {output_file}")

    def _format_for_training(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Format sample for instruction fine-tuning.

        Args:
            sample: Original training sample

        Returns:
            Formatted sample with 'instruction', 'input', 'output' fields
        """
        # Format for instruction-following fine-tuning
        # This format is compatible with many fine-tuning frameworks
        return {
            "instruction": "You are an expert Python developer specializing in test-driven development. Generate comprehensive pytest test functions based on the specification.",
            "input": sample["prompt"],
            "output": sample["completion"],
            "metadata": sample.get("metadata", {}),
        }

    def convert_to_chat_format(
        self, input_file: Path, output_file: Path
    ) -> None:
        """Convert dataset to chat/conversation format.

        Args:
            input_file: Path to JSONL file with training samples
            output_file: Output file path
        """
        samples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                chat_sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert Python developer specializing in test-driven development. Generate comprehensive pytest test functions based on specifications."
                        },
                        {
                            "role": "user",
                            "content": sample["prompt"]
                        },
                        {
                            "role": "assistant",
                            "content": sample["completion"]
                        }
                    ],
                    "metadata": sample.get("metadata", {})
                }
                f.write(json.dumps(chat_sample) + "\n")

        print(f"Converted {len(samples)} samples to chat format: {output_file}")

    def create_alpaca_format(
        self, input_file: Path, output_file: Path
    ) -> None:
        """Convert dataset to Alpaca instruction format.

        Args:
            input_file: Path to JSONL file with training samples
            output_file: Output file path
        """
        samples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                formatted = self._format_for_training(sample)
                samples.append(formatted)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save as single JSON array (Alpaca format)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

        print(f"Created Alpaca format dataset: {output_file}")

    def analyze_dataset(self, dataset_file: Path) -> dict[str, Any]:
        """Analyze dataset and return statistics.

        Args:
            dataset_file: Path to JSONL dataset file

        Returns:
            Dictionary with dataset statistics
        """
        samples = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))

        if not samples:
            return {"error": "No samples found"}

        # Calculate statistics
        prompt_lengths = []
        completion_lengths = []
        task_distribution = {}

        for sample in samples:
            if "input" in sample:
                prompt_lengths.append(len(sample["input"]))
                completion_lengths.append(len(sample["output"]))
            else:
                prompt_lengths.append(len(sample.get("prompt", "")))
                completion_lengths.append(len(sample.get("completion", "")))

            task_id = sample.get("metadata", {}).get("task_id")
            if task_id:
                task_distribution[task_id] = task_distribution.get(task_id, 0) + 1

        return {
            "num_samples": len(samples),
            "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
            "avg_completion_length": sum(completion_lengths) / len(completion_lengths),
            "max_prompt_length": max(prompt_lengths),
            "max_completion_length": max(completion_lengths),
            "task_distribution": task_distribution,
        }
