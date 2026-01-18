#!/usr/bin/env python3
"""Extract training data from baseline benchmark results."""

import argparse
from pathlib import Path

from src.training.data_extractor import TrainingDataExtractor
from src.training.prepare_dataset import DatasetPreparer
from src.config import RESULTS_DIR, QUALITY_STANDARDS


def main():
    """Extract and prepare training data."""
    parser = argparse.ArgumentParser(
        description="Extract training data from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from baseline results with default quality standards
  python extract_training_data.py

  # Extract with custom quality requirements
  python extract_training_data.py --no-first-attempt --no-validated

  # Extract and prepare for fine-tuning
  python extract_training_data.py --prepare-dataset
        """
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "baseline" / "qwen2.5-coder_7b",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("./data/training_samples.jsonl"),
        help="Output file for training samples"
    )
    parser.add_argument(
        "--first-attempt",
        action="store_true",
        default=QUALITY_STANDARDS["first_attempt_success"],
        help="Require g_test to pass on first attempt"
    )
    parser.add_argument(
        "--no-first-attempt",
        action="store_false",
        dest="first_attempt",
        help="Don't require first attempt success"
    )
    parser.add_argument(
        "--validated",
        action="store_true",
        default=QUALITY_STANDARDS["validated_success"],
        help="Require both g_test and g_impl to succeed"
    )
    parser.add_argument(
        "--no-validated",
        action="store_false",
        dest="validated",
        help="Don't require validated success"
    )
    parser.add_argument(
        "--prepare-dataset",
        action="store_true",
        help="Also prepare train/val split for fine-tuning"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./data/prepared"),
        help="Directory for prepared datasets"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING DATA EXTRACTION")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Quality standards:")
    print(f"  - First attempt success: {args.first_attempt}")
    print(f"  - Validated success: {args.validated}")
    print("=" * 60)

    # Create data extractor
    extractor = TrainingDataExtractor(
        first_attempt_success=args.first_attempt,
        validated_success=args.validated,
    )

    # Extract training samples
    print("\nExtracting training samples...")
    samples = extractor.extract_from_results_dir(args.results_dir)

    # Print statistics
    extractor.print_statistics(samples)

    # Save samples
    extractor.save_training_data(samples, args.output_file)

    # Prepare dataset if requested
    if args.prepare_dataset:
        print("\n" + "=" * 60)
        print("PREPARING DATASET FOR FINE-TUNING")
        print("=" * 60)

        preparer = DatasetPreparer(train_split=0.9)
        train_file, val_file = preparer.prepare_from_jsonl(
            args.output_file,
            args.dataset_dir
        )

        # Also create chat format
        chat_dir = args.dataset_dir / "chat_format"
        preparer.convert_to_chat_format(
            train_file,
            chat_dir / "train_chat.jsonl"
        )
        if val_file.exists():
            preparer.convert_to_chat_format(
                val_file,
                chat_dir / "val_chat.jsonl"
            )

        # Analyze dataset
        stats = preparer.analyze_dataset(train_file)
        print("\nDataset Statistics:")
        print(f"  Training samples: {stats['num_samples']}")
        print(f"  Avg prompt length: {stats['avg_prompt_length']:.0f} chars")
        print(f"  Avg completion length: {stats['avg_completion_length']:.0f} chars")

        print("\n" + "=" * 60)
        print("DATASET PREPARATION COMPLETE")
        print(f"Train file: {train_file}")
        print(f"Val file: {val_file}")
        print(f"Chat format: {chat_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
