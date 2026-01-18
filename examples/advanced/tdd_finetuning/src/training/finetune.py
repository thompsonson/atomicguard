"""LoRA fine-tuning script for qwen2.5-coder model."""

import json
from pathlib import Path
from typing import Any

from ..config import TRAINING_CONFIG


class LoRAFineTuner:
    """Fine-tune model using LoRA (Low-Rank Adaptation)."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        output_dir: Path | None = None,
        lora_config: dict[str, Any] | None = None,
    ):
        """Initialize LoRA fine-tuner.

        Args:
            base_model: Hugging Face model identifier
            output_dir: Directory to save fine-tuned model
            lora_config: LoRA configuration dictionary
        """
        self.base_model = base_model
        self.output_dir = output_dir or Path("./models/finetuned")
        self.lora_config = lora_config or TRAINING_CONFIG

    def finetune(
        self,
        train_file: Path,
        val_file: Path | None = None,
    ) -> Path:
        """Fine-tune model with LoRA.

        Args:
            train_file: Path to training dataset (JSONL)
            val_file: Path to validation dataset (JSONL, optional)

        Returns:
            Path to saved model
        """
        print("=" * 60)
        print("LORA FINE-TUNING")
        print("=" * 60)
        print(f"Base model: {self.base_model}")
        print(f"Training data: {train_file}")
        print(f"LoRA config: {self.lora_config}")
        print("=" * 60)

        try:
            # Import fine-tuning libraries
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from datasets import load_dataset

            # Load model with 4-bit quantization for efficiency
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=2048,
                dtype=None,  # Auto-detect
                load_in_4bit=True,
            )

            # Apply LoRA adapters
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.lora_config["lora_r"],
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=self.lora_config["lora_alpha"],
                lora_dropout=self.lora_config["lora_dropout"],
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

            # Load datasets
            dataset = load_dataset("json", data_files=str(train_file), split="train")

            # Format dataset for instruction tuning
            def format_prompts(examples):
                instructions = examples["instruction"]
                inputs = examples["input"]
                outputs = examples["output"]

                texts = []
                for instruction, input_text, output in zip(instructions, inputs, outputs):
                    text = f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
                    texts.append(text)
                return {"text": texts}

            dataset = dataset.map(format_prompts, batched=True)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                per_device_train_batch_size=self.lora_config["batch_size"],
                gradient_accumulation_steps=4,
                warmup_steps=self.lora_config["warmup_steps"],
                num_train_epochs=self.lora_config["num_epochs"],
                learning_rate=self.lora_config["learning_rate"],
                fp16=True,
                logging_steps=10,
                save_strategy="epoch",
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=42,
            )

            # Create trainer
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=2048,
                args=training_args,
            )

            # Train
            print("\nStarting training...")
            trainer.train()

            # Save model
            print(f"\nSaving model to {self.output_dir}")
            model.save_pretrained(str(self.output_dir))
            tokenizer.save_pretrained(str(self.output_dir))

            # Save training config
            config_file = self.output_dir / "training_config.json"
            with open(config_file, "w") as f:
                json.dump(self.lora_config, f, indent=2)

            print("=" * 60)
            print("FINE-TUNING COMPLETE")
            print(f"Model saved to: {self.output_dir}")
            print("=" * 60)

            return self.output_dir

        except ImportError as e:
            print(f"\nError: Required library not found: {e}")
            print("\nTo install required libraries for fine-tuning:")
            print("  pip install unsloth transformers trl datasets peft")
            raise

    def convert_to_ollama(self, model_dir: Path, model_name: str) -> None:
        """Convert fine-tuned model to Ollama format.

        Args:
            model_dir: Directory containing fine-tuned model
            model_name: Name for Ollama model

        Note:
            This requires additional steps and is provided as a guide.
            See Ollama documentation for latest conversion methods.
        """
        print("\n" + "=" * 60)
        print("CONVERTING TO OLLAMA FORMAT")
        print("=" * 60)

        print(f"""
To use this fine-tuned model with Ollama:

1. Convert to GGUF format:
   python -m llama_cpp.convert {model_dir} --outtype f16

2. Create Modelfile:
   FROM ./model.gguf
   PARAMETER temperature 0.2
   PARAMETER top_p 0.9

3. Import to Ollama:
   ollama create {model_name} -f Modelfile

4. Test the model:
   ollama run {model_name}

See: https://github.com/ollama/ollama/blob/main/docs/import.md
        """)
        print("=" * 60)


def main():
    """Example usage of LoRA fine-tuner."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA")
    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Path to training dataset (JSONL)"
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        help="Path to validation dataset (JSONL)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models/finetuned"),
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Base model to fine-tune"
    )

    args = parser.parse_args()

    # Create fine-tuner
    finetuner = LoRAFineTuner(
        base_model=args.base_model,
        output_dir=args.output_dir,
    )

    # Run fine-tuning
    model_path = finetuner.finetune(args.train_file, args.val_file)

    # Print conversion instructions
    finetuner.convert_to_ollama(model_path, "qwen2.5-coder:7b-tdd-finetuned")


if __name__ == "__main__":
    main()
