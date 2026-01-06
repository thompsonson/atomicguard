# Learning Loop Implementation Guide

This document provides practical implementation guidance for the Learning Loop extension (Definitions 21-24), with a focus on fine-tuning using Unsloth and LoRA adapters.

> **Depends on**: [04_learning_loop.md](04_learning_loop.md) (formal definitions)
>
> **See also**: [Unsloth Documentation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)

---

## 1. Overview

This guide covers:

1. **Dataset extraction** from ℛ per Definition 22 (Training Trace)
2. **Prompt formatting** for conditioning on Ψ and H
3. **Unsloth integration** for efficient fine-tuning
4. **LoRA adapters** to mitigate catastrophic forgetting
5. **Filtering strategies** for different training policies
6. **Incremental training** using checkpoints
7. **Evaluation** measuring E[retries] → 0

---

## 2. Dataset Extraction

Extract training traces from the repository per Definition 22.

```python
from atomicguard.dag import ArtifactDAG
from atomicguard.types import ArtifactStatus, RepositoryItem

def extract_training_data(dag: ArtifactDAG, filter_fn=None) -> list[dict]:
    """
    Extract training traces per Definition 22.

    Implements: τ = E(ℛ, Φ_training)

    Args:
        dag: The artifact DAG (ℛ)
        filter_fn: Optional custom filter (Φ_training).
                   Defaults to Φ_refinement (retry→success chains).

    Returns:
        List of training examples with prompt, completion, and metadata.
    """
    # Default: Φ_refinement - successful items with prior failures
    if filter_fn is None:
        filter_fn = lambda r: (
            r.status == ArtifactStatus.ACCEPTED and
            len(get_provenance_chain(r)) > 1
        )

    traces = dag.extract(
        filter=filter_fn,
        order_by="created_at"
    )

    return [
        {
            "prompt": format_prompt(r.spec, r.feedback_history),
            "completion": r.artifact_content,
            "text": format_full_example(r),  # Combined for SFTTrainer
            "metadata": {
                "source": r.source,
                "action_pair_id": r.action_pair_id,
                "workflow_id": r.workflow_id,
                "retry_count": len(get_provenance_chain(r)) - 1,
                "item_id": r.id
            }
        }
        for r in traces
    ]


def get_provenance_chain(r: RepositoryItem) -> list[RepositoryItem]:
    """Traverse provenance links to get full refinement chain."""
    chain = [r]
    current = r
    while current.metadata.get("parent_id"):
        parent = dag.get_item(current.metadata["parent_id"])
        if parent:
            chain.append(parent)
            current = parent
        else:
            break
    return list(reversed(chain))
```

---

## 3. Prompt Formatting

Format specification and feedback history for training per Definition 24.

```python
def format_prompt(spec: str, history: list[dict]) -> str:
    """
    Format specification and feedback history for training.

    Implements conditioning: π_θ(r.a | r.Ψ, r.H)

    Args:
        spec: The specification (r.Ψ)
        history: Feedback history (r.H) - list of {artifact, feedback} dicts

    Returns:
        Formatted prompt string
    """
    prompt = f"### Specification\n{spec}\n\n"

    if history:
        prompt += "### Previous Attempts and Feedback\n"
        # Include last N attempts to avoid context overflow
        for entry in history[-3:]:
            artifact_preview = entry.get("artifact", "")[:500]
            feedback = entry.get("feedback", "")
            prompt += f"**Attempt:**\n```\n{artifact_preview}\n```\n\n"
            prompt += f"**Feedback:** {feedback}\n\n"

    prompt += "### Solution\n"
    return prompt


def format_full_example(r: RepositoryItem) -> str:
    """
    Format complete training example (prompt + completion).

    Used for SFTTrainer's dataset_text_field.
    """
    prompt = format_prompt(r.spec, r.feedback_history)
    completion = r.artifact_content

    # Use chat template format for instruction-tuned models
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{completion}<|eot_id|>"""
```

### Alternative: Alpaca Format

```python
def format_alpaca(r: RepositoryItem) -> dict:
    """Format for Alpaca-style instruction tuning."""
    instruction = r.spec

    # Include feedback as additional input context
    input_context = ""
    if r.feedback_history:
        input_context = "Previous feedback:\n"
        for entry in r.feedback_history[-2:]:
            input_context += f"- {entry.get('feedback', '')}\n"

    return {
        "instruction": instruction,
        "input": input_context,
        "output": r.artifact_content
    }
```

---

## 4. Unsloth Integration

Use Unsloth for efficient fine-tuning with 4-bit quantization.

```python
from unsloth import FastLanguageModel
import torch

def load_model_for_training(
    model_name: str = "unsloth/llama-3-8b-Instruct",
    max_seq_length: int = 4096,
    load_in_4bit: bool = True
):
    """
    Load base model with Unsloth optimizations.

    Reference: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=load_in_4bit,
    )

    return model, tokenizer


def add_lora_adapters(model, r: int = 16, lora_alpha: int = 16):
    """
    Add LoRA adapters to mitigate catastrophic forgetting.

    LoRA keeps base model frozen, only training adapter weights.
    This preserves general capabilities while learning codebase-specific patterns.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,  # Rank - higher = more capacity, more memory
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,  # Optimized for Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=42,
    )

    return model
```

---

## 5. Training Configuration

Configure and run supervised fine-tuning.

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

def train_on_traces(
    model,
    tokenizer,
    training_data: list[dict],
    output_dir: str = "outputs",
    max_steps: int = 100,
    batch_size: int = 2,
    learning_rate: float = 2e-4
):
    """
    Fine-tune model on extracted training traces.

    Implements Definition 24: L(θ) = -E_τ[log π_θ(r.a | r.Ψ, r.H)]
    """
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(training_data)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",  # Uses format_full_example output
        max_seq_length=4096,
        dataset_num_proc=2,
        packing=False,  # Set True for short examples
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=50,
        ),
    )

    trainer.train()
    return trainer


def save_trained_model(model, tokenizer, output_path: str):
    """Save LoRA adapters (not full model) for efficient storage."""
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # For full model merge (optional, larger file):
    # model.save_pretrained_merged(output_path + "_merged", tokenizer)
```

---

## 6. Filtering Strategies

Implement different training policies via filter predicates (Definition 22).

```python
from datetime import datetime, timedelta
from atomicguard.types import ArtifactStatus

# --- Basic Predicates ---

def Φ_status(status: ArtifactStatus):
    """Select by status."""
    return lambda r: r.status == status

def Φ_source(source: str):
    """Select by source (GENERATOR or HUMAN)."""
    return lambda r: r.source == source

def Φ_action_pair(action_pair_id: str):
    """Select by action pair."""
    return lambda r: r.action_pair_id == action_pair_id

def Φ_timestamp_after(cutoff: datetime):
    """Select items created after cutoff."""
    return lambda r: r.created_at > cutoff


# --- Compound Predicates ---

def and_(*predicates):
    """Conjunction of predicates."""
    return lambda r: all(p(r) for p in predicates)

def or_(*predicates):
    """Disjunction of predicates."""
    return lambda r: any(p(r) for p in predicates)


# --- Common Training Policies ---

def Φ_refinement(dag):
    """
    Definition 21: Refinement Predicate
    Select successful items with prior failures.
    """
    return lambda r: (
        r.status == ArtifactStatus.ACCEPTED and
        len(get_provenance_chain(r)) > 1
    )

def Φ_generator_only(dag):
    """Generator refinements only (exclude human artifacts)."""
    return and_(Φ_refinement(dag), Φ_source("GENERATOR"))

def Φ_human_only(dag):
    """Human refinements only (for preference alignment)."""
    return and_(Φ_refinement(dag), Φ_source("HUMAN"))

def Φ_high_value(dag, min_retries: int = 3):
    """High-value traces (many retries = hard problems)."""
    return lambda r: (
        r.status == ArtifactStatus.ACCEPTED and
        len(get_provenance_chain(r)) >= min_retries
    )

def Φ_recent(dag, days: int = 30):
    """Recent traces only (avoid specification drift)."""
    cutoff = datetime.now() - timedelta(days=days)
    return and_(Φ_refinement(dag), Φ_timestamp_after(cutoff))

def Φ_all_successes():
    """All successful artifacts (including first-attempt)."""
    return Φ_status(ArtifactStatus.ACCEPTED)
```

### Usage Example

```python
# Train only on generator refinements from the last 30 days
training_data = extract_training_data(
    dag,
    filter_fn=and_(
        Φ_generator_only(dag),
        Φ_recent(dag, days=30)
    )
)
```

---

## 7. Incremental Training

Train incrementally as new traces accumulate, using checkpoints (Definition 14).

```python
import json
from pathlib import Path

CHECKPOINT_FILE = "training_checkpoint.json"

def load_training_checkpoint() -> dict:
    """Load last training checkpoint."""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"last_item_id": None, "timestamp": None, "traces_processed": 0}


def save_training_checkpoint(last_item_id: str, traces_processed: int):
    """Save training checkpoint."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "last_item_id": last_item_id,
            "timestamp": datetime.now().isoformat(),
            "traces_processed": traces_processed
        }, f)


def incremental_training_loop(
    dag: ArtifactDAG,
    model,
    tokenizer,
    checkpoint_interval: int = 100,
    min_new_traces: int = 10
):
    """
    Train incrementally as new traces accumulate.

    Uses Definition 14 (Checkpoint) for resumability.

    Args:
        dag: The artifact DAG
        model: The model to fine-tune
        tokenizer: The tokenizer
        checkpoint_interval: Save checkpoint every N traces
        min_new_traces: Minimum new traces before training
    """
    checkpoint = load_training_checkpoint()

    # Build filter for new traces since checkpoint
    def Φ_new_traces(r):
        if checkpoint["last_item_id"] is None:
            return Φ_refinement(dag)(r)
        return (
            Φ_refinement(dag)(r) and
            r.created_at > datetime.fromisoformat(checkpoint["timestamp"])
        )

    # Extract new traces
    new_traces = extract_training_data(dag, filter_fn=Φ_new_traces)

    if len(new_traces) < min_new_traces:
        print(f"Only {len(new_traces)} new traces. Waiting for more data.")
        return

    print(f"Training on {len(new_traces)} new traces...")

    # Fine-tune
    trainer = train_on_traces(
        model, tokenizer, new_traces,
        max_steps=len(new_traces) // 2  # Rough heuristic
    )

    # Save checkpoint
    last_item = new_traces[-1]
    save_training_checkpoint(
        last_item["metadata"]["item_id"],
        checkpoint["traces_processed"] + len(new_traces)
    )

    print(f"Checkpoint saved. Total traces processed: {checkpoint['traces_processed'] + len(new_traces)}")
```

---

## 8. Evaluation

Measure improvement in first-attempt success rate.

```python
from typing import Callable

def evaluate_improvement(
    model_before,
    model_after,
    tokenizer,
    test_specs: list[str],
    guard: Callable,
    num_samples: int = 5
) -> dict:
    """
    Measure improvement in first-attempt success rate.

    Goal: E[retries] → 0 (per Section 6.1.4)

    Args:
        model_before: Model before fine-tuning
        model_after: Model after fine-tuning
        tokenizer: Tokenizer for both models
        test_specs: List of test specifications
        guard: Guard function to validate outputs
        num_samples: Samples per spec for statistical significance

    Returns:
        Dict with before/after success rates and improvement
    """
    results = {
        "before": {"successes": 0, "total": 0},
        "after": {"successes": 0, "total": 0}
    }

    for spec in test_specs:
        prompt = format_prompt(spec, [])  # No history for first attempt

        for _ in range(num_samples):
            # Generate with model before
            output_before = generate(model_before, tokenizer, prompt)
            if guard(output_before).passed:
                results["before"]["successes"] += 1
            results["before"]["total"] += 1

            # Generate with model after
            output_after = generate(model_after, tokenizer, prompt)
            if guard(output_after).passed:
                results["after"]["successes"] += 1
            results["after"]["total"] += 1

    before_rate = results["before"]["successes"] / results["before"]["total"]
    after_rate = results["after"]["successes"] / results["after"]["total"]

    return {
        "before_success_rate": before_rate,
        "after_success_rate": after_rate,
        "improvement": after_rate - before_rate,
        "relative_improvement": (after_rate - before_rate) / before_rate if before_rate > 0 else float('inf')
    }


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate completion from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 9. Complete Training Pipeline

Putting it all together:

```python
def run_training_pipeline(
    dag: ArtifactDAG,
    model_name: str = "unsloth/llama-3-8b-Instruct",
    output_dir: str = "trained_model",
    filter_policy: str = "refinement"  # or "generator", "human", "all"
):
    """
    Complete training pipeline from repository to fine-tuned model.
    """
    # 1. Load model with LoRA
    print("Loading model...")
    model, tokenizer = load_model_for_training(model_name)
    model = add_lora_adapters(model)

    # 2. Select filter policy
    filter_map = {
        "refinement": Φ_refinement(dag),
        "generator": Φ_generator_only(dag),
        "human": Φ_human_only(dag),
        "all": Φ_all_successes()
    }
    filter_fn = filter_map.get(filter_policy, Φ_refinement(dag))

    # 3. Extract training data
    print(f"Extracting training data with policy: {filter_policy}...")
    training_data = extract_training_data(dag, filter_fn=filter_fn)
    print(f"Found {len(training_data)} training examples")

    if len(training_data) == 0:
        print("No training data found. Exiting.")
        return

    # 4. Train
    print("Starting training...")
    trainer = train_on_traces(
        model, tokenizer, training_data,
        output_dir=output_dir,
        max_steps=min(len(training_data) * 2, 500)
    )

    # 5. Save
    print(f"Saving model to {output_dir}...")
    save_trained_model(model, tokenizer, output_dir)

    print("Training complete!")
    return model, tokenizer


# Usage
if __name__ == "__main__":
    from atomicguard.dag import ArtifactDAG

    dag = ArtifactDAG.load("path/to/repository")
    model, tokenizer = run_training_pipeline(
        dag,
        filter_policy="generator",
        output_dir="./codebase_specialist"
    )
```

---

## 10. Summary

| Step | Function | Definition |
|------|----------|------------|
| Extract | `extract_training_data()` | Definition 22 (Training Trace) |
| Format | `format_prompt()` | Definition 24 (conditioning on Ψ, H) |
| Load | `load_model_for_training()` | — |
| Adapt | `add_lora_adapters()` | Mitigates catastrophic forgetting |
| Train | `train_on_traces()` | Definition 24 (Policy Update) |
| Filter | `Φ_*` predicates | Definition 21-22 (filter policies) |
| Checkpoint | `incremental_training_loop()` | Definition 14 (Checkpoint) |
| Evaluate | `evaluate_improvement()` | Goal: E[retries] → 0 |

---

## See Also

- [04_learning_loop.md](04_learning_loop.md) — Formal definitions (21-24)
- [Unsloth Documentation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) — Fine-tuning guide
- [LoRA Paper](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation
- Paper Section 6.1.4 — Policy Distillation (Tier 4)
