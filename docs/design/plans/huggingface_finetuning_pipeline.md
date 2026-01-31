# Plan: Fine-Tuning via HuggingFace Platform

> **Status**: Planned
>
> **Depends on**: [04_learning_loop.md](../extensions/04_learning_loop.md) (Definitions 21-24), [05_learning_implementation.md](../extensions/05_learning_implementation.md) (practical guide)
>
> **Related**: `src/atomicguard/infrastructure/llm/huggingface.py` (HuggingFaceGenerator)

---

## Goal

Extract training traces from the AtomicGuard artifact DAG (ℛ), export as JSONL compatible with HuggingFace AutoTrain, upload to the Hub, and fine-tune remotely — no local GPU required.

The full pipeline runs without any local GPU hardware. Training happens on HuggingFace's infrastructure via AutoTrain, and inference happens via Inference Providers (serverless, off-the-shelf models) or Inference Endpoints (dedicated, for fine-tuned models).

---

## Prerequisites

### Inference (complete)

- `HuggingFaceGenerator(GeneratorInterface)` — implemented in `src/atomicguard/infrastructure/llm/huggingface.py`
- Uses `huggingface_hub.InferenceClient` for chat completion
- Registered as entry point in `pyproject.toml`
- 39 tests in `tests/infrastructure/llm/test_huggingface.py`

### Inference (remaining)

- Add `endpoint_url` config field to `HuggingFaceGenerator` for connecting to dedicated Inference Endpoints (where fine-tuned models are deployed). Currently only supports the serverless Inference Providers path.

---

## Implementation Steps

### Step 1: Training Trace Extractor

**Location**: `src/atomicguard/application/training_export.py`

Implement Definition 22 (Training Trace) from `04_learning_loop.md`:

- Walk the artifact DAG looking for retry→success chains (Definition 21: Refinement Predicate)
- For each ACCEPTED artifact with prior REJECTED attempts, collect:
  - `r.Ψ` — specification snapshot (prompt)
  - `r.H` — feedback history (reasoning context from guard rejections)
  - `r.a` — successful artifact content (completion)
- Support the composable filter predicates from the design doc:
  - `Φ_refinement` — retry→success chains only
  - `Φ_source(GENERATOR)` — exclude human-provided artifacts
  - `Φ_source(HUMAN)` — human traces only (preference alignment)
  - `Φ_recent(days=N)` — avoid specification drift
  - `Φ_high_value(min_retries=N)` — focus on hard problems
  - `Φ_all_successes` — include first-attempt successes
- Output: list of `(prompt, completion, metadata)` tuples

### Step 2: JSONL Formatter

**Location**: same module or a sibling

Format the extracted traces as chat-completion JSONL for AutoTrain:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "<spec + feedback history>"}, {"role": "assistant", "content": "<accepted artifact>"}]}
```

- **System message**: the prompt template's `role` + `constraints` (from `prompts.json`) — this is how the generator builds its system prompt today
- **User message**: specification + feedback history (the conditioning context from Definition 24)
- **Assistant message**: the accepted artifact content
- One line per training example

This format works with both AutoTrain and standard SFT tooling (`trl` `SFTTrainer`).

### Step 3: CLI Command

**Location**: new CLI module or added to an existing one

```bash
uv run python -m atomicguard.tools.export_training \
    --dag-path ./artifacts \
    --output training_data.jsonl \
    --filter refinement \
    --prompt-template prompts.json:g_coder
```

Options:

| Option | Description | Default |
|--------|-------------|---------|
| `--dag-path` | Path to the filesystem artifact DAG | required |
| `--output` | Output JSONL file path | `training_data.jsonl` |
| `--filter` | Predicate policy | `refinement` |
| `--prompt-template` | Which prompts.json entry for system message | optional |
| `--min-retries` | For `high_value` filter | `3` |
| `--since` | Date cutoff for `recent` filter | none |

Filter options: `refinement`, `generator_only`, `human_only`, `high_value`, `recent`, `all_successes`

### Step 4: Upload to HuggingFace Hub

Either manual or programmatic:

```bash
# Manual
huggingface-cli upload thompsonson/atomicguard-traces training_data.jsonl

# Or programmatic via huggingface_hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="training_data.jsonl",
    path_in_repo="training_data.jsonl",
    repo_id="thompsonson/atomicguard-traces",
    repo_type="dataset",
)
```

This creates a dataset on the Hub that AutoTrain can consume directly.

### Step 5: Fine-Tune via AutoTrain

On HuggingFace (no local GPU):

1. Go to [AutoTrain](https://huggingface.co/autotrain) → New Project
2. Select base model (e.g. `Qwen/Qwen2.5-Coder-32B-Instruct`)
3. Point to the uploaded dataset
4. Configure: LoRA rank, learning rate, epochs
5. Launch — runs on HF GPUs, produces a model repo with LoRA adapters

The output is a new model repo on the Hub (e.g. `thompsonson/atomicguard-coder-v1`).

### Step 6: Deploy Fine-Tuned Model

Deploy the AutoTrain output model on Inference Endpoints (dedicated):

1. Go to [Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated)
2. Select the fine-tuned model repo
3. Pick GPU, region, autoscaling config
4. Get endpoint URL (e.g. `https://xyz123.endpoints.huggingface.cloud`)
5. Use with `HuggingFaceGenerator` via `endpoint_url` config

### Step 7: Tests

- Unit tests for the trace extractor: mock artifact DAG with known retry chains, verify correct traces extracted
- Unit tests for JSONL formatter: verify output format matches AutoTrain chat-completion schema
- Tests for filter predicates: verify composability, edge cases (no successes, first-attempt only, empty DAG)
- Integration test: extract from a real `FilesystemArtifactDAG` with sample data

---

## Dependency Graph

```
Step 1 (extractor) ──→ Step 2 (formatter) ──→ Step 3 (CLI)
                                                   │
                                              Step 4 (upload to Hub)
                                                   │
                                              Step 5 (AutoTrain on HF)
                                                   │
              endpoint_url on HFGenerator ──→ Step 6 (deploy on Inference Endpoints)
```

Steps 1-3 are code work (in this repo). Steps 4-6 are operational (HuggingFace platform). The `endpoint_url` addition to `HuggingFaceGenerator` can happen in parallel with any of this.

---

## Mapping to Formal Definitions

| Step | Definition | Description |
|------|-----------|-------------|
| Trace extraction | Definition 21 (Refinement Predicate) | Select retry→success chains from ℛ |
| Filter policies | Definition 22 (Training Trace) | Composable Φ predicates |
| JSONL format | Definition 24 (Policy Update) | Conditioning on (Ψ, H, a) |
| Evaluation | Theorem 9 (Completeness) | Repository items are self-contained training data |
| System preservation | Theorem 10 | Extraction is read-only, training is external |

---

## Evaluation Criteria

Per Section 6.1.4 of the paper, the goal is `E[retries] → 0`:

- Measure first-attempt success rate (epsilon) before and after fine-tuning
- Use the existing `epsilon` CLI command in the g_plan_benchmark
- Compare: `--backend huggingface --model base-model` vs `--backend huggingface --model fine-tuned-model`
- Report improvement as `Δε = ε_after - ε_before`

---

## See Also

- [04_learning_loop.md](../extensions/04_learning_loop.md) — Formal definitions (21-24)
- [05_learning_implementation.md](../extensions/05_learning_implementation.md) — Implementation guide (Unsloth/LoRA, local GPU path)
- [06_generated_workflows.md](../extensions/06_generated_workflows.md) — Generated workflows (the plans G_plan validates)
- `src/atomicguard/infrastructure/llm/huggingface.py` — HuggingFaceGenerator
- `examples/advanced/g_plan_benchmark/` — Epsilon estimation benchmark
