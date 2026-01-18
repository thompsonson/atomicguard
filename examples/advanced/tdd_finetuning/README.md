# TDD Workflow Fine-Tuning Experiment

This experiment demonstrates measurable learning in the AtomicGuard framework by fine-tuning `qwen2.5-coder:7b` on successful test generations from a TDD workflow benchmark.

## Overview

The experiment follows a three-phase protocol:

1. **Baseline Phase**: Run TDD workflow benchmark with base `qwen2.5-coder:7b` model
2. **Fine-Tuning Phase**: Extract high-quality samples and fine-tune model with LoRA
3. **Evaluation Phase**: Re-run benchmark with fine-tuned model and compare results

**Goal**: Demonstrate that models can improve on specific tasks through targeted fine-tuning on successful workflow traces.

## Background

This experiment builds on the TDD Workflow Benchmark from the AtomicGuard paper (Section 4.3, lines 897-956). The paper evaluated three model sizes of `qwen2.5-coder` on a two-step TDD workflow:

- **g_test**: Generate pytest tests from specification (validated by SyntaxGuard)
- **g_impl**: Generate implementation that passes tests (validated by DynamicTestGuard)

Paper results (50 trials per task, 6 tasks):
- **3B model**: 36% workflow success rate
- **7B model**: 57% workflow success rate
- **14B model**: 70% workflow success rate

Our experiment asks: **Can we improve the 7B model's performance through fine-tuning on successful test generations?**

## Directory Structure

```
examples/advanced/tdd_finetuning/
├── README.md                  # This file
├── EXPERIMENT.md              # Research design and methodology
├── IMPLEMENTATION_PLAN.md     # Technical implementation details
├── tasks/                     # Task specifications (6 tasks)
│   ├── balanced_brackets.json
│   ├── fibonacci.json
│   ├── matrix_transpose.json
│   ├── prime_checker.json
│   ├── reverse_linked_list.json
│   └── valid_parentheses.json
├── src/                       # Source code
│   ├── config.py             # Configuration
│   ├── generators/           # Test & implementation generators
│   ├── guards/               # Syntax & test guards
│   ├── benchmark/            # Benchmark runner
│   ├── training/             # Data extraction & fine-tuning
│   └── analysis/             # Statistical analysis
├── results/                   # Benchmark results
│   ├── baseline/             # Baseline model results
│   ├── finetuned/            # Fine-tuned model results
│   └── analysis/             # Comparison reports
├── data/                      # Training data
│   ├── training_samples.jsonl
│   └── prepared/             # Train/val splits
├── models/                    # Fine-tuned models
└── *.py                       # CLI scripts
```

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- GPU with 16GB+ VRAM (for fine-tuning)

### Install Dependencies

```bash
# Install base dependencies
pip install openai pytest scipy numpy

# For fine-tuning (optional, only needed for Phase 2)
pip install torch transformers datasets peft trl
pip install unsloth  # For efficient LoRA fine-tuning
```

### Pull Base Model

```bash
ollama pull qwen2.5-coder:7b
```

## Usage

### Phase 1: Baseline Benchmark

Run the TDD workflow benchmark with the base 7B model:

```bash
python run_benchmark.py --model baseline --trials 50
```

**What this does:**
- Loads 6 task specifications from `tasks/`
- For each task, runs 50 trials of the TDD workflow:
  1. Generate pytest tests (g_test) with up to 3 retry attempts
  2. If tests pass syntax validation, generate implementation (g_impl)
  3. If implementation passes tests, mark as success
- Saves detailed results to `results/baseline/qwen2.5-coder_7b/`
- Prints summary statistics

**Expected output:**
- Workflow success rate: ~57% (per paper)
- ~100-120 first-attempt test successes (training data candidates)
- Execution time: ~2-4 hours (depending on hardware)

### Phase 2: Fine-Tuning

#### Step 1: Extract Training Data

Extract high-quality training samples from baseline results:

```bash
python extract_training_data.py --prepare-dataset
```

**Quality standards** (OUR defined standards, not from paper):
- **First-attempt success**: g_test passed SyntaxGuard on first attempt
- **Validated success**: Both g_test and g_impl succeeded

**Expected output:**
- ~100-120 training samples extracted
- Train/val split saved to `data/prepared/`
- Statistics printed showing distribution across tasks

#### Step 2: Fine-Tune Model

Fine-tune the model using LoRA:

```bash
python -m src.training.finetune \
  --train-file data/prepared/train.jsonl \
  --val-file data/prepared/val.jsonl \
  --output-dir models/finetuned
```

**LoRA configuration:**
- r=16, alpha=32, dropout=0.05
- Learning rate: 2e-4
- Epochs: 3
- Batch size: 4
- 4-bit quantization for memory efficiency

**Expected output:**
- Fine-tuned model saved to `models/finetuned/`
- Training takes ~2-4 hours on GPU

#### Step 3: Convert to Ollama

Follow the instructions printed by the fine-tuning script to convert the model to Ollama format and create the `qwen2.5-coder:7b-tdd-finetuned` model.

### Phase 3: Evaluation

#### Step 1: Run Fine-Tuned Benchmark

```bash
python run_benchmark.py --model finetuned --trials 50
```

This runs the same benchmark with the fine-tuned model.

#### Step 2: Compare Results

```bash
python compare_results.py --output results/analysis/comparison_report.json
```

**What this does:**
- Loads baseline and fine-tuned summaries
- Performs statistical comparison using Fisher's exact test
- Calculates Cohen's h effect sizes
- Evaluates experimental hypotheses
- Prints detailed comparison report

**Expected output:**

```
STATISTICAL COMPARISON REPORT
======================================================================
OVERALL RESULTS:
  Baseline:    57.0% (171/300 trials)
  Fine-tuned:  64.0% (192/300 trials)
  Improvement: +7.0 percentage points
  Effect size: 0.289 (small)
  P-value:     0.0234 **
  → Statistically significant at α=0.05

HYPOTHESIS EVALUATION
======================================================================
H1: Fine-tuned 7B improves by ≥5pp over baseline 7B
  Result: ✓ MET
  Improvement: 7.0pp (threshold: 5.0pp)
  Significant: True

H2: Fine-tuned 7B matches 14B baseline (70%) on ≥3 tasks
  Result: ✓ MET
  Tasks matching: 3/6

OVERALL: ✓ EXPERIMENT SUCCESS
```

## Hypotheses

### H1: Measurable Improvement
**Fine-tuned 7B improves by ≥5 percentage points over baseline 7B**

- Baseline 7B: ~57% (from paper)
- Fine-tuned 7B: ≥62% (predicted)
- Statistical significance: p < 0.05 (Fisher's exact test)

### H2: Close Gap with Larger Model
**Fine-tuned 7B matches 14B baseline performance on ≥3 tasks**

- 14B baseline: ~70% (from paper)
- Fine-tuned 7B achieves ≥70% on at least 3 of 6 tasks

**Success criteria**: Either H1 OR H2 must be met.

## Key Findings from Paper

From main.tex Section 4.3 (lines 950-956):

> "The workflow success rate (43.1%) is substantially lower than individual action success rates (g_test: 60.2%, g_impl: 71.5%), highlighting error propagation in sequential workflows. Critically, 23% of workflow failures stemmed from LLM-generated tests with incorrect edge case expectations—the implementation satisfied the specification but failed the test."

**Implication for our experiment**: Fine-tuning on successful first-attempt tests should reduce specification errors in test generation, improving downstream implementation success.

## Training Data Quality Standards

We define quality criteria for training samples (NEW, not from paper):

| Quality Tier | Definition | Selection Criteria |
|--------------|------------|-------------------|
| **First-Attempt Success** | g_test passed SyntaxGuard on first attempt | `attempt == 1 AND guard_passed == True` |
| **Validated Success** | Tests passed AND subsequent g_impl succeeded | `g_test.passed AND g_impl.passed` |

**Standard we use**: First-Attempt Success + Validated Success

This ensures training samples are both syntactically correct AND lead to successful downstream implementation.

## Files Generated

### Benchmark Results

Each benchmark run creates:
- `summary.json`: Aggregate statistics
- `<task_id>_trials.jsonl`: Individual trial results (one JSON object per line)

### Training Data

- `training_samples.jsonl`: Extracted training samples
- `data/prepared/train.jsonl`: Training set (90%)
- `data/prepared/val.jsonl`: Validation set (10%)
- `data/prepared/chat_format/`: Chat-formatted datasets

### Analysis

- `comparison_report.json`: Statistical comparison results
- Console output with detailed tables and hypothesis evaluation

## Customization

### Run with Different Model

```bash
python run_benchmark.py --model custom --model-name qwen2.5-coder:14b --trials 50
```

### Change Retry Budget

```bash
python run_benchmark.py --model baseline --trials 50 --retry-budget 5
```

### Adjust Quality Standards

```bash
# Extract any validated success (not just first-attempt)
python extract_training_data.py --no-first-attempt
```

### Modify LoRA Parameters

Edit `src/config.py`:

```python
TRAINING_CONFIG = {
    "lora_r": 32,           # Increase rank
    "lora_alpha": 64,       # Increase alpha proportionally
    "learning_rate": 1e-4,  # Lower learning rate
    "num_epochs": 5,        # More epochs
}
```

## Troubleshooting

### Ollama Connection Error

Ensure Ollama is running:
```bash
ollama serve
```

Verify model is available:
```bash
ollama list
```

### Out of Memory (Fine-Tuning)

The fine-tuning script uses 4-bit quantization by default. If you still run out of memory:

1. Reduce batch size in `src/config.py`
2. Use gradient accumulation (already enabled)
3. Fine-tune on a machine with more VRAM

### Pytest Not Found

```bash
pip install pytest pytest-cov
```

### Low Training Sample Count

If you get fewer than 100 training samples:

1. Check baseline success rate - it should be ~57%
2. Try relaxing quality standards with `--no-first-attempt`
3. Run more trials (e.g., `--trials 100`)

## References

- **AtomicGuard Paper**: docs/main.tex, Section 4.3 (TDD Workflow Benchmark)
- **LoRA Paper**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
- **Fisher's Exact Test**: Statistical method for comparing proportions
- **Cohen's h**: Effect size measure for proportions

## Contributing

This is an experimental research implementation. Contributions welcome:

1. Extend to more tasks
2. Try different base models
3. Experiment with quality standards
4. Add multi-agent workflows (BDD, DDD)

## License

Part of the AtomicGuard project. See repository root for license.
