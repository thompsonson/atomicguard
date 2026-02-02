# SWE-Bench Pro Example

Multi-language bug-fix evaluation using [SWE-Bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) (ScaleAI). Runs AtomicGuard workflow arms across 731 instances in Python, Go, JavaScript, and TypeScript, then evaluates patches using the official Docker-based harness.

## Architecture

```
dataset.py         Load & filter instances from HuggingFace
language.py        Per-language config (syntax checkers, test patterns, code fences)
generators/        Language-aware prompt builders (subclass ablation generators)
guards/            Language-aware validation (syntax + test pattern checks)
experiment_runner  Orchestrates workflow arms across instances (sequential or parallel)
evaluation.py      Wraps official scaleapi/SWE-bench_Pro-os eval script
demo.py            Click CLI tying it all together
```

For Python instances, the standard ablation generators and guards are used directly. For Go, JavaScript, and TypeScript, `MultiLangPatchGenerator`, `MultiLangTestGenerator`, and `MultiLangTestSyntaxGuard` inject language-appropriate code fences, framework instructions, and syntax checks.

## Prerequisites

```bash
# Install the project with experiment and examples dependency groups
uv sync --group examples --extra experiment

# Set your HuggingFace token (for model inference)
export HF_TOKEN="hf_your_token_here"

# Docker is required for evaluation (the eval harness runs tests in containers)
docker --version
```

## Usage

### Run experiments

```bash
# Single arm, 2 instances, Python only (good for a first trial)
uv run python -m examples.swe_bench_pro.demo --debug experiment \
    --arms singleshot \
    --language python \
    --max-instances 2

# All arms, all languages, parallel execution
uv run python -m examples.swe_bench_pro.demo experiment \
    --arms singleshot,s1_direct,s1_tdd \
    --max-workers 4

# Resume a previous run
uv run python -m examples.swe_bench_pro.demo experiment \
    --arms singleshot \
    --resume

# Custom model
uv run python -m examples.swe_bench_pro.demo experiment \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --arms singleshot
```

> **Note:** `--debug` is a top-level flag and must appear *before* the subcommand name.

### Evaluate patches

```bash
# Evaluate predictions from a previous experiment run
uv run python -m examples.swe_bench_pro.demo evaluate \
    --predictions-dir output/swe_bench_pro/predictions

# Use a pre-cloned eval repo
uv run python -m examples.swe_bench_pro.demo evaluate \
    --predictions-dir output/swe_bench_pro/predictions \
    --eval-repo /path/to/SWE-bench_Pro-os
```

### Visualize results

```bash
uv run python -m examples.swe_bench_pro.demo visualize \
    --results output/swe_bench_pro/results.jsonl \
    --resolved output/swe_bench_pro/predictions/eval_output/eval_results.json
```

### List dataset instances

```bash
uv run python -m examples.swe_bench_pro.demo list-instances
```

## Output

```
output/swe_bench_pro/
├── results.jsonl              # One JSON line per (instance, arm) run
├── artifact_dags/             # Per-instance workflow artifacts
│   └── <instance_id>/
│       └── <arm>/
├── predictions/               # Formatted for the eval harness
│   ├── 02_singleshot.json
│   ├── 03_s1_direct.json
│   └── eval_output/           # Eval harness results
│       └── eval_results.json
└── eval_logs/                 # Per-instance evaluation logs
    └── <run_id>/
        ├── <instance_id>.log
        └── summary.log
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `experiment` | Run workflow arms across SWE-Bench Pro instances |
| `evaluate` | Run Docker-based evaluation on generated patches |
| `visualize` | Generate charts from experiment results |
| `list-instances` | Show dataset statistics by language and repo |

### experiment options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-Coder-32B-Instruct` | HuggingFace model ID |
| `--arms` | `singleshot,s1_direct,s1_tdd` | Comma-separated arm names |
| `--language` | all | Filter: `python`, `go`, `javascript`, `typescript` |
| `--output-dir` | `output/swe_bench_pro` | Results directory |
| `--split` | `test` | Dataset split |
| `--max-instances` | 0 (all) | Cap on instances |
| `--max-workers` | 1 | Parallel workers (1 = sequential) |
| `--resume` | off | Resume from existing results |

## Relationship to swe_bench_ablation

This example reuses several components from `examples/swe_bench_ablation`:

- **Workflow configs**: JSON files in `swe_bench_ablation/workflows/` define the action pair structure
- **Base generators**: `PatchGenerator`, `TestGenerator`, `AnalysisGenerator`, `LocalizationGenerator`
- **Base guards**: `PatchGuard`, `TestSyntaxGuard`, `AnalysisGuard`, `LocalizationGuard`
- **Data structures**: `ArmResult`, `EvalResult`
- **Analysis/visualization**: `generate_visualizations`, `load_results`

The multi-language subclasses (`MultiLang*`) override `_build_prompt` to inject language-appropriate text while preserving the same workflow structure.

## Configuration

- [`prompts.json`](prompts.json) -- Language-neutral prompt templates (uses "VALID CODE" instead of "VALID PYTHON")
- [`logging.json`](logging.json) -- Third-party logger suppression list. Edit `suppress_loggers` to control which libraries are silenced at WARNING level during `--debug` runs
- Workflow configs are loaded from `examples/swe_bench_ablation/workflows/`
