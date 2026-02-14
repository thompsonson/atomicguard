# Claude Code Instructions

## Project Overview

AtomicGuard is a guard-railed workflow framework for LLM code generation, evaluated on SWE-Bench Pro. Generators produce code, guards validate it, and workflows chain them with retry/backtracking logic.

## Prerequisites

```bash
uv sync                # install all dependencies
source .env            # load API keys (always do this first)
```

Docker is required for SWE-Bench evaluation guards (TestRedGuard, TestGreenGuard, FullEvalGuard).

## Running Python

Always use `uv run python`. Never bare `python`.

```bash
uv run python -m examples.swe_bench_pro.demo experiment ...
uv run python -m examples.dashboard ...
```

## Environment Variables

Create a `.env` file at the project root:

```bash
export LLM_API_KEY="sk-or-v1-..."      # OpenRouter API key
export HF_TOKEN="hf_..."                # HuggingFace token
export OPENAI_API_KEY="sk-proj-..."     # OpenAI API key
```

Always `source .env` before running anything.

## Running Experiments

Quick start:

```bash
source .env && uv run python -m examples.swe_bench_pro.demo experiment \
  --model moonshotai/kimi-k2-0905 \
  --provider openrouter \
  --arms s1_tdd_rule_backtrack \
  --max-instances 5
```

Common flags:

| Flag | Description |
|------|-------------|
| `--model` | Model ID (e.g. `moonshotai/kimi-k2-0905`) |
| `--provider` | `ollama`, `openrouter`, `huggingface`, `openai` |
| `--arms` | Comma-separated workflow variants |
| `--language` | Filter by language: `python`, `go`, `javascript`, `typescript` |
| `--output-dir` | Output path (default: `output/swe_bench_pro`) |
| `--max-instances` | Limit number of instances |
| `--instances` | Comma-separated instance ID substrings |
| `--max-workers` | Parallel workers (default: 1) |
| `--resume` | Resume a previous run |
| `--evaluate` | Run Docker-based evaluation after generation |

Available arms: `singleshot`, `s1_direct`, `s1_tdd`, `s1_tdd_verified`, `s1_tdd_behavior`, `s1_decomposed`, `s1_tdd_review`, `s1_tdd_review_backtrack`, `s1_tdd_rule_backtrack`, `adaptive_backtrack`.

Other subcommands:

```bash
uv run python -m examples.swe_bench_pro.demo evaluate --predictions-dir output/...
uv run python -m examples.swe_bench_pro.demo list-instances --split test
uv run python -m examples.swe_bench_pro.demo analyze-errors --results output/.../results.jsonl
```

## Reviewing Results (Dashboard)

```bash
uv run python -m examples.dashboard --output-dir output/
```

Opens a read-only FastAPI dashboard at `http://0.0.0.0:8000`. The DAG viewer shows per-instance artifact graphs to review workflow progress, guard feedback, and retries.

Flags: `--host`, `--port`, `--workflows-dir`, `--prompts`.

## Experiment Notes

Place a `NOTES.md` in the experiment output directory:

```
output/{experiment_name}/NOTES.md
```

This file is rendered as HTML on the dashboard experiment detail page. Use it to summarise findings, failures, and mitigations.

## Output Directory Layout

```
output/
└── {experiment_name}/
    ├── {model}/
    │   └── artifact_dags/
    │       └── {instance_id}/
    │           └── {arm}/
    ├── predictions/
    ├── results.jsonl        # per-instance results (ArmResult)
    ├── summary.json         # aggregate stats by arm
    └── NOTES.md             # experiment notes (shown on dashboard)
```

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) messages (e.g. `feat:`, `fix:`, `docs:`). A pre-commit hook validates formatting.

## Testing & Code Quality

All commands via the `justfile`:

| Command | Description |
|---------|-------------|
| `just test` | Unit tests (domain + application + infrastructure) |
| `just test-all` | All tests (unit + architecture) |
| `just coverage` | Unit tests with coverage report |
| `just smoke` | Zero-dependency smoke tests |
| `just lint` | Lint check with ruff |
| `just fmt` | Auto-format with ruff |
| `just typecheck` | Type check with mypy |
| `just ci` | Full CI pipeline (lint + fmt-check + typecheck + test-all + smoke) |

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/atomicguard/` | Core framework (domain models, guards, generators, workflows) |
| `examples/swe_bench_pro/` | SWE-Bench Pro experiment runner and demo CLI |
| `examples/swe_bench_common/` | Shared guards and workflows for SWE-Bench |
| `examples/dashboard/` | FastAPI dashboard for browsing results |
| `examples/basics/` | Introductory examples (mock, ollama, multiagent, backtracking) |
