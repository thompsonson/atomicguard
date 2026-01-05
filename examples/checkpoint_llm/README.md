# Checkpoint LLM Demo (Level 3)

LLM-powered workflow demonstrating checkpoint/resume with real AI generation.

## What This Teaches

- Using OllamaGenerator with checkpoint/resume
- Guard configuration via `guard_config.test_file`
- Real LLM generation that may pass or fail
- Human intervention when AI gets it wrong

## Task: Template Engine

Write a `render_template(template: str, context: dict) -> str` function that:

1. Replaces `{{ variable }}` with values from context
2. Supports `{% if variable %}text{% endif %}` conditionals
3. Handles missing keys (leave unchanged or remove block)

This task is more challenging than simple string manipulation, requiring regex patterns.

## Prerequisites

```bash
# Start Ollama
ollama serve

# Pull the model
ollama pull qwen2.5-coder:7b
```

## Quick Start

```bash
# 1. Run workflow (LLM generates code, guard runs tests)
uv run python examples/checkpoint_llm/demo.py run --host http://localhost:11434

# 2. If checkpoint created, edit the artifact:
#    File: examples/checkpoint_llm/output/artifacts/g_template.py

# 3. Resume workflow
uv run python examples/checkpoint_llm/demo.py resume <checkpoint_id>
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Execute workflow, validate with tests |
| `resume <id>` | Resume from checkpoint with edited artifact |
| `list` | List all checkpoints |
| `clean` | Remove output directory |
| `show-config` | Display workflow.json, prompts.json, and test file |

## Options

```bash
uv run python examples/checkpoint_llm/demo.py run \
  --host http://localhost:11434 \
  --model qwen2.5-coder:7b \
  --verbose
```

## How It Works

1. **OllamaGenerator** generates `render_template` function from specification
2. **DynamicTestGuard** runs pytest tests from `test_template.py`
3. If tests fail after `rmax` attempts, checkpoint is created
4. Human edits the generated code
5. Resume validates the human-edited artifact

## Key Difference from Level 2

| Aspect | Level 2 (`checkpoint_tdd/`) | Level 3 (`checkpoint_llm/`) |
|--------|----------------------------|----------------------------|
| Generator | MockImplGenerator | OllamaGenerator (real LLM) |
| Requires LLM | No | Yes (Ollama) |
| Outcome | Always fails (deterministic) | May pass or fail |
| Steps | 2 (g_test → g_impl) | 1 (g_template) |
| Test source | Dependency artifact | guard_config.test_file |

## Complexity Levels

| Level | Example | Generator | Config | Steps |
|-------|---------|-----------|--------|-------|
| 1 | `checkpoint/` | Mock | Hardcoded | 1 |
| 2 | `checkpoint_tdd/` | Mock | workflow.json | 2 |
| **3** | **`checkpoint_llm/`** | **LLM** | **workflow.json** | **1** |
| 4+ | `sdlc/`, `add/` | LLM | Full config | N |
