# TDD Stack - Fully Automated (HuggingFace)

Fully automated TDD workflow using HuggingFace Inference API. No human review required — guards handle all validation automatically.

## Workflow

```
g_test: Generate tests → SyntaxGuard + ImportGuard → Pass/Fail
                                                        ↓ (if pass)
g_impl: Generate implementation → DynamicTestGuard → Pass/Fail
            ↑                          |
            └──────────────────────────┘ (retry with feedback)
```

**Step 1 (g_test)**: LLM generates pytest-style tests. `SyntaxGuard` validates Python syntax, `ImportGuard` catches undefined names — no human needed.

**Step 2 (g_impl)**: LLM generates implementation. `DynamicTestGuard` runs the tests from step 1 against the implementation in a subprocess.

## Prerequisites

```bash
# Install the optional dependency
uv pip install huggingface_hub

# Set your API token
export HF_TOKEN="hf_your_token_here"
```

Get a token at: <https://huggingface.co/settings/tokens>

## Usage

```bash
# Run with defaults (Qwen/Qwen2.5-Coder-32B-Instruct)
uv run python -m examples.tdd.03_huggingface.run

# Override model
uv run python -m examples.tdd.03_huggingface.run --model Qwen/Qwen2.5-Coder-32B-Instruct

# Verbose logging
uv run python -m examples.tdd.03_huggingface.run -v
```

## Output

Results are stored in `output/`:

```
output/
├── artifacts/          # FilesystemArtifactDAG storage
│   ├── objects/        # Individual artifact JSON files
│   └── index.json      # Artifact index
├── results.json        # Workflow results summary
└── results.log         # Execution log
```

Each artifact in `objects/` contains the full provenance chain — generated code, guard results, feedback history, and context snapshots.

## How It Differs from Other TDD Examples

| Example | Test Guards | Human Review? | LLM Backend |
|---------|-----------|---------------|-------------|
| 01_human_review | Syntax + Human | Yes | Ollama |
| 02_import_guard | Syntax + Import + Human | Yes | Ollama |
| **03_huggingface** | **Syntax + Import** | **No** | **HuggingFace** |

This is the first fully end-to-end automated TDD example — the entire generate → validate → retry loop runs without human intervention.

## Configuration

- [`workflow.json`](workflow.json) — Workflow structure, guard chain, and model
- [`prompts.json`](prompts.json) — Role, constraints, and task for each step
