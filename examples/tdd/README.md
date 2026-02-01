# TDD Workflow Examples

This directory contains examples demonstrating Test-Driven Development (TDD) workflows using AtomicGuard's guard composition patterns.

## Progression

| Level | Directory | Guards | Description |
|-------|-----------|--------|-------------|
| 1 | [01_human_review/](01_human_review/) | SyntaxGuard + HumanReviewGuard | Human validates tests before implementation |
| 2 | [02_import_guard/](02_import_guard/) | SyntaxGuard + ImportGuard + HumanReviewGuard | Automated import validation + human review |
| 3 | [03_huggingface/](03_huggingface/) | SyntaxGuard + ImportGuard / DynamicTestGuard | Fully automated TDD with HuggingFace |

## 01_human_review - Human-in-the-Loop TDD

Demonstrates the core TDD workflow where a human reviewer validates LLM-generated tests before implementation begins.

**Guard chain**: `SyntaxGuard → HumanReviewGuard`

Key concepts:

- Human approval/rejection with feedback
- LLM self-correction from rejection feedback
- Two-step workflow: tests → implementation

```bash
cd 01_human_review
python run.py --host http://localhost:11434
```

## 02_import_guard - Defense-in-Depth TDD

Adds automated import validation before human review, catching common LLM mistakes (like missing `import pytest`) automatically.

**Guard chain**: `SyntaxGuard → ImportGuard → HumanReviewGuard`

Key concepts:

- Fail-fast validation before human sees the code
- Pure AST analysis (no subprocess)
- LLM self-correction via guard feedback

```bash
cd 02_import_guard
python run.py --host http://localhost:11434
```

## 03_huggingface - Fully Automated TDD

Removes the human review step entirely. Guards handle all validation automatically using HuggingFace Inference API as the LLM backend.

**Guard chain (g_test)**: `SyntaxGuard → ImportGuard`
**Guard (g_impl)**: `DynamicTestGuard`

Key concepts:

- Fully automated end-to-end TDD (no human intervention)
- HuggingFace Inference API via `GeneratorRegistry`
- `FilesystemArtifactDAG` for persistent artifact storage
- Configuration-driven workflow (`workflow.json` + `prompts.json`)

```bash
export HF_TOKEN="hf_your_token_here"
uv run python -m examples.tdd.03_huggingface.run
```

## Progression

**01_human_review** shows what happens when humans must catch all errors - including import mistakes that slip through syntax validation.

**02_import_guard** demonstrates how automated guards can reduce human cognitive load by catching mechanical errors automatically, so humans can focus on semantic validation (test quality, coverage, logic).

**03_huggingface** takes this further by removing the human entirely — if your guard chain is strong enough, the workflow runs fully unattended.

## Prerequisites

### Ollama (examples 01, 02)

```bash
ollama serve
ollama pull qwen2.5-coder:14b
```

### HuggingFace (example 03)

```bash
uv pip install huggingface_hub
export HF_TOKEN="hf_your_token_here"
```

## Next Steps

After understanding TDD workflows:

- **checkpoint/** - Add checkpoint/resume for human-in-the-loop intervention
- **gui/** - Visual workflow monitoring tools
