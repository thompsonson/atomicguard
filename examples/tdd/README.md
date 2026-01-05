# TDD Workflow Examples

This directory contains examples demonstrating Test-Driven Development (TDD) workflows using AtomicGuard's guard composition patterns.

## Progression

| Level | Directory | Guards | Description |
|-------|-----------|--------|-------------|
| 1 | [01_human_review/](01_human_review/) | SyntaxGuard + HumanReviewGuard | Human validates tests before implementation |
| 2 | [02_import_guard/](02_import_guard/) | SyntaxGuard + ImportGuard + HumanReviewGuard | Automated import validation + human review |

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

## Why Two Examples?

**01_human_review** shows what happens when humans must catch all errors - including import mistakes that slip through syntax validation.

**02_import_guard** demonstrates how automated guards can reduce human cognitive load by catching mechanical errors automatically, so humans can focus on semantic validation (test quality, coverage, logic).

## Prerequisites

- Ollama running with `qwen2.5-coder:14b` model (or configure different LLM)

```bash
ollama serve
ollama pull qwen2.5-coder:14b
```

## Next Steps

After understanding TDD workflows:

- **checkpoint/** - Add checkpoint/resume for human-in-the-loop intervention
- **gui/** - Visual workflow monitoring tools
