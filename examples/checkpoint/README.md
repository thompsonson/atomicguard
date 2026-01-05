# Checkpoint Examples

This directory contains a progression of checkpoint/resume workflow examples, demonstrating human-in-the-loop intervention patterns.

## Progression

| Level | Directory | Generator | Config | Description |
|-------|-----------|-----------|--------|-------------|
| 1 | [01_basic/](01_basic/) | Mock (always fails) | Hardcoded | Simple checkpoint/resume mechanics |
| 2 | [02_tdd/](02_tdd/) | Mock (partial success) | workflow.json + prompts.json | Config-driven multi-step with dependencies |
| 3 | [03_llm/](03_llm/) | Real LLM (Ollama) | workflow.json + prompts.json | Real AI generation with checkpoint recovery |
| 4 | [04_sdlc/](04_sdlc/) | Custom multi-agent | workflow.json + prompts.json | Full SDLC pipeline with 4 specialized agents |

## Learning Path

### Level 1: Basic Mechanics

Start here to understand the fundamental checkpoint/resume flow:

- Generator intentionally fails validation
- Checkpoint created after `rmax` exhausted
- Human edits artifact file directly
- Resume re-validates the amended artifact

```bash
cd 01_basic
python demo.py run      # Creates checkpoint
# Edit output/artifacts/g_correct_add.py
python demo.py resume   # Validates fix
```

### Level 2: Config-Driven Workflows

Learn how to define workflows declaratively:

- Steps defined in `workflow.json`
- Prompts templated in `prompts.json`
- Multi-step workflow with dependencies (`g_test` → `g_impl`)
- Partial success: completed steps preserved in checkpoint

```bash
cd 02_tdd
python demo.py run      # Step 1 passes, Step 2 fails
# Edit output/artifacts/g_impl.py
python demo.py resume   # Only re-runs failed step
```

### Level 3: Real LLM Generation

Apply checkpoint patterns to real AI workflows:

- Uses `OllamaGenerator` with actual LLM
- Non-deterministic outcomes (LLM may pass or fail)
- `DynamicTestGuard` runs pytest for validation
- Real-world scenario: LLM-generated code + human refinement

```bash
cd 03_llm
python demo.py run      # LLM generates code
# If checkpoint created, edit output/artifacts/g_template.py
python demo.py resume   # Re-validates
```

### Level 4: Multi-Agent SDLC Pipeline

Production-grade multi-agent workflow:

- 4 specialized generators: ConfigExtractor, ADD, BDD, Coder
- Complex dependency graph with parallel execution support
- Custom guards per domain (architecture, BDD, tests)
- Hierarchical context composition (global Ω + local ℛ)

```bash
cd 04_sdlc
python demo.py run --docs ./sample_input/
# If checkpoint created, edit failed artifact
python demo.py resume
```

## Key Concepts

### Checkpoint Triggers

- `rmax` exhaustion (generator failed too many times)
- Guard escalation (`fatal=True`)
- Explicit checkpoint request

### Human Amendment Types

- **ARTIFACT**: Replace failed artifact with human-edited version
- **FEEDBACK**: Provide additional guidance for retry
- **SKIP**: Skip failed step (mark as externally satisfied)

### Output Structure

Each example generates:

```
output/
├── instructions.md     # What to do next
├── context.md          # Specification + feedback history
├── artifacts/          # Failed artifacts for editing
└── checkpoints/        # Checkpoint state persistence
```

## Prerequisites

- **Level 1-2**: No external dependencies (mock generators)
- **Level 3-4**: Requires Ollama with `qwen2.5-coder:7b` model (or configure different LLM)
