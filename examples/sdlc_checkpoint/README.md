# SDLC Checkpoint Example

Multi-agent SDLC workflow with checkpoint/resume support. This example demonstrates:

1. **Multi-step workflow** with dependent action pairs
2. **Checkpoint/resume** for human intervention when steps fail
3. **Self-contained generators and guards** (no external dependencies)

## Workflow Structure

```
g_config ─────────────┬──────────────────────────┐
                      │                          │
                      v                          v
                  g_add (pytest-arch)      g_bdd (Gherkin)
                      │                          │
                      └──────────┬───────────────┘
                                 │
                                 v
                             g_coder
                                 │
                          ┌──────┴──────┐
                          │  CHECKPOINT │  (on failure)
                          │   /RESUME   │
                          └─────────────┘
```

## Action Pairs

| Step | Generator | Guard | Description |
|------|-----------|-------|-------------|
| g_config | ConfigExtractorGenerator | ConfigGuard | Extract project configuration |
| g_add | ADDGenerator | ArchitectureTestsGuard | Generate pytest-arch tests |
| g_bdd | BDDGenerator | BDDGuard | Generate Gherkin scenarios |
| g_coder | CoderGenerator | AllTestsPassGuard | Generate implementation |

## Prerequisites

```bash
# Start Ollama
ollama serve

# Pull model
ollama pull qwen2.5-coder:14b
```

## Usage

### Run the workflow

```bash
uv run python -m examples.sdlc_checkpoint.demo run --host http://localhost:11434
```

### If checkpoint is created (step fails)

1. Review the instructions:

   ```bash
   cat examples/sdlc_checkpoint/output/instructions.md
   ```

2. Edit the artifact file:

   ```bash
   # Edit the file for the failed step
   vim examples/sdlc_checkpoint/output/artifacts/g_config.json
   ```

3. Resume the workflow:

   ```bash
   uv run python -m examples.sdlc_checkpoint.demo resume <checkpoint_id>
   ```

### Other commands

```bash
# List checkpoints
uv run python -m examples.sdlc_checkpoint.demo list

# Clean output
uv run python -m examples.sdlc_checkpoint.demo clean

# Show configuration
uv run python -m examples.sdlc_checkpoint.demo show-config
```

## File Structure

```
examples/sdlc_checkpoint/
├── __init__.py
├── README.md
├── demo.py                    # CLI: run, resume, list, clean
├── workflow.json              # Workflow configuration
├── prompts.json               # Prompt templates
├── models.py                  # Pydantic models
│
├── generators/
│   ├── __init__.py
│   ├── config.py              # ConfigExtractorGenerator
│   ├── add.py                 # ADDGenerator
│   ├── bdd.py                 # BDDGenerator
│   └── coder.py               # CoderGenerator
│
├── guards/
│   ├── __init__.py
│   ├── config_guard.py        # ConfigGuard
│   ├── architecture_guard.py  # ArchitectureTestsGuard
│   ├── bdd_guard.py           # BDDGuard
│   └── tests_guard.py         # AllTestsPassGuard
│
├── sample_input/
│   ├── architecture.md        # Architecture documentation
│   └── requirements.md        # Requirements documentation
│
└── output/                    # Created at runtime
    ├── artifact_dag/          # All artifacts
    ├── checkpoints/           # Checkpoint storage
    ├── artifacts/             # Human-editable artifacts
    ├── instructions.md        # What to do next
    └── context.md             # Full context
```

## How Checkpoints Work

1. When a step fails after exhausting retries, a checkpoint is created
2. The checkpoint contains:
   - Failed step identifier
   - All completed artifacts
   - Failure feedback from the guard
3. Human-readable files are written to `output/`:
   - `instructions.md` - What happened and how to fix it
   - `context.md` - Full specification and constraints
   - `artifacts/{step}.json` - The failed artifact to edit
4. After editing, `resume` validates the human-provided artifact
5. If valid, workflow continues from that point

## Key Features

### Self-Contained

All generators and guards are defined within this example folder:

- No imports from other examples
- Easy to understand and modify
- Good template for creating new examples

### Checkpoint/Resume

Uses `ResumableWorkflow` from atomicguard core:

- Automatic checkpoint on failure
- Human-in-the-loop editing
- Resume from any failed step

### Multi-Agent

The workflow orchestrates multiple specialized agents:

- Config extractor for project setup
- ADD for architecture tests
- BDD for behavior scenarios
- Coder for implementation
