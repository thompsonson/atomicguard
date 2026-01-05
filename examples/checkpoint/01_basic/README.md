# Checkpoint/Resume Demo

Interactive demo of human-in-the-loop workflow resumption.

## Quick Start

```bash
# 1. Run workflow (fails, creates checkpoint)
uv run python examples/checkpoint/demo.py run

# 2. Edit the artifact file (change "a - b" to "a + b")
# File path shown in output

# 3. Resume workflow
uv run python examples/checkpoint/demo.py resume <checkpoint_id>
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Execute workflow, create checkpoint on failure |
| `resume <id>` | Resume from checkpoint with edited artifact |
| `list` | List all checkpoints |
| `clean` | Remove output directory |

## Options

- `run --verbose`: Show full context inline
- `resume --artifact <path>`: Use custom artifact file

## Output Files

On checkpoint, creates `output/` with:

- `instructions.md` - What to do next
- `context.md` - Specification and feedback history
- `artifacts/{step}.py` - File to edit
