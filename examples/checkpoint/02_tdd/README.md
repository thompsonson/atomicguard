# Checkpoint TDD Demo (Level 2)

Config-driven TDD workflow demonstrating checkpoint/resume with `workflow.json` and `prompts.json`.

## What This Teaches

- How to define workflows in `workflow.json`
- How to define prompt templates in `prompts.json`
- How generators are registered and referenced by name
- Multi-step workflows with dependencies (g_test → g_impl)
- Partial success: checkpoint captures completed steps

## Quick Start

```bash
# 1. Run workflow (g_test passes, g_impl fails)
uv run python examples/checkpoint_tdd/demo.py run

# 2. Edit the artifact file (fix the bugs - comments show what to change)
# File: examples/checkpoint_tdd/output/artifacts/g_impl.py

# 3. Resume workflow
uv run python examples/checkpoint_tdd/demo.py resume <checkpoint_id>
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Execute workflow, create checkpoint on failure |
| `resume <id>` | Resume from checkpoint with edited artifact |
| `list` | List all checkpoints |
| `clean` | Remove output directory |
| `show-config` | Display workflow.json and prompts.json |

## Configuration Files

### workflow.json

Defines the workflow structure:

```json
{
  "name": "checkpoint-tdd-demo",
  "specification": "Create a Stack class...",
  "rmax": 2,
  "action_pairs": {
    "g_test": {
      "generator": "MockTestGenerator",
      "guard": "syntax"
    },
    "g_impl": {
      "generator": "MockImplGenerator",
      "guard": "dynamic_test",
      "requires": ["g_test"]
    }
  }
}
```

### prompts.json

Defines prompt templates for each step (used for context display):

```json
{
  "g_test": {
    "role": "Test Engineer practicing TDD",
    "task": "Write unit tests for Stack",
    "constraints": "Use pytest-style...",
    "feedback_wrapper": "GUARD REJECTION:\n{feedback}"
  }
}
```

## Generator Registration

Mock generators are registered so workflow.json can reference them:

```python
# In demo.py
GeneratorRegistry.register("MockTestGenerator", MockTestGenerator)
GeneratorRegistry.register("MockImplGenerator", MockImplGenerator)
```

In real workflows, use built-in generators:

```json
"generator": "OllamaGenerator",
"generator_config": {"model": "qwen2.5-coder:7b"}
```

## The Bug to Fix

The mock implementation has two LIFO vs FIFO bugs with comments:

```python
def pop(self):
    # BUG: Change self._items.pop(0) to self._items.pop()
    return self._items.pop(0)

def peek(self):
    # BUG: Change self._items[0] to self._items[-1]
    return self._items[0]
```

## Complexity Levels

| Level | Example | Generator | Config | Steps |
|-------|---------|-----------|--------|-------|
| 1 | `checkpoint/` | Mock | Hardcoded | 1 |
| **2** | **`checkpoint_tdd/`** | **Mock** | **workflow.json** | **2** |
| 3+ | `sdlc/`, `add/` | LLM | Full config | N |
