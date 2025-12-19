# TDD Workflow with Human Review Guard

This example demonstrates a Test-Driven Development (TDD) workflow using AtomicGuard's dual-state agent framework. A human reviewer validates LLM-generated tests before implementation begins.

## Overview

The workflow follows TDD principles:

1. **Generate Tests First** - An LLM creates pytest-style unit tests for a Stack data structure
2. **Human Review** - A human validates the tests before they're accepted
3. **Generate Implementation** - A second LLM implements code to pass the tests
4. **Automated Validation** - DynamicTestGuard runs pytest against the implementation

### Key Concepts

| Concept | Description |
|---------|-------------|
| **ActionPair** | Couples a generator (LLM) with a guard (validator) as an atomic unit |
| **CompositeGuard** | Chains multiple guards; all must pass for artifact acceptance |
| **HumanReviewGuard** | Blocks workflow for human CLI approval/rejection |
| **DynamicTestGuard** | Runs pytest in isolated subprocess against implementation |
| **Artifact** | The output of a generator (e.g., test code, implementation code) |
| **Provenance** | History of rejected attempts with feedback |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TDD Workflow                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: g_test (Generate Tests)                               │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐       │
│  │ Ollama   │───▶│ SyntaxGuard │───▶│ HumanReviewGuard │──┐    │
│  │ Generator│    │ (AST check) │    │ (CLI approval)   │  │    │
│  └──────────┘    └─────────────┘    └──────────────────┘  │    │
│        ▲                                                  │    │
│        │              Rejection feedback                  │    │
│        └──────────────────────────────────────────────────┘    │
│                                                  │              │
│                                                  ▼              │
│                                           ┌──────────┐         │
│                                           │ Test     │         │
│                                           │ Artifact │         │
│                                           └──────────┘         │
│                                                  │              │
│  Step 2: g_impl (Generate Implementation)        │              │
│  ┌──────────┐    ┌──────────────────┐           │              │
│  │ Ollama   │───▶│ DynamicTestGuard │◀──────────┘              │
│  │ Generator│    │ (runs pytest)    │                          │
│  └──────────┘    └──────────────────┘                          │
│        ▲                    │                                   │
│        │    Test failures   │                                   │
│        └────────────────────┘                                   │
│                             │                                   │
│                             ▼                                   │
│                      ┌──────────────┐                          │
│                      │ Implementation│                          │
│                      │ Artifact      │                          │
│                      └──────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Ollama** running with a coding model:

   ```bash
   ollama serve
   ollama pull qwen2.5-coder:14b  # Recommended
   ```

2. **AtomicGuard** installed:

   ```bash
   pip install atomicguard
   # Or from source:
   pip install -e .
   ```

## Quick Start

```bash
cd examples/tdd_human_review
python run.py --host http://localhost:11434
```

### Expected Interaction

1. The LLM generates test code for a Stack class
2. You see the generated tests and are prompted:

   ```
   ============================================================
   REVIEW GENERATED TESTS
   ============================================================
   [Generated test code appears here]

   Approve this artifact? [y/n]:
   ```

3. Type `y` to approve or `n` to reject (with feedback)
4. If approved, implementation generation begins
5. Tests run automatically against the implementation

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `http://localhost:11434` | Ollama API URL |
| `--model` | from workflow.json | Override model name |
| `--output` | `./output/results.json` | Results file path |
| `-v, --verbose` | False | Enable debug logging |

## Scenario 1: Human Catches Broken Tests (Success)

This scenario demonstrates the value of human review. The LLM generates tests that use `pytest.raises()` without importing pytest—syntactically valid Python, but will fail at runtime.

### The Problem

```python
# Generated test code (BROKEN - missing import)
class TestStack:
    def test_pop_empty_stack_raises_index_error(self):
        stack = Stack()
        with pytest.raises(IndexError):  # NameError: 'pytest' not defined!
            stack.pop()
```

**SyntaxGuard passes** because this is valid Python syntax. The `pytest` reference is just an undefined name—not a syntax error.

### Human Review Catches It

```
Approve this artifact? [y/n]: n
Rejection reason: the test does not import pytest
```

The LLM receives feedback and regenerates:

```python
import pytest
from implementation import Stack

class TestStack:
    def test_pop_empty_stack_raises_index_error(self):
        stack = Stack()
        with pytest.raises(IndexError):
            stack.pop()
```

### Result

After human approval, the implementation step succeeds:

- **Duration**: 85 seconds
- **Test attempts**: 3 (2 rejections, 1 approval)
- **Implementation attempts**: 1 (passed first try)
- **All 10 tests pass**

See [output/example_success.json](output/example_success.json) for the full run data.

### Lesson

Human review acts as **semantic validation**—catching issues that automated guards miss. The feedback loop allows the LLM to learn from mistakes within a single workflow execution.

## Scenario 2: Human Approves Broken Tests (Failure)

This scenario shows what happens when the human reviewer doesn't catch the missing import.

### The Mistake

```
Approve this artifact? [y/n]: y  # Approved without noticing missing import!
```

### The Cascade

1. **Implementation Step Begins**: The LLM generates correct Stack code
2. **DynamicTestGuard Runs**: pytest executes the tests
3. **Tests Fail**: `NameError: name 'pytest' is not defined`
4. **LLM Tries to Fix**: But it's told "do NOT modify the tests"
5. **Repeated Failures**: Same error, 4 attempts
6. **Workflow Fails**: Max attempts reached

```
TEST FAILURE:
NameError: name 'pytest' is not defined

Instruction: The implementation failed the test suite. Analyze the failure above and fix your code.
```

The implementation generator **cannot fix the tests**—it's constrained by its prompt to only modify implementation code. The broken tests block all progress.

See [output/example_failure.json](output/example_failure.json) for failure data.

### Lesson

1. **Human review is critical**—approving broken tests blocks the entire workflow
2. **Guard constraints matter**—the implementation generator can't escape its role
3. **Consider additional guards**—`pytest --collect-only` could catch import errors automatically

## Configuration Files

### workflow.json

Defines the workflow structure, model, and guard types:

```json
{
  "name": "TDD Stack with Human Review",
  "specification": "Write pytest-style unit tests for a Stack...",
  "model": "qwen2.5-coder:14b",
  "rmax": 3,
  "action_pairs": {
    "g_test": {
      "guard": "composite",
      "guards": ["syntax", "human"],
      "human_prompt_title": "REVIEW GENERATED TESTS"
    },
    "g_impl": {
      "guard": "dynamic_test",
      "requires": ["g_test"]
    }
  }
}
```

### prompts.json

Contains PromptTemplate fields for each generator:

```json
{
  "g_test": {
    "role": "You are a Test Engineer practicing TDD...",
    "constraints": "1. Use pytest-style test classes...",
    "task": "Write unit tests for a Stack data structure...",
    "feedback_wrapper": "GUARD REJECTION:\n{feedback}\n\nInstruction: ..."
  },
  "g_impl": {
    "role": "You are a Python Developer...",
    "constraints": "1. Class must be named exactly `Stack`...",
    "task": "Implement a Stack data structure...",
    "feedback_wrapper": "TEST FAILURE:\n{feedback}\n\nInstruction: ..."
  }
}
```

## Extending This Example

### Different Data Structures

Modify `workflow.json` specification and `prompts.json` task fields:

- Queue (FIFO)
- LinkedList
- BinaryTree
- HashMap

### Additional Guards

Add guards to the composite chain in `workflow.json`:

```json
"guards": ["syntax", "test_collection", "human"]
```

Then implement in `run.py`:

```python
elif guard_type == "test_collection":
    return TestCollectionGuard()  # Runs pytest --collect-only
```

### Different Models

Override via CLI or modify `workflow.json`:

```bash
python run.py --model codellama:13b
```

## Future Improvements

1. **TestCollectionGuard**: Run `pytest --collect-only` to catch import errors before human review
2. **AST Import Validation**: Parse code and detect undefined names
3. **Parallel Generation**: Generate multiple test variants for human selection
4. **Provenance Visualization**: Show attempt history in a timeline

## Troubleshooting

### "Cannot connect to Ollama"

```bash
# Ensure Ollama is running
ollama serve

# Check available models
ollama list
```

### "No tests collected by pytest"

The test code may not follow pytest conventions. Ensure:

- Test class names start with `Test`
- Test method names start with `test_`
- File would be named `test_*.py`

### "Implementation keeps failing"

Check if the test code itself is broken (missing imports, syntax errors in test logic). If so, you'll need to restart the workflow and reject the broken tests.
