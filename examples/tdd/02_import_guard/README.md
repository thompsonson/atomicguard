# TDD Workflow with Import Guard

This example demonstrates **defense-in-depth validation** using `ImportGuard` before human review. It catches common LLM mistakes (like missing `import pytest`) automatically, preventing cascade failures in multi-step workflows.

## Overview

**Key benefits:**

- Automated import validation before human sees the code
- Fail-fast: catch errors early in the pipeline
- Guard composition: SyntaxGuard → ImportGuard → HumanReviewGuard
- LLM self-correction via guard feedback

### How ImportGuard Works

The guard uses **pure AST analysis** (no subprocess) to:

1. Parse the code into an AST
2. Collect all names that are used (referenced)
3. Collect all names that are defined (imports, assignments, function/class definitions)
4. Report any undefined names as missing imports

This ensures the human reviewer only sees code with valid imports.

### Key Difference: ImportGuard vs TestCollectionGuard

| Aspect | ImportGuard | TestCollectionGuard |
|--------|-------------|---------------------|
| Approach | Pure AST analysis | pytest --collect-only |
| Execution | No code execution | Subprocess + pytest |
| Speed | Fast (milliseconds) | Slower (subprocess overhead) |
| Atomicity | Single responsibility | Bundles import + naming + module errors |
| Dependencies | None | Requires pytest |

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TDD Workflow with Import Guard                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: g_test (Generate Tests)                                   │
│  ┌──────────┐   ┌───────────┐   ┌─────────────────┐   ┌─────────┐ │
│  │ Ollama   │──▶│ Syntax    │──▶│ Import          │──▶│ Human   │ │
│  │ Generator│   │ Guard     │   │ Guard           │   │ Review  │ │
│  └──────────┘   └───────────┘   └─────────────────┘   └─────────┘ │
│        ▲                                                    │      │
│        │              Rejection feedback                    │      │
│        └────────────────────────────────────────────────────┘      │
│                                                    │               │
│                                                    ▼               │
│                                             ┌──────────┐          │
│                                             │ Test     │          │
│                                             │ Artifact │          │
│                                             └──────────┘          │
│                                                    │               │
│  Step 2: g_impl (Generate Implementation)          │               │
│  ┌──────────┐    ┌──────────────────┐             │               │
│  │ Ollama   │───▶│ DynamicTestGuard │◀────────────┘               │
│  │ Generator│    │ (runs pytest)    │                             │
│  └──────────┘    └──────────────────┘                             │
│        ▲                    │                                      │
│        │    Test failures   │                                      │
│        └────────────────────┘                                      │
│                             │                                      │
│                             ▼                                      │
│                      ┌──────────────┐                             │
│                      │ Implementation│                             │
│                      │ Artifact      │                             │
│                      └──────────────┘                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
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
cd examples/basics/03_tdd_import_guard
python run.py --host http://localhost:11434
```

## Expected Behavior

### Scenario: LLM Generates Tests Without `import pytest`

1. **LLM generates tests** using `pytest.raises()` without importing pytest
2. **SyntaxGuard passes** (valid Python syntax)
3. **ImportGuard fails** with clear feedback:

   ```
   GUARD REJECTION:
   Undefined names (missing imports?): pytest

   Instruction: Your test code has structural issues. Fix the specific problem above.
   ```

4. **LLM regenerates** with `import pytest` at the top
5. **ImportGuard passes**, human reviews correct tests
6. **Implementation succeeds**

### Why This Matters

Without automated import validation:

1. Human reviewer might miss `import pytest`
2. Tests get approved with missing import
3. Implementation step begins
4. DynamicTestGuard fails: `NameError: name 'pytest' is not defined`
5. LLM tries to fix implementation (but can't modify tests)
6. Workflow fails after max attempts - **unrecoverable**

**The ImportGuard catches this early**, allowing the LLM to self-correct before human review.

## Configuration Files

### workflow.json

```json
{
  "name": "TDD Stack with Import Guard",
  "action_pairs": {
    "g_test": {
      "guard": "composite",
      "guards": ["syntax", "import", "human"],
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

Note: The prompts do **not** mention `import pytest`. The guard catches this automatically, demonstrating that guards can compensate for incomplete prompts.

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `http://localhost:11434` | Ollama API URL |
| `--model` | from workflow.json | Override model name |
| `--output` | `./output/results.json` | Results file path |
| `-v, --verbose` | False | Enable debug logging |

## The ImportGuard

Located at `src/atomicguard/guards/static/imports.py`, this guard:

1. **Parses code into an AST** (no code execution)
2. **Collects defined names** (imports, functions, classes, assignments)
3. **Collects used names** (all Name nodes in Load context)
4. **Compares** used vs defined + builtins
5. **Returns GuardResult** with undefined names as feedback

### What It Catches

- Missing imports (`import pytest`, `from typing import ...`)
- Typos in variable names
- Use of undefined functions or classes

### What It Doesn't Catch

- Invalid test class/function naming conventions
- Logical errors in test assertions
- Runtime errors (only static analysis)
- Tests that don't cover requirements

These issues are caught by other guards (DynamicTestGuard) or human review.

## Extending This Example

### Add More Guards

You can add additional guards to the chain:

```json
"guards": ["syntax", "test_collection", "coverage_check", "human"]
```

### Different Data Structures

Modify `workflow.json` specification for:

- Queue
- LinkedList
- BinaryTree
- HashMap

### Guard Composition

The `ImportGuard` is designed as an atomic guard with a single responsibility. It can be combined with other atomic guards using `CompositeGuard` to build layered validation.

## Troubleshooting

### "Undefined names (missing imports?): ..."

The ImportGuard detected names that are used but not defined. This means:

- A module is used without being imported (e.g., `pytest.raises()` without `import pytest`)
- A variable is referenced before assignment
- There's a typo in a name

### Guard passes but tests fail at runtime

The ImportGuard only validates static structure. It doesn't catch:

- Runtime errors in executed code
- Missing attributes on objects
- Type mismatches

These are caught by DynamicTestGuard during test execution.

### Human review still sees issues

The ImportGuard validates imports only. Human review is still important for:

- Verifying test coverage
- Checking assertion logic
- Ensuring tests match requirements
- Validating naming conventions

## Pedagogical Value

This example teaches:

1. **Defense in depth** - Multiple automated guards before human review
2. **Fail-fast validation** - Catch errors as early as possible in the pipeline
3. **Guard composition** - CompositeGuard chains multiple validators
4. **LLM feedback loops** - Guards train LLMs within a single workflow
5. **Separation of concerns** - SyntaxGuard (syntax) vs ImportGuard (imports) vs HumanReviewGuard (judgment)
6. **Guard atomicity** - Each guard has a single responsibility for maintainability
