# Part 1: Quick Start

Get ADD running in 5 minutes.

## Step 1: Verify Prerequisites

```bash
# Check Python version (need 3.12+)
python --version

# Check Ollama is running
curl http://localhost:11434/api/tags

# If Ollama isn't running, start it:
ollama serve
```

## Step 2: Run the Example

From the project root directory:

```bash
uv run python -m examples.add.run --host http://localhost:11434 -v
```

You should see output like this:

```
============================================================
Workflow: ADD Agent Example
============================================================
Model: ollama:qwen2.5-coder:14b
Base URL: http://localhost:11434/v1
Output dir: /path/to/examples/add/output
Prompts: /path/to/examples/add/prompts.json
Min gates: 3
Min tests: 3
Max retries: 3
============================================================

Executing ADD workflow...

INFO     | [ADD] Starting generation pipeline
INFO     | [ADD] === Action Pair 1: Gates Extraction ===
INFO     | [DocParser] Calling Ollama...
INFO     | [DocParser] Got valid structured response
INFO     | [gates_extraction] ✓ Passed: Extracted 8 gates
INFO     | [ADD] === Action Pair 2: Test Generation ===
INFO     | [TestCodeGen] Calling Ollama...
INFO     | [TestCodeGen] Got valid structured response
INFO     | [test_generation] ✓ Passed: All guards passed
INFO     | [ADD] === Action Pair 3: File Writing ===
INFO     | [file_writing] ✓ Passed: Valid manifest: 8 tests, 2 files

=== SUCCESS ===

Artifact ID: 432a6554-2837-41de-a9dc-13b1123db987
Attempt: 1
Duration: 160.50s

Test count: 8
Gates covered: ['Gate1', 'Gate2', 'Gate3', 'Gate4', 'Gate5', 'Gate6', 'Gate7', 'Gate8']

--- Generated Files ---
  - tests/architecture/test_gates.py
  - tests/architecture/__init__.py

Check examples/add/output to see the generated files
```

## Step 3: Examine the Output

### Generated Test File

Open the generated test file:

```bash
cat examples/add/output/tests/architecture/test_gates.py
```

You'll see tests like:

```python
"""
Test suite to enforce architecture gates defined in the project.
"""

from pytestarch import get_evaluable_architecture, Rule
import pytest

@pytest.fixture(scope="module")
def evaluable():
    return get_evaluable_architecture("/project", "/project/src")

def test_gate1_domain_no_infra_imports(evaluable):
    rule = (
        Rule()
        .modules_that()
        .are_sub_modules_of("domain")
        .should_not()
        .import_modules_that()
        .are_sub_modules_of("infrastructure")
    )
    rule.assert_applies(evaluable)

# ... more tests for each gate
```

### Run Log

For detailed execution trace:

```bash
cat examples/add/output/run.log
```

### Artifacts

Internal artifacts (for debugging) are stored in:

```bash
ls examples/add/output/artifacts/
```

## What Just Happened?

1. **ADD read the sample documentation** from `sample_docs/architecture.md`
2. **Extracted 8 architecture gates** (rules about layer dependencies)
3. **Generated 8 pytestarch tests** (one for each gate)
4. **Validated the generated code** using guards (syntax, naming, API correctness)
5. **Wrote the test files** to the output directory

## Next Steps

Now that you've seen ADD in action:

- [Part 2: Understanding Input](02-understanding-input.md) - Learn how to write architecture documentation
- [Part 3: Understanding Output](03-understanding-output.md) - Understand the generated tests

---

**Previous**: [00 - Overview](00-overview.md) | **Next**: [02 - Understanding Input](02-understanding-input.md)
