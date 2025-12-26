# Part 5: Customization

Adapt ADD for your own project.

## Command Line Options

### Basic Options

```bash
uv run python -m examples.add.run --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `http://localhost:11434` | Ollama API URL |
| `--model` | `ollama:qwen2.5-coder:14b` | Model to use |
| `--docs` | `sample_docs/architecture.md` | Architecture documentation path |
| `--workdir` | `examples/add/output` | Output directory for tests |
| `-v, --verbose` | Off | Enable DEBUG logging |

### Threshold Options

| Option | Default | Description |
|--------|---------|-------------|
| `--min-gates` | 3 | Minimum gates required |
| `--min-tests` | 3 | Minimum tests required |
| `--rmax` | 3 | Max retries per stage |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | None | Save results to JSON |
| `--log-file` | `output/run.log` | Log file path |
| `--artifact-dir` | `output/artifacts` | Artifact storage |

## Using Your Own Documentation

### Step 1: Create Your Documentation

Create a file like `docs/architecture.md`:

```markdown
# My Project Architecture

## Layer Structure

- **domain/**: Core business logic
- **application/**: Use cases
- **infrastructure/**: External adapters

## Architecture Gates

### Gate 1: Domain Independence

**Rule**: Domain must not import from infrastructure.

**Rationale**: Keep domain pure and testable.

**Scope**: All modules in `domain/`

**Constraint Type**: dependency

### Gate 2: ...
```

### Step 2: Run ADD

```bash
uv run python -m examples.add.run \
  --docs docs/architecture.md \
  --workdir tests/architecture \
  --host http://localhost:11434 \
  -v
```

### Step 3: Verify Output

```bash
cat tests/architecture/test_gates.py
```

## Configuration Files

### prompts.json

Customize the prompts sent to the LLM:

```json
{
  "gates_extraction": {
    "role": "You are an architecture documentation parser.",
    "constraints": "For each gate:\n- Assign a unique gate_id...",
    "task": "Extract architecture gates from the provided documentation.",
    "feedback_wrapper": "VALIDATION FAILED:\n{feedback}\n\nFix the issues."
  },
  "test_generation": {
    "role": "You are a Python test generator for architecture tests.",
    "constraints": "REQUIRED IMPORTS:\nfrom pytestarch import...",
    "task": "Generate pytestarch tests for the provided architecture gates.",
    "feedback_wrapper": "TEST GENERATION FAILED:\n{feedback}\n\nFix the issue."
  }
}
```

Use a custom prompts file:

```bash
uv run python -m examples.add.run --prompts my-prompts.json
```

### workflow.json

Configure workflow settings:

```json
{
  "name": "My Architecture Tests",
  "model": "qwen2.5-coder:14b",
  "rmax": 5,
  "min_gates": 5,
  "min_tests": 5
}
```

Use a custom workflow file:

```bash
uv run python -m examples.add.run --workflow my-workflow.json
```

## Using Different Models

### Ollama Models

```bash
# Default
uv run python -m examples.add.run --model ollama:qwen2.5-coder:14b

# CodeLlama
uv run python -m examples.add.run --model ollama:codellama:34b

# DeepSeek Coder
uv run python -m examples.add.run --model ollama:deepseek-coder:6.7b
```

### Remote Ollama

```bash
# GPU server
uv run python -m examples.add.run --host http://gpu-server:11434
```

### OpenAI (future support)

```bash
# When implemented
uv run python -m examples.add.run --model openai:gpt-4
```

## Adjusting Retry Behavior

### More Retries

For complex documentation or weaker models:

```bash
uv run python -m examples.add.run --rmax 5
```

### Fewer Retries

For simple documentation or strong models:

```bash
uv run python -m examples.add.run --rmax 1
```

## Adjusting Thresholds

### Stricter Validation

Require more gates and tests:

```bash
uv run python -m examples.add.run \
  --min-gates 10 \
  --min-tests 10
```

### Lenient Validation

For quick testing:

```bash
uv run python -m examples.add.run \
  --min-gates 1 \
  --min-tests 1
```

## Saving Results

### JSON Output

```bash
uv run python -m examples.add.run \
  --output results/experiment1.json
```

Output:

```json
{
  "timestamp": "2024-12-20T13:22:50.123456",
  "duration_seconds": 160.5,
  "success": true,
  "artifact_id": "432a6554-...",
  "attempt_number": 1,
  "manifest": {
    "test_count": 8,
    "gates_covered": ["Gate1", "Gate2", ...],
    "files": [...]
  }
}
```

### Custom Log File

```bash
uv run python -m examples.add.run \
  --log-file logs/run-$(date +%Y%m%d-%H%M%S).log
```

## Example: Full Production Run

```bash
uv run python -m examples.add.run \
  --host http://gpu-server:11434 \
  --model ollama:qwen2.5-coder:14b \
  --docs docs/architecture.md \
  --workdir src/tests/architecture \
  --min-gates 8 \
  --min-tests 8 \
  --rmax 5 \
  --output results/production-run.json \
  --verbose
```

## Exercise: Custom Configuration

1. Copy `sample_docs/architecture.md` to `my-docs/architecture.md`
2. Add a custom Gate 9 (see Part 2)
3. Create a custom `my-prompts.json` with modified role text
4. Run ADD with your custom configuration:

```bash
uv run python -m examples.add.run \
  --docs my-docs/architecture.md \
  --prompts my-prompts.json \
  --workdir my-output \
  --min-gates 9 \
  -v
```

1. Verify 9 tests were generated

---

**Previous**: [04 - The Pipeline](04-pipeline.md) | **Next**: [06 - Programmatic Usage](06-programmatic.md)
