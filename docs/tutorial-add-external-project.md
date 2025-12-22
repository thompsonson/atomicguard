# Using ADD on External Projects

This tutorial shows how to use AtomicGuard's ADD (Architecture-Driven Development) workflow to generate architecture tests for any Python project.

## Prerequisites

- Python 3.12+
- Ollama running locally with a capable model
- Your project has architecture documentation in Markdown format

---

## Step 1: Install atomicguard

```bash
cd /path/to/your-project

# Option A: Install from PyPI (when published)
uv add atomicguard

# Option B: Install from local development copy
uv add --editable /path/to/atomicguard
```

---

## Step 2: Run ADD Against Your Documentation

```bash
cd /path/to/your-project

uv run python -m examples.add.run \
  --docs documentation/architecture.md \
  --workdir tests/architecture \
  --host http://localhost:11434 \
  --model ollama:qwen2.5-coder:14b \
  --min-gates 3 \
  --min-tests 3 \
  --rmax 5 \
  -v
```

### Options

| Option | Description |
|--------|-------------|
| `--docs` | Path to your architecture documentation |
| `--workdir` | Directory where tests will be written |
| `--host` | Ollama API URL |
| `--model` | Model to use (format: `ollama:model-name`) |
| `--min-gates` | Minimum architecture gates to extract |
| `--min-tests` | Minimum tests to generate |
| `--rmax` | Maximum retry attempts per action pair |
| `-v` | Verbose output |

---

## Step 3: Verify Generated Tests

```bash
# Check generated test file
cat tests/architecture/test_gates.py

# Verify only whitelisted pytestarch methods are used
grep -E '\.(classes_that|should_be_in_packages|or_|and_)\(' \
  tests/architecture/test_gates.py || echo "✓ No invalid methods"
```

---

## Step 4: Fix Fixture Paths

The generated tests use placeholder paths. Edit `tests/architecture/test_gates.py` to use your actual project paths:

```python
# Change FROM:
@pytest.fixture(scope="module")
def evaluable():
    return get_evaluable_architecture("/project", "/project/src")

# Change TO:
@pytest.fixture(scope="module")
def evaluable():
    return get_evaluable_architecture(
        "/path/to/your-project",
        "/path/to/your-project/src"
    )
```

**Tip**: Include a `Package Configuration` section in your documentation with `Source Root` to have ADD automatically use the correct path:

```markdown
## Package Configuration

- **Source Root**: `src/mypackage`
- **Package Name**: `mypackage`
```

---

## Step 5: Run the Architecture Tests

```bash
cd /path/to/your-project

# Ensure pytestarch is installed
uv add pytestarch

# Run the generated tests
uv run pytest tests/architecture/test_gates.py -v
```

---

## Expected Results

1. ADD extracts architecture gates from your documentation
2. Generated tests enforce dependency rules like:
   - Domain must not import infrastructure
   - Application layer depends on domain, not infrastructure
   - Repository pattern isolation
3. Tests either pass (architecture is clean) or fail (violations found)

---

## Troubleshooting

### ADD fails to run

```bash
# Check if atomicguard is installed
uv pip list | grep atomicguard

# Check if pytestarch is available
uv run python -c "import pytestarch; print(pytestarch.__version__)"

# Check Ollama is running
curl http://localhost:11434/api/tags
```

### Tests fail to import

```bash
# Ensure pytestarch is installed in your project
uv add pytestarch
```

### Guard failures during generation

If ADD exhausts retries, check the verbose output for guard feedback. Common issues:

- Generated code uses invalid pytestarch API methods
- Test names don't follow `test_` convention
- Syntax errors in generated code

---

## Architecture Documentation Tips

For best results, include these sections in your documentation:

1. **Package Configuration** - Source root and package name
2. **Architecture Gates** - Numbered rules with clear constraints
3. **Layer Boundaries** - Which layers can import which
4. **Ubiquitous Language** - Domain terms and their definitions

Example:

```markdown
## Architecture Gates

### Gate 1: Domain Independence
The domain layer must not import from infrastructure or application layers.

### Gate 2: Repository Pattern
All data access must go through repository interfaces defined in domain.

## Layer Boundaries

- `domain` → no external dependencies
- `application` → depends on `domain` only
- `infrastructure` → depends on `domain` and `application`
```
