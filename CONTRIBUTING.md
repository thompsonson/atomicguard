# Contributing to AtomicGuard

Thank you for your interest in contributing to AtomicGuard!

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/thompsonson/atomicguard.git
   cd atomicguard
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev,test]"
   ```

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/atomicguard --cov-report=html

# Run specific test file
pytest tests/guards/test_syntax.py

# Run tests matching a pattern
pytest -k "test_valid"
```

### Code Quality

```bash
# Linting
ruff check src tests

# Formatting
ruff format src tests

# Type checking
mypy src
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/) for automatic versioning and changelog generation.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat:` - New features (triggers minor version bump)
- `fix:` - Bug fixes (triggers patch version bump)
- `docs:` - Documentation changes
- `chore:` - Maintenance tasks
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `ci:` - CI/CD changes

### Breaking Changes

Add `BREAKING CHANGE:` in the footer or `!` after the type to trigger a major version bump:

```
feat!: remove deprecated API

BREAKING CHANGE: The old API has been removed.
```

### Examples

```
feat(guards): add JSONSchemaGuard for structured output validation
fix(workflow): handle empty dependency list correctly
docs: update installation instructions
test(syntax): add edge cases for Python 3.12 syntax
refactor(agent): simplify retry logic
```

## Pull Request Process

1. Create a feature branch from `main`:

   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes with appropriate tests

3. Ensure all checks pass:

   ```bash
   pytest
   ruff check src tests
   ruff format --check src tests
   mypy src
   ```

4. Commit using conventional commits

5. Push and open a PR against `main`

6. Wait for CI to pass and request review

## Project Structure

```
src/atomicguard/
├── domain/           # Core business logic (no external deps)
│   ├── models.py     # Immutable dataclasses
│   ├── interfaces.py # ABC ports
│   └── exceptions.py # Domain exceptions
├── application/      # Orchestration layer
│   ├── action_pair.py
│   ├── agent.py
│   └── workflow.py
├── infrastructure/   # External system adapters
│   ├── llm/          # LLM implementations
│   └── persistence/  # Storage implementations
└── guards/           # Guard implementations
```

## Adding a New Guard

1. Create a new file in `src/atomicguard/guards/`
2. Implement `GuardInterface` from `domain.interfaces`
3. Add tests in `tests/guards/`
4. Export from `guards/__init__.py`

Example:

```python
from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

class MyGuard(GuardInterface):
    def validate(self, artifact: Artifact, **deps) -> GuardResult:
        # Your validation logic
        if valid:
            return GuardResult(passed=True, feedback="Validation passed")
        return GuardResult(passed=False, feedback="Why it failed")
```

## Questions?

Open an issue on GitHub or reach out to the maintainers.
