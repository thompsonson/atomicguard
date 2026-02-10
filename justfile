# AtomicGuard Development Commands

# Default: run unit tests
default: test

# ─── Testing ─────────────────────────────────────────────

# Run unit tests (domain + application + infrastructure)
test:
    PYTHONPATH=src python -m pytest tests/domain/ tests/application/ tests/infrastructure/ -q

# Run architecture tests only
test-arch:
    PYTHONPATH=src python -m pytest tests/architecture/ -v

# Run all tests
test-all: test test-arch

# Run with coverage report
coverage:
    PYTHONPATH=src python -m pytest --cov=src/atomicguard --cov-report=term-missing tests/domain/ tests/application/ tests/infrastructure/

# ─── Smoke Tests ─────────────────────────────────────────

# Run all zero-dependency smoke tests
smoke:
    PYTHONPATH=src python -m examples.basics.01_mock
    PYTHONPATH=src python -m examples.basics.05_versioned_env
    PYTHONPATH=src python -m examples.basics.06_extraction
    PYTHONPATH=src python -m examples.basics.07_multiagent
    PYTHONPATH=src python -m examples.basics.08_incremental

# Core smoke test only (fastest check that the framework works)
smoke-core:
    PYTHONPATH=src python -m examples.basics.01_mock

# ─── Code Quality ────────────────────────────────────────

# Lint check
lint:
    uv run ruff check src tests

# Format check
fmt-check:
    uv run ruff format --check src tests

# Auto-format
fmt:
    uv run ruff format src tests

# Type check
typecheck:
    uv run mypy src

# ─── CI Pipeline ─────────────────────────────────────────

# Full CI: lint + typecheck + all tests + smoke
ci: lint fmt-check typecheck test-all smoke
