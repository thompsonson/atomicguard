# Architecture Cleanup Issues

Anti-patterns identified during architectural review (2026-02-10). Each issue is self-contained — tests pass before and after. Ordered by dependency and risk.

---

## Dependency Graph

```
Issue 1 (move to domain/) ──────────────── standalone
Issue 2 (remove Extension 10) ──────────── standalone
Issue 3 (DI fix) ───────────────────────── standalone
Issue 4 (domain purity) ────────────────── standalone
Issue 5 (checkpoint eval) ──────────────── after Issue 4
Issue 6 (runner dedup) ─────────────────── after Issues 1, 2
```

Issues 1-4 are independent and can be done in any order. Recommend starting with Issue 1 (quick win) then Issue 2 (biggest payoff).

---

## Issue 1: Move `StagnationInfo` and `FeedbackSummarizer` to domain/

**Problem**: `StagnationInfo` (frozen dataclass implementing Definition 44) and `FeedbackSummarizer` (pure business rules — similarity comparison, error signature extraction) are in `application/` but contain no orchestration or infrastructure coordination. They are domain concepts.

**Files to change**:
- `src/atomicguard/application/feedback_summarizer.py` → split into:
  - `src/atomicguard/domain/models.py` (add `StagnationInfo`)
  - `src/atomicguard/domain/feedback_summarizer.py` (move `FeedbackSummarizer`)
- Update imports in:
  - `src/atomicguard/application/agent.py`
  - `tests/application/test_feedback_summarizer.py`

**Risk**: None. Pure file move + import updates. No behavior change.

**Acceptance criteria**: All existing tests pass. No application/ imports needed for `StagnationInfo` or `FeedbackSummarizer`.

---

## Issue 2: Remove Extension 10 (WorkflowEventStore/Emitter)

**Problem**: The `WorkflowEventStore` is a materialized view of data already in the artifact DAG. Every event type (STEP_START, STEP_PASS, STEP_FAIL, STAGNATION, ESCALATE, CASCADE_INVALIDATE) is either directly stored in the DAG as artifact metadata or derivable from DAG data + workflow configuration. This creates:
- Two sources of truth (already found a bug: `final_error_analysis.py:108` globs wrong path, silently returns empty)
- 10 `if emitter:` blocks polluting the core workflow loop
- 6 extra files of infrastructure for derived data

**Context**: The formal framework (domain_definitions.md §2.2.3) defines the execution trace as the DAG itself. Extension 10 was implemented without formal definitions (the "Definition 50-51" referenced in code docstrings don't exist). Workflow events are S_workflow transitions, not S_env artifacts — but all the data they capture is already persisted via S_env (the artifact DAG).

**Files to remove**:
- `src/atomicguard/domain/workflow_event.py` (WorkflowEvent, WorkflowEventType, EscalationEventRecord)
- `src/atomicguard/infrastructure/persistence/workflow_events.py` (InMemoryWorkflowEventStore, FilesystemWorkflowEventStore)
- `src/atomicguard/application/workflow_event_emitter.py` (WorkflowEventEmitter)
- `scripts/trace_report.py`

**Files to modify**:
- `src/atomicguard/domain/interfaces.py` — remove `WorkflowEventStoreInterface`
- `src/atomicguard/application/workflow.py` — remove `event_store` parameter, remove all `if emitter:` blocks, remove `emitter` parameter from `_invalidate_dependents()`
- `src/atomicguard/domain/__init__.py` — remove workflow_event exports
- `src/atomicguard/__init__.py` — remove workflow_event exports
- `examples/swe_bench_pro/experiment_runner.py` — remove `event_store` wiring, remove `FilesystemWorkflowEventStore` import
- `examples/swe_bench_pro/final_error_analysis.py` — remove `load_trace_escalations()`, `EscalationInfo`, trace sections from report generation
- `examples/swe_bench_ablation/experiment_runner.py` — remove `event_store` wiring
- `examples/swe_bench_ablation/demo.py` — remove `event_store` parameter from `build_workflow()`
- Update all affected tests

**Risk**: Medium. Removes functionality but that functionality is broken (path bug) and unused. Verify no downstream consumers depend on trace files.

**Acceptance criteria**: All existing tests pass (minus removed Extension 10 tests). `workflow.py` execution loop has no event-store related code. Experiment runners still produce correct results via artifact DAG.

---

## Issue 3: Fix dependency inversion in `application/workflow.py`

**Problem**: The application layer lazy-imports concrete infrastructure implementations:

```python
# Line 91-94
if artifact_dag is None:
    from atomicguard.infrastructure.persistence import InMemoryArtifactDAG
    artifact_dag = InMemoryArtifactDAG()

# Line 518-521
if checkpoint_dag is None:
    from atomicguard.infrastructure.persistence import InMemoryCheckpointDAG
    checkpoint_dag = InMemoryCheckpointDAG()
```

This creates a bidirectional dependency: application → infrastructure, violating the dependency inversion principle. The application layer should only depend on domain interfaces (ports).

**Files to change**:
- `src/atomicguard/application/workflow.py` — make `artifact_dag` a required parameter (remove `None` default and lazy import). Same for `checkpoint_dag` if still present after Issue 5.
- Update all call sites that rely on the default (tests, examples) to inject explicitly.

**Risk**: Low. Tests that relied on the default will need an explicit `InMemoryArtifactDAG()` passed in — this is the correct pattern (inject at the edge).

**Acceptance criteria**: No `from atomicguard.infrastructure` imports anywhere in `src/atomicguard/application/`. All tests pass.

---

## Issue 4: Clean up `domain/workflow.py` purity

**Problem**: Three domain layer purity violations in one file:

### 4a: `WorkflowRegistry` is infrastructure in domain/

A singleton in-memory registry with `store()` / `resolve()`. This is persistence logic. The domain should compute `W_ref` (pure hash function); storage belongs in infrastructure.

Additionally, `compute_workflow_ref()` has a side-effecting `store=True` parameter that writes to the registry — making a "pure" hash function impure.

### 4b: `HumanAmendmentProcessor` calls `self._artifact_dag.store()`

Domain logic directly persisting to the repository. The domain should create the model; the application layer should coordinate persistence.

### 4c: Silent `except Exception: pass` (~line 418)

Swallows all exceptions during feedback history reconstruction. Masks bugs completely.

**Files to change**:
- `src/atomicguard/domain/workflow.py`:
  - Extract `WorkflowRegistry` to `src/atomicguard/infrastructure/workflow_registry.py`
  - Make `compute_workflow_ref()` pure (remove `store` parameter)
  - Move `HumanAmendmentProcessor` storage calls to application layer
  - Replace silent exception with explicit error handling or logging
- Update imports across codebase

**Risk**: Low-medium. `compute_workflow_ref()` callers that relied on auto-store need updating. `HumanAmendmentProcessor` callers need to handle persistence themselves.

**Acceptance criteria**: `domain/workflow.py` has no side effects (no writes to registries, no DAG storage). No silent exception swallowing. All tests pass.

---

## Issue 5: Evaluate and remove `CheckpointDAG`

**Problem**: `WorkflowCheckpoint` captures `completed_steps`, `artifact_ids`, `failure_feedback`, `provenance_ids` — all derivable from the artifact DAG. The DAG IS the checkpoint: to resume, reconstruct `WorkflowState` from accepted artifacts.

**Analysis needed**:
- Is checkpoint/resume functionality used in any experiment runner or workflow?
- Is `HumanAmendment` used anywhere in practice?
- Does any code call `CheckpointDAGInterface` methods?

**If unused (likely)**:
- Remove `CheckpointDAGInterface` from `domain/interfaces.py`
- Remove `WorkflowCheckpoint`, `HumanAmendment`, `AmendmentType`, `FailureType` from `domain/models.py`
- Remove `infrastructure/persistence/checkpoint.py`
- Remove `WorkflowResumer`, `HumanAmendmentProcessor` from `domain/workflow.py`
- Remove `checkpoint_dag` parameter threading through `Workflow`
- Update tests

**If used**: Keep but document that DAG-based resume is the intended path forward. Consider a follow-up issue to migrate to DAG-based resume.

**Depends on**: Issue 4 (HumanAmendmentProcessor cleanup)

**Risk**: Medium. Need to verify no active workflows depend on checkpoint resume.

**Acceptance criteria**: If removed — no checkpoint-related code in the codebase. Workflow constructor is simpler. All tests pass.

---

## Issue 6: Deduplicate experiment runner workflow construction

**Problem**: ~400 lines of identical logic duplicated between `swe_bench_pro/experiment_runner.py` and `swe_bench_ablation/demo.py`:
- `build_workflow()` — workflow assembly from JSON config
- `_topological_sort()` — identical implementation
- Generator/guard registry construction — overlapping registries
- `ArmResult` imported cross-example (`swe_bench_pro` imports from `swe_bench_ablation`)

Additionally, domain models are scattered across example scripts:
- `ArmResult` in `swe_bench_ablation/experiment_runner.py`
- `ArmStats`, `GuardFailureStats` in `swe_bench_pro/experiment_runner.py`
- `ExperimentOverview`, `FinalError`, `RetryError`, `EscalationInfo` in `swe_bench_pro/final_error_analysis.py`

Both runners also import `FilesystemArtifactDAG` directly instead of using `ArtifactDAGInterface`.

**Proposed changes**:
- Create `src/atomicguard/application/workflow_builder.py` — shared `build_workflow()`, topological sort, registry construction
- Move shared models (`ArmResult`, etc.) to `domain/` or a shared location
- Both runners import from the shared location
- Use `ArtifactDAGInterface` port in runner signatures (concrete type injected at CLI entry point)

**Depends on**: Issues 1 (StagnationInfo moved), 2 (event_store removed from build_workflow signatures)

**Risk**: Medium-high. Largest change, touches active experiment code. Needs careful testing with both runners.

**Acceptance criteria**: No cross-example imports. No duplicated `build_workflow()`. Runners use port interfaces, not concrete types. Both runners produce identical results to before. All tests pass.

---

## Background: Formal Framework Alignment

These issues were identified by reviewing the implementation against the formal framework in `docs/design/agent_design_process/domain_definitions.md`:

- **S_env** (Information State): The append-only artifact DAG. The execution trace IS the repository (§2.2.3).
- **S_workflow** (Control State): Deterministic FSM tracking guard satisfaction and transition history (§2.4.2).
- **Separation of concerns**: Domain is pure models and business rules. Application orchestrates. Infrastructure adapts.

The anti-patterns arose primarily from Extension 09/10 implementation that introduced parallel stores and observer-pattern coupling instead of working within the existing architectural boundaries.

---

## Issue 0: Architecture Test Infrastructure

**Problem**: `pytestarch>=4.0.1` is in main `dependencies` (line 42 of `pyproject.toml`) instead of the test dependency group. The existing `tests/architecture/test_gate10_infrastructure.py` hand-rolls checks using `importlib`/`inspect` instead of using PyTestArch's `LayerRule` API, which is purpose-built for DDD/Hexagonal enforcement.

**Note**: PyTestArch was originally added as a main dependency because it was used as a guard in the C4AI ML Agents project. In AtomicGuard it is purely a test tool.

This issue should be done **first** — the architecture tests then serve as guardrails for Issues 1-6.

### 0a: Move PyTestArch to test dependencies

```toml
# Remove from main dependencies:
dependencies = [
    # "pytestarch>=4.0.1",  ← REMOVE
]

# Add to test dependency group:
test = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytestarch>=4.0.1",
]
```

### 0b: Rewrite architecture tests using LayerRule API

Replace hand-rolled `importlib`/`inspect` tests with PyTestArch's declarative layer rules.

**File**: `tests/architecture/conftest.py` (NEW)

```python
import os
import pytest
from pytestarch import EvaluableArchitecture, LayeredArchitecture, get_evaluable_architecture


@pytest.fixture(scope="session")
def evaluable() -> EvaluableArchitecture:
    src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    project_path = os.path.join(src_dir, "atomicguard")
    return get_evaluable_architecture(src_dir, project_path)


@pytest.fixture(scope="session")
def layers() -> LayeredArchitecture:
    return (
        LayeredArchitecture()
        .layer("domain").containing_modules(["atomicguard.domain"])
        .layer("application").containing_modules(["atomicguard.application"])
        .layer("infrastructure").containing_modules(["atomicguard.infrastructure"])
    )
```

**File**: `tests/architecture/test_layer_rules.py` (NEW)

```python
from pytestarch import LayerRule


class TestDDDLayerRules:
    """Permanent architecture rules — these enforce the Hexagonal architecture."""

    def test_domain_does_not_access_application(self, evaluable, layers):
        """Domain must be pure — no orchestration dependency."""
        rule = (
            LayerRule()
            .based_on(layers)
            .layers_that().are_named("domain")
            .should_not()
            .access_layers_that().are_named("application")
        )
        rule.assert_applies(evaluable)

    def test_domain_does_not_access_infrastructure(self, evaluable, layers):
        """Domain must not know about adapters."""
        rule = (
            LayerRule()
            .based_on(layers)
            .layers_that().are_named("domain")
            .should_not()
            .access_layers_that().are_named("infrastructure")
        )
        rule.assert_applies(evaluable)

    def test_application_does_not_access_infrastructure(self, evaluable, layers):
        """Application depends on domain ports, not concrete adapters."""
        rule = (
            LayerRule()
            .based_on(layers)
            .layers_that().are_named("application")
            .should_not()
            .access_layers_that().are_named("infrastructure")
        )
        rule.assert_applies(evaluable)
```

### 0c: Keep and update existing Gate 10 tests

The hand-rolled tests in `test_gate10_infrastructure.py` that check specific things (interface naming, abstract methods, mock injectability) are still valuable — PyTestArch doesn't cover those. Keep them but remove any that duplicate the new LayerRule tests.

### Acceptance criteria

- `pytestarch` is in test dependencies only, not main dependencies
- `tests/architecture/test_layer_rules.py` exists with 3 layer rules
- Running `PYTHONPATH=src python -m pytest tests/architecture/ -v` shows all rules
- **Note**: Some rules will FAIL initially (Issue 3: application imports infrastructure). That's expected — the test documents the violation that Issue 3 fixes.

---

## Convention Enforcement Tests

Permanent tests that catch anti-patterns import-based layer rules cannot detect. Add to `tests/architecture/test_conventions.py`.

### Frozen dataclass convention

```python
def test_domain_models_are_frozen():
    """All domain dataclasses must be frozen (except WorkflowState)."""
```

Scan `domain/models.py` for all `@dataclass` classes via AST. Assert every one has `frozen=True` except the documented exception (`WorkflowState`). Catches someone adding a mutable dataclass to the domain.

### Immutable collections in domain models

```python
def test_domain_models_use_tuples_not_lists():
    """Frozen domain model fields should use tuple, not list."""
```

Inspect type hints of all frozen domain dataclasses. Assert fields use `tuple[...]` not `list[...]`, and `MappingProxyType` not `dict[...]` (except `WorkflowState`). Catches someone adding `field: list[str]` to a frozen model.

### No silent exception swallowing

```python
def test_no_bare_except_pass():
    """No silent exception swallowing in src/."""
```

AST-walk all `.py` files under `src/atomicguard/`. Flag any `except` handler where the body is only `pass` or `...`. The `domain/workflow.py:418` silent exception would have been caught.

### Interface naming convention

Already exists in Gate 10D — keep as-is. Inspect `domain/interfaces.py` for all ABCs, assert they end with `Interface`.

### Ports have abstract methods

```python
def test_all_interface_methods_are_abstract():
    """Every public method on a port must be abstract."""
```

For each class in `domain/interfaces.py` ending with `Interface`, assert every public method (not starting with `_`) has `__isabstractmethod__ = True`. Catches someone adding a concrete method to a port.

---

## Smoke Tests

Zero-dependency examples that exercise the core framework end-to-end. These should be runnable in CI without API keys, Docker, or external services.

| Example | What it exercises | Priority |
|---|---|---|
| `basics/01_mock.py` | `DualStateAgent`, `MockGenerator`, `SyntaxGuard`, `ActionPair`, retry loop, `InMemoryArtifactDAG` | **Critical** — core agent loop |
| `basics/05_versioned_env.py` | `compute_workflow_ref`, `WorkflowRegistry`, `Context.amend()` | High — Extension 01 |
| `basics/06_extraction.py` | Predicate queries, `extract()`, all predicate combinators | High — Extension 02 |
| `basics/07_multiagent.py` | `MultiAgentSystem`, shared repository, agent coordination | Medium — Extension 03 |
| `basics/08_incremental.py` | `compute_config_ref`, change detection, Merkle propagation | Medium — Extension 07 |

**Not suitable for smoke tests** (require external dependencies):
- `basics/02_ollama.py` — needs Ollama running
- `basics/03_huggingface.py` — needs HF API key
- `checkpoint/01_basic/demo.py` — needs `click`, uses filesystem checkpoint (depends on Issue 5 outcome)

---

## Justfile

Standardized commands for development workflow. Create as `justfile` in repository root.

```just
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
```

### Usage

```bash
just              # run unit tests (default)
just test-arch    # check architecture rules
just smoke        # run zero-dep examples
just ci           # full pipeline
```
