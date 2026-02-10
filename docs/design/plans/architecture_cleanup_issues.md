# Architecture Cleanup Issues

Anti-patterns identified during architectural review (2026-02-10). Each issue is self-contained — tests pass before and after. Ordered by dependency and risk.

## Status Summary (2026-02-10)

| Issue | Status | Commit |
|-------|--------|--------|
| 0: Architecture test infrastructure | DONE | `2c3f98d` |
| 1: Move StagnationInfo/FeedbackSummarizer to domain/ | DONE | `fd50863` |
| 2: Remove Extension 10 (WorkflowEventStore/Emitter) | DONE | `e27d20a` |
| 3: Fix dependency inversion in application/workflow.py | DONE | `7e7ccea` |
| 4: Fix silent exception swallowing in domain/ | DONE | `e44e170` |
| 5: Remove CheckpointDAG (DAG is the checkpoint) | PLANNED | ~3,500 lines across 24 files |
| 6: Deduplicate experiment runner workflow construction | PLANNED | extract to `examples/swe_bench_common/` |

**Test results after all changes**: 384 passed, 44 skipped, 0 xfailed. All 3 architecture layer rules pass. All convention tests pass.

---

## Dependency Graph

```
Issue 1 (move to domain/) ──────────────── standalone       ✓ DONE
Issue 2 (remove Extension 10) ──────────── standalone       ✓ DONE
Issue 3 (DI fix) ───────────────────────── standalone       ✓ DONE
Issue 4 (domain purity) ────────────────── standalone       ✓ DONE
Issue 5 (remove checkpoint) ────────────── after Issue 4    📋 PLANNED
Issue 6 (runner dedup) ─────────────────── after Issues 1,2 📋 PLANNED (after Issue 5)
```

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

## Issue 5: Remove `CheckpointDAG` — the artifact DAG IS the checkpoint

**Status**: PLANNED

**Depends on**: Issue 4 (silent exception cleanup) — DONE

### Analysis

The CheckpointDAG is redundant. Nearly every piece of data it stores is either directly in the artifact DAG or derivable from it:

| WorkflowCheckpoint field | Already in Artifact DAG? |
|---|---|
| `workflow_id` | Yes — every `artifact.workflow_id` |
| `specification` | Yes — `artifact.context.specification` (immutable per workflow) |
| `constraints` | Yes — `artifact.context.constraints` (immutable per workflow) |
| `rmax` | Not in artifacts — but this is a **workflow config parameter**, not historical data. The caller provides it. See note below. |
| `completed_steps` | Yes — query artifacts where `status=ACCEPTED` and group by `action_pair_id` |
| `artifact_ids` | Yes — direct mapping `action_pair_id → artifact_id` for accepted artifacts |
| `failure_type` | Derivable — `guard_result.fatal` → ESCALATION, else RMAX_EXHAUSTED |
| `failed_step` | Yes — `action_pair_id` of the last rejected artifact |
| `failed_artifact_id` | Yes — the artifact itself exists in the DAG |
| `failure_feedback` | Yes — `artifact.guard_result.feedback` |
| `provenance_ids` | Yes — `artifact.previous_attempt_id` chain |
| `workflow_ref` | Yes — `artifact.workflow_ref` (already on every Artifact, models.py:128) |

**WorkflowCheckpoint is a denormalized snapshot of derivable data.** To resume, reconstruct `WorkflowState` from the DAG: accepted artifacts = completed steps.

This is the same reasoning that justified removing Extension 10 (Issue 2): the event store was a materialized view of data already in the DAG. The checkpoint system is the same pattern — redundant derived data stored separately, creating two sources of truth.

**Note on `rmax`, `r_patience`, `e_max`**: These are workflow **configuration parameters** — they describe how the workflow is configured to run *now*, not what happened in the past. They belong to the workflow definition (and are included in the W_ref hash), not to the execution trace. The artifact DAG captures historical data (what happened); the workflow config captures operational parameters (how to run). On resume, the caller provides the current workflow config — these values don't need to be stored in a checkpoint.

**Verdict**: Proceed with removal. The redundancy is effectively 100% for historical data. `ResumableWorkflow` is deprecated, `Workflow.execute()` never creates checkpoints, and the "recommended" replacement (`CheckpointService` + `WorkflowResumeService`) has no real-world usage outside tests.

**HumanAmendment** content becomes an `Artifact` with `source=HUMAN`. Amendment metadata (`amendment_type`, `created_by`, `context`, `additional_rmax`) maps to `artifact.metadata`.

### Scope: ~3,500 lines across 24 files

| Category | Files | ~Lines |
|---|---|---|
| Domain (models, interfaces, workflow) | 3 | 610 |
| Application (checkpoint_service, resume_service, workflow) | 4 | 700 |
| Infrastructure (persistence/checkpoint.py) | 2 | 380 |
| Tests | 6 | 700+ |
| Examples (checkpoint/01-04, base/checkpoint) | 6 | 600+ |
| Docs | 3 | 500+ |

### Files to remove entirely

| File | What it contains | Lines |
|---|---|---|
| `src/atomicguard/application/checkpoint_service.py` | `CheckpointService` — creates denormalized snapshots | ~131 |
| `src/atomicguard/application/resume_service.py` | `WorkflowResumeService`, `ResumeResult` — reads denormalized snapshots | ~243 |
| `src/atomicguard/infrastructure/persistence/checkpoint.py` | `InMemoryCheckpointDAG`, `FilesystemCheckpointDAG` | ~362 |
| `tests/application/test_checkpoint_service.py` | Tests for CheckpointService | ~100+ |
| `tests/application/test_resume_service.py` | Tests for WorkflowResumeService | ~100+ |
| `tests/application/test_resumable_workflow.py` | Tests for deprecated ResumableWorkflow | ~500+ |
| `tests/infrastructure/persistence/test_checkpoint.py` | Tests for checkpoint DAG implementations | ~100+ |
| `examples/base/checkpoint.py` | Shared checkpoint CLI utilities | ~80+ |
| `examples/checkpoint/01_basic/demo.py` | Basic checkpoint demo | ~80+ |
| `examples/checkpoint/02_tdd/demo.py` | TDD checkpoint demo | ~80+ |
| `examples/checkpoint/03_llm/demo.py` | LLM checkpoint demo | ~50+ |
| `examples/checkpoint/04_sdlc/demo.py` | SDLC checkpoint demo | ~80+ |

### Files to modify

| File | Changes |
|---|---|
| `src/atomicguard/domain/interfaces.py` | Remove `CheckpointDAGInterface` (lines 171-266) |
| `src/atomicguard/domain/models.py` | Remove `WorkflowCheckpoint`, `HumanAmendment`, `FailureType`, `AmendmentType` (lines 274-350). Remove `checkpoint` field from `WorkflowResult`. |
| `src/atomicguard/domain/workflow.py` | Remove `WorkflowResumer`, `HumanAmendmentProcessor`, `ResumeResult`, `RestoredWorkflowState`, `ProcessResult` (lines 211-648). Keep `WorkflowRegistry`, `compute_workflow_ref`, `resolve_workflow_ref`, `compute_config_ref`. |
| `src/atomicguard/application/workflow.py` | Remove `ResumableWorkflow` class (lines 387-845). Remove checkpoint-related imports (`CheckpointDAGInterface`, `HumanAmendment`, `AmendmentType`, `FailureType`, `WorkflowCheckpoint`). |
| `src/atomicguard/application/__init__.py` | Remove `CheckpointService`, `WorkflowResumeService`, `ResumeResult` exports |
| `src/atomicguard/infrastructure/persistence/__init__.py` | Remove `InMemoryCheckpointDAG`, `FilesystemCheckpointDAG` exports |
| `src/atomicguard/domain/__init__.py` | No changes needed (doesn't export checkpoint types) |
| `src/atomicguard/__init__.py` | No changes needed (doesn't export checkpoint types) |
| `tests/conftest.py` | Remove `sample_checkpoint`, `sample_amendment` fixtures and checkpoint-related imports |
| `tests/domain/test_models.py` | Remove tests for `WorkflowCheckpoint`, `HumanAmendment`, `FailureType`, `AmendmentType` |
| `tests/extensions/test_ext01_versioned_env.py` | Remove tests that use `WorkflowResumer` or `WorkflowCheckpoint`; keep `compute_workflow_ref` tests |
| `tests/architecture/test_gate10_infrastructure.py` | Remove references to `CheckpointDAGInterface` |

### Implementation order

1. **Remove ResumableWorkflow** from `application/workflow.py` — it's deprecated and the biggest consumer
2. **Remove CheckpointService and WorkflowResumeService** — application services that only exist for checkpoint operations
3. **Remove WorkflowResumer and HumanAmendmentProcessor** from `domain/workflow.py` — domain classes that read from checkpoint
4. **Remove domain models** — `WorkflowCheckpoint`, `HumanAmendment`, `FailureType`, `AmendmentType` from `models.py`; `CheckpointDAGInterface` from `interfaces.py`; `checkpoint` field from `WorkflowResult`
5. **Remove infrastructure** — `persistence/checkpoint.py`
6. **Remove checkpoint tests and examples** — all test files and example demos
7. **Clean up re-exports and fixtures** — `__init__.py` files, `conftest.py`

### What stays

- `compute_workflow_ref()`, `resolve_workflow_ref()`, `WorkflowRegistry` — W_ref is a domain concept (Definition 11), independent of checkpoints
- `compute_config_ref()` — Extension 07 config fingerprinting, unrelated
- `WorkflowIntegrityError` — needed by W_ref verification
- `Artifact.workflow_ref` field — already on every artifact
- `Artifact.source` with `ArtifactSource.HUMAN` — stays for human-provided artifacts

### Risk

Medium. Removes ~3,500 lines but all checkpoint functionality is either unused by core workflows or redundant. The `Workflow.execute()` method (the non-deprecated path) never creates checkpoints — it returns `WorkflowResult` with `ESCALATION` or `FAILED` status. Checkpoint creation only happens in the deprecated `ResumableWorkflow`.

**Accepted trade-off**: Resume capability is removed entirely. If needed in the future, it can be redesigned on top of the artifact DAG (reconstruct `WorkflowState` from accepted artifacts, accept config params from the caller).

### Verification

```bash
# Run full test suite (expect ~104 fewer tests: 384 → ~280, 0 failures)
PYTHONPATH=src python -m pytest tests/ -v

# Check architecture rules pass
PYTHONPATH=src python -m pytest tests/architecture/ -v

# Verify no lingering references
grep -r "CheckpointDAG\|WorkflowCheckpoint\|HumanAmendment\|ResumableWorkflow" src/

# Smoke test core workflow
PYTHONPATH=src python -m examples.basics.01_mock
```

### Commit strategy

Single atomic commit — all removals in one commit to keep the tree green at every point:
```
Remove checkpoint infrastructure (Issue 5)

Remove deprecated ResumableWorkflow, CheckpointService, WorkflowResumeService,
CheckpointDAGInterface, WorkflowCheckpoint, HumanAmendment, and all related
infrastructure, tests, and examples (~3,500 lines across 24 files).

The checkpoint system was a denormalized snapshot of data already in the artifact
DAG. Core Workflow.execute() never used checkpoints. Resume capability can be
redesigned on top of ArtifactDAG if needed in the future.
```

### Acceptance criteria

- All architecture layer rules pass
- All convention tests pass
- No `CheckpointDAG`, `WorkflowCheckpoint`, `HumanAmendment` references in `src/`
- `WorkflowResult.checkpoint` field removed
- Tests pass (~280 pass, ~104 fewer than before, 0 failures)
- No regression in `Workflow.execute()` behavior

---

## Issue 6: Deduplicate experiment runner workflow construction

**Status**: PLANNED

**Depends on**: Issues 1, 2 (DONE). Issue 5 recommended first (removes checkpoint wiring from examples).

### Analysis

Two experiment runners with significant duplication:

| Runner | File | Lines | Purpose |
|---|---|---|---|
| Ablation | `examples/swe_bench_ablation/experiment_runner.py` | 353 | Sequential single-language benchmark runner |
| Pro | `examples/swe_bench_pro/experiment_runner.py` | 1251 | Parallel multi-language runner with Docker, progress tracking |

Pro is a **language-aware, Docker-capable superset** of Ablation. Key divergences:
- Language config handling (`LanguageConfig`, language-aware registries)
- `ThreadPoolExecutor` for parallel execution
- `ProgressTracker` with ETA and per-arm stats
- Test infrastructure extraction (`_get_test_infrastructure`, `_find_sample_test`)
- Enhanced error handling with `_run_git()` helper

### Duplication map

| Function | Ablation | Pro | Overlap |
|---|---|---|---|
| `_topological_sort()` | in `demo.py` | lines 119-134 | **100% identical** |
| `load_prompts()` | in `demo.py` | lines 98-116 | **100% identical** (redefined) |
| `load_workflow_config()` | in `demo.py` | lines 90-95 | **90%** (same logic, different `_WORKFLOW_DIR`) |
| `_load_existing_results()` | lines 322-352 | lines 1221-1250 | **100% identical** |
| `__init__()` | lines 46-73 | lines 607-624 | **95%** (same params) |
| `run_instance()` | lines 75-185 | lines 630-792 | **70%** (same 4-phase structure; Pro adds language context) |
| `_prepare_repo()` | lines 259-320 | lines 992-1080 | **80%** (same core; Pro adds `_run_git()` helper) |
| `run_all()` | lines 187-257 | lines 798-986 | **40%** (Ablation=sequential, Pro=parallel+progress) |

### Cross-imports (smell)

`ArmResult` defined in ablation, imported by pro at line 21:
```python
from examples.swe_bench_ablation.experiment_runner import ArmResult
```

### Why `examples/base/` cannot be used directly

`examples/base/` provides shared CLI demo infrastructure (`BaseRunner`, `load_prompts`, `load_workflow_config`, `build_guard`), but its abstractions are **fundamentally incompatible** with the SWE-bench runners:

| Aspect | `examples/base/` | SWE runners need |
|---|---|---|
| Execution model | Single `.execute()` → one result | Batch `.run_all(arms, instances)` → many results |
| Config loading | `load_prompts(path: Path)` — explicit path | `load_prompts()` — embedded discovery from `__file__` parent |
| Workflow config | `load_workflow_config(path: Path)` — explicit path | `load_workflow_config(variant: str)` — variant-based |
| BaseRunner | ABC with `.get_specification()` abstract | Dataset-driven: `instance.problem_statement` |
| Guard building | Generic `build_guard(config)` registry | Language-aware with Docker guards, instance context |
| Result type | `ExecutionResult` (single workflow run) | `ArmResult` (batch metadata: tokens, timing, patches) |
| Persistence | One-off `save_workflow_results()` | Incremental JSONL with resume logic |
| Repository | No concept of repo management | `_prepare_repo()` with git clone/checkout/reset |

The SWE runners are **batch experiment harnesses** built around datasets, not CLI demo wrappers. The duplication is between the two SWE runners themselves, not between the runners and `examples/base/`.

### Approach: extract shared SWE-bench utilities

Create `examples/swe_bench_common/` for code shared between ablation and pro runners. This is distinct from `examples/base/` (which serves CLI demos).

```
examples/swe_bench_common/
├── __init__.py
├── models.py          # ArmResult (moved from ablation)
├── config.py          # topological_sort, load_prompts, load_workflow_config
├── git.py             # prepare_repo (core clone/checkout logic)
└── results.py         # load_existing_results (JSONL resume logic)
```

#### `models.py` — shared data models (~30 lines)

```python
@dataclass
class ArmResult:
    """Result of running one arm on one instance."""
    instance_id: str
    arm: str
    patch_content: str | None = None
    # ... (moved from ablation/experiment_runner.py)
```

#### `config.py` — shared config utilities (~60 lines)

```python
def topological_sort(action_pairs: dict) -> list[str]: ...  # Identical in both
def load_prompts(prompts_dir: Path) -> dict[str, PromptTemplate]: ...  # Shared discovery logic
def load_workflow_config(workflow_dir: Path, variant: str) -> dict: ...  # Variant-based loading
```

#### `git.py` — shared repo management (~50 lines)

```python
def prepare_repo(clone_dir: str, repo_url: str, base_commit: str, instance_id: str) -> str:
    """Clone or reset repo to base commit. Core logic shared by both runners."""
    ...
```

Pro overrides this with its enhanced `_run_git()` wrapper for better error handling.

#### `results.py` — shared JSONL persistence (~30 lines)

```python
def load_existing_results(results_dir: Path) -> tuple[list[ArmResult], set[str]]:
    """Load existing JSONL results for resume. Identical in both runners."""
    ...
```

### Refactored runners

**Ablation** (~200 lines, down from 353):
```python
from examples.swe_bench_common.models import ArmResult
from examples.swe_bench_common.config import topological_sort, load_prompts, load_workflow_config
from examples.swe_bench_common.git import prepare_repo
from examples.swe_bench_common.results import load_existing_results

class ExperimentRunner:
    def run_instance(self, instance, arm) -> ArmResult: ...  # Core run logic
    def run_all(self, arms, ...) -> list[ArmResult]: ...     # Sequential
```

**Pro** (~900 lines, down from 1251):
```python
from examples.swe_bench_common.models import ArmResult
from examples.swe_bench_common.config import topological_sort, load_prompts, load_workflow_config
from examples.swe_bench_common.results import load_existing_results
# Pro overrides prepare_repo with _run_git() wrapper

class SWEBenchProRunner:
    def run_instance(self, instance, arm) -> ArmResult: ...  # Language-aware
    def run_all(self, arms, ...) -> list[ArmResult]: ...     # Parallel + progress
    def _prepare_repo(self, instance) -> str: ...            # Enhanced with _run_git()
    def _get_test_infrastructure(self, ...): ...              # Pro-specific
    def _find_sample_test(self, ...): ...                     # Pro-specific
```

### What stays in each runner (not shared)

**Ablation-specific:**
- `get_generator_registry()`, `get_guard_registry()` — hardcoded ablation generators/guards
- `build_workflow()` — ablation-specific workflow assembly
- `ExperimentRunner.run_instance()` — ablation execution flow
- `ExperimentRunner.run_all()` — sequential execution

**Pro-specific:**
- `_get_generator_registry(lang_config)`, `_get_guard_registry(lang_config)` — language-aware registries
- `build_workflow(config, prompts, lang_config, ...)` — language-aware workflow assembly with Docker guards
- `ProgressTracker`, `ArmStats`, `GuardFailureStats` — progress tracking
- `SWEBenchProRunner._get_test_infrastructure()`, `_find_sample_test()` — test infrastructure extraction
- `SWEBenchProRunner.run_all()` — parallel execution with `ThreadPoolExecutor`

### Implementation order

1. Create `examples/swe_bench_common/` with `__init__.py`
2. Move `ArmResult` to `examples/swe_bench_common/models.py`
3. Extract `topological_sort`, `load_prompts`, `load_workflow_config` to `examples/swe_bench_common/config.py`
4. Extract `prepare_repo` core logic to `examples/swe_bench_common/git.py`
5. Extract `load_existing_results` to `examples/swe_bench_common/results.py`
6. Refactor ablation runner to import from `examples/swe_bench_common/`
7. Refactor pro runner to import from `examples/swe_bench_common/`
8. Remove cross-import from pro → ablation
9. Verify both runners still produce identical results

### Risk

Medium. Touches active experiment code. Both runners must produce identical output after refactoring. Pro's enhanced `_prepare_repo()` (with `_run_git()` wrapper) keeps its own version, importing only the base git logic if useful.

### Acceptance criteria

- No cross-example imports (pro does not import from ablation)
- No duplicated `_topological_sort()`, `_load_existing_results()`, `load_prompts()`
- `ArmResult` defined in one place (`examples/swe_bench_common/models.py`)
- Both runners import shared functions from `examples/swe_bench_common/`
- Estimated ~200 lines of duplication eliminated
- All existing tests pass

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
