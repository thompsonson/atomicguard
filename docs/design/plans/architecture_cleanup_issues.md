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
