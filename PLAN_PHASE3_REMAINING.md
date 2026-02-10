# Plan: Remaining Phase 3 Work — Experiment Runner Validation

## Branch

**Work on `feature/informed-backtracking`** (current tip: `581c903`)

All core implementation (Phases 1A, 1B, 2, and Phase 3 wiring) is COMPLETE.

```bash
git checkout feature/informed-backtracking
git pull origin feature/informed-backtracking
```

---

## What's Already Implemented

Everything below is done and committed on `feature/informed-backtracking`:

### Core Library (Phases 1A + 1B + 2)

| Component | File | Status |
|-----------|------|--------|
| `StagnationDetected.stagnant_guard` | `src/atomicguard/domain/exceptions.py` | Done |
| `FeedbackSummarizer.detect_stagnation_by_guard()` | `src/atomicguard/application/feedback_summarizer.py` | Done |
| `StagnationInfo.stagnant_guard` + `approaches_tried: tuple` | `src/atomicguard/application/feedback_summarizer.py` | Done |
| `DualStateAgent.escalation_by_guard` | `src/atomicguard/application/agent.py` | Done |
| `WorkflowStep.escalation_by_guard` + `Workflow.add_step()` | `src/atomicguard/application/workflow.py` | Done |
| `Workflow.__init__(event_store=...)` + event emission | `src/atomicguard/application/workflow.py` | Done |
| `WorkflowState.get_satisfied_guards()` + `unsatisfy()` | `src/atomicguard/domain/models.py` | Done |
| `escalation_by_guard` in schema | `src/atomicguard/schemas/workflow.schema.json` | Done |
| `WorkflowEventType`, `WorkflowEvent`, `EscalationEventRecord` | `src/atomicguard/domain/workflow_event.py` | Done |
| `WorkflowEventStoreInterface` | `src/atomicguard/domain/interfaces.py` | Done |
| `InMemoryWorkflowEventStore`, `FilesystemWorkflowEventStore` | `src/atomicguard/infrastructure/persistence/workflow_events.py` | Done |
| `WorkflowEventEmitter` | `src/atomicguard/application/workflow_event_emitter.py` | Done |
| Trace report CLI | `scripts/trace_report.py` | Done |
| BFS uses `deque` in `_get_transitive_dependents()` | `src/atomicguard/application/workflow.py` | Done |

### Experiment Runner Wiring (Phase 3 partial)

| Component | File | Status |
|-----------|------|--------|
| `build_workflow()` accepts `event_store` + `escalation_by_guard` | `examples/swe_bench_ablation/demo.py` | Done |
| `run_instance()` creates `FilesystemWorkflowEventStore` | `examples/swe_bench_ablation/experiment_runner.py` | Done |
| `build_workflow()` accepts `event_store` + `escalation_by_guard` | `examples/swe_bench_pro/experiment_runner.py` | Done |
| `run_instance()` creates `FilesystemWorkflowEventStore` | `examples/swe_bench_pro/experiment_runner.py` | Done |
| `escalation_by_guard` in workflow config | `examples/swe_bench_ablation/workflows/05_s1_tdd_verified.json` | Done |
| `EscalationInfo` + `load_trace_escalations()` + escalation summary | `examples/swe_bench_pro/final_error_analysis.py` | Done |

### Tests

| Test File | Status |
|-----------|--------|
| `tests/application/test_feedback_summarizer.py` | Done (includes `TestDetectStagnationByGuard`) |
| `tests/application/test_workflow_escalation.py` | Done (includes `TestGuardSpecificEscalation`) |
| `tests/domain/test_workflow_event.py` | Done |
| `tests/application/test_workflow_event_emitter.py` | Done |
| `tests/infrastructure/test_workflow_event_store.py` | Done |
| `tests/examples/test_swe_bench_pro.py` | Done (updated for current API) |

**All 394 core tests pass** (`tests/domain/`, `tests/application/`, `tests/infrastructure/`).

**Note:** `tests/examples/` has pre-existing failures due to `examples/base/cli.py` using Python 3.12+ syntax (PEP 695 type params: `def common_options[F: Callable]`). This is NOT caused by our changes — it's an environment issue (Python 3.11). Tests that don't import from `examples.base` pass fine.

---

## What Remains (Phase 3 completion)

Two optional features from the original plan were not implemented. These are convenience/UX features for live experiment monitoring — NOT required for running experiments.

### Task 1: Add `--traces` flag to final_error_analysis.py

**File**: `examples/swe_bench_pro/final_error_analysis.py`

The trace loading infrastructure (`EscalationInfo`, `load_trace_escalations()`) is already in the file. What's missing is a CLI flag to opt into trace analysis from the command line.

**Current CLI** (bottom of the file):
```python
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m examples.swe_bench_pro.final_error_analysis <results.jsonl>")
        sys.exit(1)
    results_path = sys.argv[1]
    summary = analyze_final_errors(results_path)
    print(f"\nAnalysis complete: {summary}")
```

**Change**: Replace with argparse to add `--traces` flag:
```python
if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Final error analysis for experiments")
    parser.add_argument("results_path", help="Path to results.jsonl")
    parser.add_argument("--traces", action="store_true", help="Include workflow execution traces in report")
    args = parser.parse_args()

    summary = analyze_final_errors(args.results_path)
    print(f"\nAnalysis complete: {summary}")
```

Note: `analyze_final_errors()` already auto-detects the `traces/` directory and includes trace data when it exists — the `--traces` flag is informational at this point. No functional change needed in the function itself.

### Task 2: Add `--watch` live progress mode

**File**: `examples/swe_bench_pro/final_error_analysis.py`

Add a function for live monitoring during long experiment runs:

```python
def watch_progress(
    results_path: str | Path,
    refresh_interval: int = 30,
) -> None:
    """Live progress monitoring for running experiments.

    Watches results.jsonl and prints updates as new results appear.

    Args:
        results_path: Path to results.jsonl
        refresh_interval: Seconds between checks
    """
    import time

    results_path = Path(results_path)
    last_count = 0

    print(f"Watching {results_path} (Ctrl+C to stop)")
    print("-" * 60)

    try:
        while True:
            if results_path.exists():
                overview = compute_experiment_overview(results_path)
                if overview.total_runs != last_count:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    success_pct = (
                        overview.successful_runs / overview.total_runs * 100
                        if overview.total_runs > 0 else 0
                    )
                    print(
                        f"[{timestamp}] {overview.total_runs} runs: "
                        f"{overview.successful_runs} success ({success_pct:.0f}%), "
                        f"{overview.failed_runs} failed"
                    )
                    last_count = overview.total_runs
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\nStopped.")
```

Add `--watch` and `--interval` to the CLI argparse block:

```python
    parser.add_argument("--watch", action="store_true", help="Live progress monitoring mode")
    parser.add_argument("--interval", type=int, default=30, help="Watch refresh interval in seconds")
    args = parser.parse_args()

    if args.watch:
        watch_progress(args.results_path, args.interval)
    else:
        summary = analyze_final_errors(args.results_path)
        print(f"\nAnalysis complete: {summary}")
```

---

## Testing

Run test command:
```bash
PYTHONPATH=src python -m pytest tests/domain/ tests/application/ tests/infrastructure/ --tb=short
```

Expected: 394 passed, 43 skipped.

**Do NOT run `tests/examples/`** — those have pre-existing Python 3.12 syntax failures unrelated to this work.

---

## Implementation Order

1. Update CLI in `final_error_analysis.py` to use argparse with `--traces` flag
2. Add `watch_progress()` function
3. Add `--watch` and `--interval` to argparse
4. Run tests to verify no regressions
5. Commit and push to `feature/informed-backtracking`

---

## Commit Message Template

```
feat: add --watch and --traces CLI flags to final_error_analysis

- Replace sys.argv CLI with argparse for proper flag handling
- Add --traces flag (trace analysis auto-detects traces/ directory)
- Add --watch mode for live progress monitoring during experiments
- Add --interval flag to control watch refresh rate (default 30s)
```
