# Plan A: Experiment Runner Enhancements

## Branch

**Work on `feature/informed-backtracking`** (current tip: `581c903`)

```bash
git checkout feature/informed-backtracking
git pull origin feature/informed-backtracking
```

---

## Context

The experiment runner is fully functional. Workflow traces (Extension 10) and guard-specific escalation (Extension 09) are wired into both ablation and pro runners. This plan adds small quality-of-life features to make running experiments smoother.

### Current CLI commands in `demo.py`

| Command | What it does |
|---------|-------------|
| `experiment` | Run arms across instances. Has `--evaluate` to chain eval+viz. |
| `evaluate` | Re-run evaluation on existing predictions |
| `visualize` | Generate charts from results |
| `list-instances` | Show dataset statistics |
| `analyze-errors` | Post-mortem analysis of failed runs |

### What exists

- `ProgressTracker` class in `experiment_runner.py` — logs progress every N runs
- `final_error_analysis.py` — has `EscalationInfo`, `load_trace_escalations()`, escalation summary in report, auto-detects `traces/` directory
- `rich` is already in the examples dependency group (no new deps needed)
- Traces written to `output/traces/<instance_id>/<arm>/` by both runners

---

## Task 1: Shell Wrapper + Environment Template

### 1.1 Create `.env.example`

**File**: `examples/swe_bench_pro/.env.example` (NEW)

```bash
# SWE-Bench Pro Experiment Configuration
# Copy to .env and fill in values:
#   cp .env.example .env

# Required
PROVIDER=openrouter           # ollama, openrouter, huggingface, openai
BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...

# Model
MODEL=moonshotai/kimi-k2-0905

# Experiment
ARMS=s1_tdd                   # Comma-separated: singleshot, s1_direct, s1_tdd, s1_tdd_verified, etc.
OUTPUT_DIR=output/swe_bench_pro
MAX_INSTANCES=0               # 0 = all
MAX_WORKERS=1                 # Parallel workers
LANGUAGE=                     # python, go, javascript, typescript, or empty for all
INSTANCES=                    # Comma-separated instance ID substrings, or empty for all

# Evaluation (requires Docker)
EVALUATE=false                # Set to true to run evaluation after patches
EVAL_MAX_WORKERS=4
```

### 1.2 Create `run_experiment.sh`

**File**: `examples/swe_bench_pro/run_experiment.sh` (NEW, chmod +x)

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# Load .env if it exists
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading config from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "No .env file found at $ENV_FILE"
    echo "Copy .env.example to .env and configure:"
    echo "  cp ${SCRIPT_DIR}/.env.example ${SCRIPT_DIR}/.env"
    exit 1
fi

# Validate required vars
: "${PROVIDER:?PROVIDER must be set in .env}"
: "${LLM_API_KEY:?LLM_API_KEY must be set in .env}"

# Defaults
MODEL="${MODEL:-moonshotai/kimi-k2-0905}"
ARMS="${ARMS:-s1_tdd}"
OUTPUT_DIR="${OUTPUT_DIR:-output/swe_bench_pro}"
MAX_INSTANCES="${MAX_INSTANCES:-0}"
MAX_WORKERS="${MAX_WORKERS:-1}"
EVALUATE="${EVALUATE:-false}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-4}"

# Build command
CMD="uv run python -m examples.swe_bench_pro.demo experiment"
CMD+=" --model $MODEL"
CMD+=" --provider $PROVIDER"
CMD+=" --arms $ARMS"
CMD+=" --output-dir $OUTPUT_DIR"
CMD+=" --max-workers $MAX_WORKERS"
CMD+=" --log-file ${OUTPUT_DIR}/experiment.log"

[[ -n "${BASE_URL:-}" ]] && CMD+=" --base-url $BASE_URL"
[[ -n "${API_KEY:-}" ]] && CMD+=" --api-key $API_KEY"
[[ "$MAX_INSTANCES" != "0" ]] && CMD+=" --max-instances $MAX_INSTANCES"
[[ -n "${LANGUAGE:-}" ]] && CMD+=" --language $LANGUAGE"
[[ -n "${INSTANCES:-}" ]] && CMD+=" --instances $INSTANCES"
[[ "$EVALUATE" == "true" ]] && CMD+=" --evaluate --eval-max-workers $EVAL_MAX_WORKERS"

# Resume if results already exist
[[ -f "${OUTPUT_DIR}/results.jsonl" ]] && CMD+=" --resume"

# Allow extra args
CMD+=" $*"

echo "Running: $CMD"
echo "Output:  $OUTPUT_DIR"
echo "---"
eval "$CMD"

# Post-experiment analysis
echo ""
echo "--- Post-experiment analysis ---"
uv run python -m examples.swe_bench_pro.demo analyze-errors \
    --results "${OUTPUT_DIR}/results.jsonl" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Done. Results in: $OUTPUT_DIR/"
echo "  results.jsonl     - Raw results"
echo "  traces/           - Workflow execution traces"
echo "  artifact_dags/    - Artifact DAGs"
[[ "$EVALUATE" == "true" ]] && echo "  predictions/      - SWE-Bench predictions"
echo "  final_error_analysis_latest.md - Error report"
```

---

## Task 2: Add `--watch` Flag with `rich.live`

### 2.1 Add `--watch` to `experiment` command

**File**: `examples/swe_bench_pro/demo.py`

Add a new Click option to the `experiment` command:

```python
@click.option(
    "--watch",
    is_flag=True,
    help="Show live progress table (requires rich)",
)
```

Add `watch: bool` parameter to the `experiment()` function signature.

### 2.2 Add rich live display to ProgressTracker

**File**: `examples/swe_bench_pro/experiment_runner.py`

The `ProgressTracker` class already collects all the data. Add a method that renders a `rich.table.Table`:

```python
def to_rich_table(self) -> "rich.table.Table":
    """Render current progress as a rich Table."""
    from rich.table import Table

    table = Table(title="Experiment Progress", show_lines=True)
    table.add_column("ARM", style="cyan")
    table.add_column("Progress", justify="right")
    table.add_column("Resolved", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Errors", justify="right", style="yellow")
    table.add_column("Retries", justify="right")
    table.add_column("Avg Time", justify="right")

    with self._lock:
        elapsed = time.time() - self._start_time
        for arm in sorted(self._arm_stats.keys()):
            stats = self._arm_stats[arm]
            total = stats.total
            pct = f"{total}/{self._total_runs}" if self._total_runs else "0"

            guard_failures = sum(stats.failed_by_guard.values())
            avg_time = f"{stats.total_wall_time / total:.0f}s" if total else "-"

            table.add_row(
                arm,
                pct,
                str(stats.eval_resolved),
                str(guard_failures),
                str(stats.errors),
                str(stats.total_retries),
                avg_time,
            )

        # Footer row with totals
        table.add_section()
        total_completed = self._completed
        pct = f"{total_completed}/{self._total_runs}"
        eta = ""
        if total_completed > 0:
            rate = total_completed / elapsed
            remaining = self._total_runs - total_completed
            eta_s = remaining / rate if rate > 0 else 0
            eta = f"ETA: {_format_duration(eta_s)}"
        table.add_row(
            f"TOTAL ({_format_duration(elapsed)})",
            pct,
            "", "", "", "",
            eta,
        )

    return table
```

### 2.3 Wire `--watch` in `run_all()`

Pass a `watch` parameter through to `run_all()`. When enabled, wrap the execution loop with `rich.live.Live`:

In `SWEBenchProRunner.run_all()`, add `watch: bool = False` parameter.

When `watch=True` and `max_workers == 1` (sequential):

```python
if watch:
    from rich.live import Live

    with Live(progress.to_rich_table(), refresh_per_second=1) as live:
        for idx, instance, arm in work_items:
            results.append(_execute_and_record(idx, instance, arm))
            live.update(progress.to_rich_table())
else:
    for idx, instance, arm in work_items:
        results.append(_execute_and_record(idx, instance, arm))
```

When `watch=True` and `max_workers > 1` (parallel), add periodic table refresh:

```python
if watch:
    from rich.live import Live
    import time as _time

    with Live(progress.to_rich_table(), refresh_per_second=0.5) as live:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = { ... }  # same as existing
            while futures:
                # Check for completed futures with short timeout
                done = {f for f in futures if f.done()}
                for future in done:
                    iid, arm = futures.pop(future)
                    try:
                        arm_result = future.result()
                        results.append(arm_result)
                    except Exception:
                        ...
                live.update(progress.to_rich_table())
                if futures:
                    _time.sleep(0.5)
```

### 2.4 Pass `watch` through from demo.py

In `demo.py experiment()`, pass `watch` to `runner.run_all()`:

```python
results = runner.run_all(
    arms=arm_list,
    split=split,
    language=language,
    max_instances=max_instances if max_instances > 0 else None,
    resume_from=output_dir if resume else None,
    max_workers=max_workers,
    instance_filter=instance_filter,
    watch=watch,  # NEW
)
```

---

## Task 3: Add `--traces` to `analyze-errors` Command

### 3.1 Update `analyze-errors` in demo.py

**File**: `examples/swe_bench_pro/demo.py`

Add option to the `analyze-errors` command:

```python
@click.option(
    "--traces/--no-traces",
    default=True,
    help="Include workflow execution traces in report (default: enabled)",
)
def analyze_errors(results: str, output_dir: str | None, traces: bool) -> None:
```

When `traces=False`, pass `trace_dir=None` to skip trace loading (useful for speed on large experiments where you only want the basic report).

Currently `analyze_final_errors()` auto-detects the `traces/` directory. To support `--no-traces`, add a parameter:

**File**: `examples/swe_bench_pro/final_error_analysis.py`

Update `analyze_final_errors()`:

```python
def analyze_final_errors(
    results_path: str | Path,
    output_dir: str | Path | None = None,
    include_traces: bool = True,  # NEW
) -> dict[str, object]:
    ...
    # Extension 10: Look for trace directory
    trace_dir = None
    if include_traces:
        trace_dir = results_path.parent / "traces"
        if not trace_dir.exists():
            trace_dir = None
    ...
```

---

## Testing

```bash
PYTHONPATH=src python -m pytest tests/domain/ tests/application/ tests/infrastructure/ --tb=short
```

Expected: 394 passed, 43 skipped.

**Do NOT run `tests/examples/`** — pre-existing Python 3.12 syntax failures in `examples/base/cli.py` unrelated to this work.

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `examples/swe_bench_pro/.env.example` | CREATE | Documented config template |
| `examples/swe_bench_pro/run_experiment.sh` | CREATE | Unified shell entry point (chmod +x) |
| `examples/swe_bench_pro/demo.py` | MODIFY | Add `--watch` to `experiment`, `--traces` to `analyze-errors` |
| `examples/swe_bench_pro/experiment_runner.py` | MODIFY | Add `to_rich_table()` to ProgressTracker, `watch` param to `run_all()` |
| `examples/swe_bench_pro/final_error_analysis.py` | MODIFY | Add `include_traces` param to `analyze_final_errors()` |

---

## Implementation Order

1. Create `.env.example` and `run_experiment.sh` (Task 1)
2. Add `to_rich_table()` to `ProgressTracker` (Task 2.2)
3. Add `--watch` flag to `demo.py` and wire through `run_all()` (Task 2.1, 2.3, 2.4)
4. Add `--traces` flag to `analyze-errors` command (Task 3)
5. Run tests
6. Commit and push

---

## Commit Message Template

```
feat: add experiment shell wrapper, --watch progress, and --traces flag

- Create .env.example with documented experiment configuration
- Create run_experiment.sh shell wrapper for full pipeline
- Add --watch flag to experiment command for rich live progress table
- Add ProgressTracker.to_rich_table() for real-time display
- Add --traces/--no-traces flag to analyze-errors command
```
