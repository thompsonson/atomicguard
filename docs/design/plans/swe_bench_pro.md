# Custom Docker-Based Evaluator for SWE-PolyBench

**Goal:** Replace the broken `swebench.harness.run_evaluation` call with a custom evaluator that uses SWE-PolyBench's own Dockerfile and test_command fields.

**Problem:** `swebench.harness.run_evaluation` fails with `KeyError` because its hardcoded `MAP_REPO_VERSION_TO_SPECS` doesn't include SWE-PolyBench repos. SWE-PolyBench provides its own `Dockerfile`, `test_command`, `F2P`, and `P2P` per instance — we should use those directly.

---

## Files to Modify

### 1. `examples/swe_bench_ablation/dataset.py`

- Add `test_command: str = ""` field to `SWEInstance` dataclass
- Load it from dataset rows: `row.get("test_command", "")`

### 2. `examples/swe_bench_ablation/evaluation.py` (main work)

**Keep:** `prepare_predictions()` unchanged.

**Replace:** `run_evaluation()` and `load_evaluation_results()` with:

#### New dataclass: `EvalResult`

```python
@dataclass
class EvalResult:
    instance_id: str
    resolved: bool
    fail_to_pass_results: dict[str, bool]
    pass_to_pass_results: dict[str, bool]
    error: str | None = None
    log: str = ""
    wall_time_seconds: float = 0.0
```

#### Docker helpers

- `_write_patch_to_container(container, content, path)` — uses `container.put_archive()` with tarfile
- `_exec_in_container(container, cmd, timeout)` — returns `(exit_code, output)`

#### Core: `evaluate_single_instance(instance, model_patch, timeout)`

Per-instance Docker evaluation flow:

1. Write `instance.dockerfile` to temp dir, `client.images.build()`
2. `client.containers.create(image, command="sleep infinity")`, start it
3. Apply `instance.test_patch` via `git apply` inside container
4. Apply `model_patch` via `git apply` inside container
5. Run `instance.test_command` inside container
6. Parse test output with language-specific parser
7. Check F2P (all must pass) and P2P (all must still pass)
8. Cleanup container + image in `finally` block

Returns `EvalResult` — never raises (all errors caught and stored in `error` field).

#### Test output parsers

- `_parse_pytest_output(output)` — Python: look for `PASSED`/`FAILED` lines
- `_parse_java_test_output(output)` — Java: Maven surefire / Gradle patterns
- `_parse_js_test_output(output)` — JS/TS: Jest `PASS`/`FAIL` patterns
- `_parse_generic_output(output)` — Fallback: exit code heuristic
- `_parse_test_results(output, language)` — Dispatch to the above

#### Fuzzy test matching

- `_fuzzy_match_test(test_id, parsed_results)` — Exact match, then normalized (`.` ↔ `::`), then substring
- `_check_resolution(parsed, f2p, p2p)` — Returns `(resolved, f2p_results, p2p_results)`

#### Orchestrator: `run_evaluation()`

New signature:

```python
def run_evaluation(
    predictions_path: str | Path,
    instances: dict[str, SWEInstance],
    max_workers: int = 4,
    timeout_per_instance: int = 600,
    output_path: str | Path | None = None,
    run_id: str = "experiment_7_2",
) -> dict[str, bool]:
```

- Uses `ThreadPoolExecutor(max_workers)` for parallel Docker evaluation
- Early Docker-availability check: `docker.from_env().ping()`
- Writes `resolved.json` (`{instance_id: bool}`) and `eval_details_{run_id}.jsonl`

#### Simplified: `load_evaluation_results()`

Reads the `resolved.json` directly (no more swebench output parsing).

### 3. `examples/swe_bench_ablation/demo.py`

Add `evaluate` CLI command:

```bash
python -m examples.swe_bench_ablation.demo evaluate \
  --predictions output/experiment_7_2/predictions/02_singleshot.jsonl \
  --max-workers 4 --timeout 600 --split test
```

Options: `--predictions` (required), `--output`, `--max-workers`, `--timeout`, `--split`, `--run-id`

Loads dataset, builds `{instance_id: SWEInstance}` lookup, calls `run_evaluation()`, prints summary.

---

## Error Handling

| Failure | Handling | Result |
|---------|----------|--------|
| No Dockerfile / test_command | Skip with warning | `resolved=False` |
| Docker not running | Fail fast, clear message | Abort |
| Image build failure | Catch `BuildError` | `resolved=False` |
| Patch apply failure | Log, return early | `resolved=False` |
| Test timeout | Docker exec timeout | `resolved=False` |
| Unparseable test output | Fall back to exit code | Best-effort |
| Thread crash | Catch in `as_completed` | `resolved=False` |

---

## Full CLI Workflow After Implementation

```bash
# 1. Run experiment (existing)
python -m examples.swe_bench_ablation.demo experiment \
  --model "Qwen/Qwen2.5-Coder-14B-Instruct:featherless-ai" \
  --arms singleshot,s1_direct,s1_tdd --max-instances 5

# 2. Prepare predictions (one-liner)
python -c "
from examples.swe_bench_ablation.evaluation import prepare_predictions
from examples.swe_bench_ablation.analysis import load_results
results = load_results('output/experiment_7_2/results.jsonl')
prepare_predictions(results, 'output/experiment_7_2')
"

# 3. Evaluate each arm (NEW)
python -m examples.swe_bench_ablation.demo evaluate \
  --predictions output/experiment_7_2/predictions/02_singleshot.jsonl
python -m examples.swe_bench_ablation.demo evaluate \
  --predictions output/experiment_7_2/predictions/03_s1_direct.jsonl
python -m examples.swe_bench_ablation.demo evaluate \
  --predictions output/experiment_7_2/predictions/04_s1_tdd.jsonl

# 4. Visualize with real pass/fail data (existing)
python -m examples.swe_bench_ablation.demo visualize \
  --results output/experiment_7_2/results.jsonl \
  --resolved output/experiment_7_2/predictions/resolved.json
```

---

## Verification

1. Syntax check all modified files
2. `evaluate --help` shows in CLI
3. Test with Docker running + the 1-instance results from the smoke test
4. Verify `resolved.json` output format matches what `visualize --resolved` expects
5. Test graceful failure when Docker is not running
