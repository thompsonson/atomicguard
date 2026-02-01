# Review: Custom Docker-Based Evaluator for SWE-PolyBench

**Reviewed:** `docs/design/plans/swe_bench_pro.md`
**Date:** 2026-02-01
**Reviewer:** Claude (automated)

---

## Summary

The plan proposes replacing the broken `swebench.harness.run_evaluation` subprocess call
with a custom Docker-based evaluator that uses SWE-PolyBench's own `Dockerfile` and
`test_command` fields directly. This is the right approach — the current `evaluation.py`
shells out to `swebench.harness.run_evaluation`, which fails with a `KeyError` because its
hardcoded `MAP_REPO_VERSION_TO_SPECS` doesn't cover SWE-PolyBench repos.

**Verdict: Approve with required changes.** The architecture is sound, but there are
correctness, robustness, and performance gaps that should be addressed before
implementation.

---

## What the Plan Gets Right

1. **Correct root cause diagnosis.** The swebench harness is repo-specific and can't handle
   SWE-PolyBench repos. Using the per-instance `Dockerfile` and test metadata directly is
   the correct fix.

2. **`EvalResult` dataclass is well-designed.** Capturing `resolved`, per-test results,
   error, log, and wall time gives sufficient observability. The "never raises" contract is
   good for batch evaluation.

3. **Error handling table is thorough.** Covering image build failures, patch apply failures,
   timeouts, unparseable output, and thread crashes shows the right failure-mode thinking.

4. **Fuzzy test matching is necessary.** SWE-PolyBench test IDs can differ in format from
   actual test runner output (e.g., `::` vs `.` in pytest). Including this from the start
   avoids a class of false negatives.

5. **`prepare_predictions()` kept unchanged.** Correct — it's already working and orthogonal
   to the evaluator change.

6. **CLI integration is clean.** The `evaluate` subcommand fits naturally alongside
   `experiment` and `visualize`, and the option surface is minimal.

---

## Required Changes

### R1. Verify `test_command` exists in SWE-PolyBench schema

The plan adds `test_command: str = ""` to `SWEInstance` and loads it from
`row.get("test_command", "")`. However, the SWE-PolyBench dataset column names are
non-obvious — the loader already handles `"F2P"` vs `"fail_to_pass"` and `"Dockerfile"` vs
`"dockerfile"` (see `dataset.py:73-74,87`). The plan should:

- Confirm the exact column name for the test command (it may be `test_command`, `test_cmd`,
  `eval_script`, or something else).
- Add a fallback lookup similar to the existing Dockerfile loader pattern:
  `row.get("test_command", row.get("test_cmd", ""))`.

If the field doesn't exist at all, the plan needs a different strategy (e.g., inferring the
test command from the language and test framework).

### R2. Address Docker image caching

Building a Docker image per instance from scratch is the dominant cost in SWE-bench
evaluation. Many SWE-PolyBench instances share the same repo and base environment. The plan
doesn't mention caching at all. At minimum:

- Hash the Dockerfile content and reuse existing images with the same hash.
- Consider a two-layer approach: build a base image per repo, then apply per-instance
  patches in a container (cheaper than a full image rebuild).

Without caching, evaluation of even 50 instances could take hours just for image builds.

### R3. `resolved.json` will be overwritten across arms

The CLI workflow (section "Full CLI Workflow After Implementation") runs `evaluate`
separately for each arm, but all write to the same default output path. Step 3 shows:

```bash
python -m ... evaluate --predictions .../02_singleshot.jsonl
python -m ... evaluate --predictions .../03_s1_direct.jsonl
python -m ... evaluate --predictions .../04_s1_tdd.jsonl
```

Each invocation would overwrite the previous `resolved.json` unless:
- The output path defaults to something arm-specific, or
- Results are merged (append) rather than overwritten, or
- The `--output` or `--run-id` flag is used to differentiate.

The plan should specify the default behavior and document it in the CLI workflow example.

### R4. Add container resource limits

The plan creates containers with `client.containers.create(image, command="sleep infinity")`
but specifies no resource limits. A runaway test (e.g., infinite loop, OOM allocation) could
consume all host resources and block other workers. Add:

```python
client.containers.create(
    image,
    command="sleep infinity",
    mem_limit="4g",
    cpu_period=100000,
    cpu_quota=200000,  # 2 CPUs
    network_disabled=True,  # no network for test execution
)
```

At minimum, `mem_limit` and `network_disabled` should be required. The timeout on
`_exec_in_container` handles CPU-bound runaways but not memory-bound ones.

### R5. Specify container working directory handling

Step 3-5 of the evaluation flow apply patches and run commands "inside container," but don't
specify the working directory. The Dockerfile's `WORKDIR` determines where commands execute.
The plan should:

- Document that `_exec_in_container` uses the Dockerfile's `WORKDIR` (or `/` by default).
- Alternatively, extract the repo path from the Dockerfile and pass it explicitly as the
  `workdir` argument to `container.exec_run()`.

Getting this wrong means `git apply` and test commands will fail silently in the wrong
directory.

---

## Recommended Improvements

### I1. Rate-limit concurrent Docker builds

`ThreadPoolExecutor(max_workers=4)` runs 4 evaluations in parallel. If all 4 start building
images simultaneously, the Docker daemon can become resource-starved (especially during
compilation-heavy builds). Consider:

- A semaphore limiting concurrent image builds to 1-2, while allowing more concurrent test
  runs.
- Or sequential builds with a shared image cache, then parallel test execution.

### I2. Add per-instance log files

`EvalResult.log` captures output in memory, but for debugging, writing per-instance log
files (e.g., `eval_logs/{instance_id}.log`) is much more practical than reading a JSONL
file. The `eval_details_{run_id}.jsonl` is good for programmatic access but painful for
manual debugging.

### I3. Docker image tag naming convention

The plan doesn't specify how images are tagged. Suggest:
`atomicguard-eval:{instance_id}` — makes cleanup easy with
`docker rmi $(docker images -q 'atomicguard-eval:*')` and helps with debugging.

### I4. Expand test parser coverage

The plan lists parsers for pytest, Java (Maven/Gradle), and JS (Jest). SWE-PolyBench covers
more languages. Consider:

| Language | Test Framework | Pattern |
|----------|---------------|---------|
| Python   | pytest        | `PASSED`/`FAILED` |
| Java     | Maven Surefire / Gradle | `Tests run:` / `BUILD SUCCESSFUL` |
| JS/TS    | Jest          | `Tests:` summary line |
| Go       | `go test`     | `--- PASS` / `--- FAIL` |
| Rust     | `cargo test`  | `test result:` summary line |
| C/C++    | CTest/gtest   | `[  PASSED  ]` / `[  FAILED  ]` |

The generic fallback (exit code) is fine for v1, but the plan should note which languages
are fully supported vs best-effort.

### I5. Clarify patch application order

Steps 3 and 4 apply `test_patch` then `model_patch`. This is correct for SWE-bench
semantics (test_patch adds the regression tests, model_patch is the candidate fix), but the
plan should state *why* this order matters. If reversed, the test_patch might conflict with
the model_patch because both modify similar regions.

### I6. Add a cleanup / prune CLI command

Over multiple evaluation runs, Docker images and stopped containers accumulate. Consider
adding an `evaluate --cleanup` flag or separate `cleanup` CLI command that removes all
`atomicguard-eval:*` images and exited containers.

### I7. Thread safety of output files

The `ThreadPoolExecutor` workers produce `EvalResult` objects. The plan says results are
written to `resolved.json` and `eval_details_{run_id}.jsonl`, but doesn't specify when. If
writing incrementally (per-completion), a lock is needed for the JSONL file. If writing at
the end (after all futures complete), this is simpler and sufficient given the data size.
Recommend the latter approach and state it explicitly.

---

## Structural Observations

### Scope is appropriate

The plan modifies 3 files (`dataset.py`, `evaluation.py`, `demo.py`) with the bulk of new
code in `evaluation.py`. This is well-contained and doesn't touch the core framework.

### No tests planned

The plan's "Verification" section lists manual checks (syntax check, `--help`, smoke test)
but no automated tests. Given the evaluation module is the only way to measure experiment
results, it should have at minimum:

- A unit test for each test output parser with sample output strings.
- A unit test for fuzzy matching logic.
- A unit test for `_check_resolution` with known F2P/P2P lists.
- An integration test that builds a trivial Docker image and runs a simple command.

These don't require SWE-PolyBench data and can run in CI.

### `SWEInstance` is frozen

The dataclass is `frozen=True`, so adding `test_command` is a non-breaking change (just a
new field with a default). Good.

### Backward compatibility of `run_evaluation`

The new `run_evaluation` signature is completely different from the current one (which takes
`dataset_name` and `split`). Since the current version is broken anyway, this is fine, but
any code calling the old signature will break at import time rather than runtime. The plan
should note this is an intentional breaking change.

---

## Questions for the Author

1. Has the SWE-PolyBench dataset schema been verified for a `test_command` field? What
   is the exact column name and typical content (e.g., `pytest tests/`, `mvn test`,
   `npm test`)?

2. What is the expected evaluation throughput? Building Docker images is slow —
   is the 600s per-instance timeout sufficient for image build + patch + test?

3. Should the evaluator support running on pre-built images (for cases where the image
   was already built in a previous run)?

4. Is `network_disabled=True` acceptable for all test commands, or do some SWE-PolyBench
   instances require network access during testing?

5. The plan doesn't mention the `pass_to_pass` (P2P) check granularity. If a model patch
   fixes all F2P tests but breaks 1 of 50 P2P tests, is that `resolved=False`? (Standard
   SWE-bench says yes, but this should be stated.)
