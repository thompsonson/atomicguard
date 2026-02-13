# Qwen3-Coder-Next TDD-Verified Failure Analysis

**Date**: 2026-02-08
**Model**: `qwen/qwen3-coder-next`
**Arm**: `s1_tdd_verified`
**Instances**: 10 (SWE-Bench Pro Python)

## TDD Pipeline Clarification

The pipeline has these stages:

```
AnalysisGuard → TestSyntaxGuard → TestRedGuard → PatchGuard → TestGreenGuard → SWE-Bench Eval
                                       ↑                           ↑
                              (test FAILS on bug)        (test PASSES after patch)
```

**TestGreenGuard is NOT the final SWE-Bench evaluation**. It runs the LLM-generated test on patched code to verify the fix works before running the official evaluation.

## Final Results

| Metric | Value |
|--------|-------|
| **Total runs** | 10 |
| **Resolved** | 0 (0%) |
| **Total time** | 1h 44m (6235s) |
| **Total tokens** | 2.1M |
| **Avg time/instance** | 10.4 min |
| **Total retries** | 70 |

**Failure Breakdown:**

| Guard | Failures | Percentage |
|-------|----------|------------|
| TestGreenGuard | 7 | 70% |
| PatchGuard | 2 | 20% |
| FullEvalGuard | 1 | 10% |

## Failure Categories

### 1. PatchGuard Failures (1 instance)

The model repeatedly generates search strings that don't match the actual file content:

```
You searched for:
'    def run(self, terms, variables=None, **kwargs):\n        # Get the use_netrc option...'

Found exact content at line 209:
     209:     def run(self, terms, variables=None, **kwargs):
     210:
     211:         self.set_options(var_options=variables, direct=kwargs)
```

**Observation**: Even WITH grounding feedback showing the exact content, the model continues to hallucinate code that doesn't exist.

### 2. TestGreenGuard Failures (5 instances)

Two sub-categories:

#### 2a. Test Environment Issues

Example from qutebrowser:
```
E   UserWarning: PyQt5 already imported
FAILED test_atomicguard.py::test_hide_qt_warning_filters_messages
```

The LLM-generated test has import order issues that cause test infrastructure warnings. This is a **test generation quality issue**, not a patch issue.

#### 2b. Incorrect Semantic Patches

The patch applies cleanly but doesn't actually fix the bug. Example:
```json
{
  "patch": "... from typing import Iterator, Optional, Callable, cast, Any",
  "reasoning": "The test code shows usage of `Any` type..."
}
```

The model is addressing superficial issues (missing imports) rather than the actual bug.

## Feedback History Pattern

Typical failure progression in `ap_gen_patch`:

1. **Attempt 1-2**: "No patch or edits found in output" - model doesn't generate valid JSON
2. **Attempt 3**: "All edits have identical search and replace strings" - no-op patch
3. **Attempt 4-6**: "No patch or edits found in output" - continues to fail
4. **Attempt 7**: Finally generates a patch, but it's semantically wrong

## Key Insights

1. **Grounding is working** - PatchGuard shows exact line content when search fails
2. **Model ignores grounding** - Despite "Use the EXACT content shown above", model continues to hallucinate
3. **Test quality affects results** - TestGreenGuard failures may be test issues, not patch issues
4. **Semantic correctness is the bottleneck** - Getting patches to apply is solved; getting patches that fix bugs is not

## Critical Insight: Linting Before Testing

Many TestGreenGuard failures are actually **static analysis problems**, not test problems:

```
NameError: name 'IntEnum' is not defined
```

This error from the 23-edit, 476-line patch could be caught in **<1 second** by linting, instead of **10-60 seconds** in Docker.

### Comparison

| Method | Time | Catches |
|--------|------|---------|
| `python -m py_compile file.py` | ~0.1s | Syntax errors |
| `python -c "import module"` | ~0.5s | Import errors, undefined names |
| `ruff check file.py` | ~0.2s | Undefined names, unused imports |
| Docker test run (current) | 10-60s | Everything |

### Proposed Solution: LintGuard

Add a fast pre-flight check between PatchGuard and TestGreenGuard:

```
PatchGuard → LintGuard → TestGreenGuard
             (fast)       (slow, Docker)
```

Benefits:
1. **Save time** - catch 80%+ of broken patches in <1s
2. **Better feedback** - "missing import" vs "test failed"
3. **Reduce Docker overhead** - only test compilable patches

## Recommendations

1. **Add LintGuard** - Fast static analysis before expensive Docker tests
2. **Test generation needs improvement** - Tests should be isolated from import order issues
3. **Consider patch complexity** - Simple models may not understand complex bugs
4. **Add semantic validation** - Pre-screen patches for likelihood of addressing the bug
5. **Model capability** - qwen3-coder-next may not be suited for complex multi-file fixes

## Artifacts Location

- Results: `output/swe_bench_pro_qwen3_verify_fixes/results.jsonl`
- Artifact DAGs: `output/swe_bench_pro_qwen3_verify_fixes/artifact_dags/`
- Full logs: `/tmp/claude-1000/-home-mt-Projects-thompsonson-atomicguard/tasks/bee775b.output`
