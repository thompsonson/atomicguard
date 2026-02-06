# AtomicGuard Experimental Design: Guard-Driven Synthesis on SWE-bench

**Version:** 0.2 — Working Document
**Date:** February 6, 2026
**Branch:** `feature/composite-guard-tdd`

---

## 1. Research Question

Can progressive decomposition of code repair tasks into Atomic Action Pairs — each with deterministic guards — systematically improve resolve rates over single-shot generation, and does execution-verified guard composition (CompositeGuard) close the gap between syntactic validation and semantic correctness?

### Sub-questions

1. **Decomposition value**: Does separating comprehension (analysis) from generation (patch) improve outcomes?
2. **Guard-Driven Synthesis**: Does synthesising a test *before* the fix constrain the solution space and improve resolve rates?
3. **Execution verification**: Does running tests inside the target Docker environment (red/green guards) catch failures that static guards miss?
4. **Specification richness**: Does the presence of human-augmented requirements (SWE-bench Pro) interact differently with guard-driven strategies than raw issue text (SWE-bench Verified)?

---

## 2. Benchmark Selection

### 2.1 SWE-bench Pro (Primary Evaluation)

| Property | Value |
|----------|-------|
| Instances | 731 (public set) |
| Languages | Python, Go, JavaScript, TypeScript |
| Repositories | 41 (consumer apps, B2B, dev tools) |
| Avg patch size | 107.4 lines across 4.1 files |
| Baseline resolve | ~23% capped / ~45% uncapped (frontier + SWE-Agent) |
| Contamination | Low (GPL copyleft, legal deterrent) |

**Agent receives**: `problem_statement` + `requirements` + `interface` (human-augmented)
**Agent does NOT receive**: gold `patch`, `test_patch`, `fail_to_pass`/`pass_to_pass` test lists

**Why primary**: Multi-language validates framework is language-agnostic. Multi-file patches stress dependency-aware caching. Human-augmented `requirements` + `interface` fields map directly to Guard-Driven Synthesis — they provide explicit specification material that the test generator can consume. Low baseline leaves room to demonstrate architectural improvement.

### 2.2 SWE-bench Verified (Reference Comparison)

| Property | Value |
|----------|-------|
| Instances | 500 |
| Languages | Python only |
| Repositories | 12 popular Python libraries |
| Baseline resolve | ~70%+ (frontier, saturating) |
| Contamination | High (permissive licenses, heavily trained on) |

**Agent receives**: `problem_statement` + optional `hints_text` (raw issue comments)
**No**: `requirements` or `interface` fields

**Why reference**: Established benchmark everyone reports against. Saturation demonstrates that on well-specified, contaminated, single-language tasks, raw model capability approaches sufficiency — architectural contribution matters less. Contrast with Pro results shows where scaffolding provides value.

### 2.3 Key Structural Differences

| Dimension | SWE-bench Verified | SWE-bench Pro |
|-----------|-------------------|---------------|
| Specification quality | Raw GitHub issues | Human-augmented requirements + interface |
| Specification source | `problem_statement` + `hints_text` | `problem_statement` + `requirements` + `interface` |
| Task complexity | Often single-file | Avg 4.1 files, 107 lines |
| Bootstrap problem | Pure unknown spec | Partially specified (requirements exist) |
| Guard-Driven fit | Agent must discover spec | Agent has explicit spec to synthesise guards from |
| Evaluation | Hidden test suite | Hidden test suite (same mechanism) |

**Implication for experimental design**: On Verified, the test generator must infer specification from ambiguous issue text. On Pro, it has structured requirements to constrain generation. We expect Guard-Driven Synthesis to show larger relative improvement on Pro because the specification material is richer.

---

## 3. Experimental Arms — Current Design

All arms share a common pipeline: the experiment runner appends a repository file listing (filtered by language, capped at 80 files) to the problem statement before workflow execution. All generators/guards see real file paths from the checked-out repo.

**Global settings** (apply to all arms unless overridden):
- `require_git_apply: true` — patches must apply cleanly via `git apply`
- `require_syntax_valid: true` — generated code must parse in the target language

### 3.1 Arms Overview

```
Arm 01  baseline         localize -> patch                    Deterministic guards only
Arm 02  singleshot       patch                               Single action pair, no decomposition
Arm 03  s1_direct        analysis -> patch                    Comprehension before generation
Arm 04  s1_tdd           analysis -> test -> fix               Guard-Driven Synthesis (static guards)
Arm 05  s1_tdd_verified  analysis -> test -> fix               Guard-Driven Synthesis (CompositeGuard)
Arm 06  s1_tdd_behavior  analysis -> test -> fix               Prompt variant of Arm 05
```

### 3.2 Arm Specifications

#### Arm 01: Baseline (localize -> patch)

**Workflow**: `ap_localize` -> `ap_patch`
**rmax**: 3
**Thesis**: File-finding helps, but without comprehension, patch quality is limited.

```
+---------------+     +-----------+
| ap_localize   |---->| ap_patch  |
|               |     |           |
| G: localization     | G: patch  |
|  (>=1 file,   |     |  (git     |
|   <=5 files)  |     |   apply + |
|               |     |   syntax) |
+---------------+     +-----------+
```

**Guards**: Both deterministic. Localization guard validates file count constraints. Patch guard validates clean application and syntax.

#### Arm 02: Singleshot (patch only)

**Workflow**: `ap_singleshot`
**rmax**: 3
**Thesis**: Direct problem-to-patch. No decomposition. The "Code Promise Only" baseline from the article.

```
+------------------+
|  ap_singleshot   |
|                  |
|  G: patch        |
|   (git apply +   |
|    syntax)       |
+------------------+
```

**Guards**: Deterministic only. Catches syntactic failures but provides no semantic feedback on retry.

#### Arm 03: S1 Direct (analysis -> patch)

**Workflow**: `ap_analysis` -> `ap_patch`
**rmax**: 3
**Thesis**: Separating comprehension from generation prevents hallucinated file contents. The analysis artifact provides structured context (bug type, root cause, affected files, fix approach) that constrains patch generation.

```
+--------------+     +-----------+
| ap_analysis  |---->| ap_patch  |
|              |     |           |
| G: analysis  |     | G: patch  |
|  (schema     |     |  (git     |
|   valid)     |     |   apply + |
|              |     |   syntax) |
+--------------+     +-----------+
```

**Guards**: Analysis guard validates JSON schema (bug_type, root_cause, affected_files, fix_approach, confidence). Patch guard is deterministic.

#### Arm 04: S1 TDD (analysis -> test -> fix, static guards)

**Workflow**: `ap_analysis` -> `ap_gen_test` -> `ap_gen_fix`
**rmax**: 3
**Thesis**: Guard-Driven Synthesis — synthesise the verification suite before the implementation. The test constrains the patch from "generate anything" to "satisfy these assertions."

```
+--------------+     +--------------+     +--------------+
| ap_analysis  |---->| ap_gen_test  |---->| ap_gen_fix   |
|              |     |              |     |              |
| G: analysis  |     | G: test_     |     | G: patch     |
|              |     |    syntax    |     |  (git apply  |
|              |     |              |     |   + syntax)  |
+--------------+     +--------------+     +--------------+
```

**Guards**: All static/deterministic. Test syntax guard validates the generated test parses correctly. But it does NOT verify the test actually fails on buggy code or passes after the fix. This is the "structural promise only" tier.

**Limitation**: A syntactically valid test that always passes (existence checks, type checks) will be accepted — the test_syntax guard cannot detect this.

#### Arm 05: S1 TDD Verified (analysis -> test -> fix, CompositeGuard)

**Workflow**: `ap_analysis` -> `ap_gen_test` -> `ap_gen_patch`
**rmax**: 6
**Thesis**: Full red-green-refactor TDD loop with execution verification. The CompositeGuard chains enforce that:
1. The generated test **fails on buggy code** (red phase — confirms test discriminates)
2. The generated patch **makes the test pass** (green phase — confirms fix works)
3. The patch **passes the full evaluation suite** (regression — confirms no breakage)

```
+--------------+     +-----------------------+     +----------------------------------+
| ap_analysis  |---->|    ap_gen_test         |---->|       ap_gen_patch               |
|              |     |                        |     |                                  |
| G: analysis  |     | G: composite           |     | G: composite                     |
|              |     |   +- test_syntax (Gval)|     |   +- patch (Gval)               |
|              |     |   +- test_red   (Gver)|     |   +- test_green (Gver)           |
|              |     |      [Docker exec]     |     |   +- full_eval  (Gver)           |
|              |     |                        |     |      [Docker exec]                |
+--------------+     +-----------------------+     +----------------------------------+
```

**Guard chain — ap_gen_test**:
- `test_syntax` (G_val): Structural promise — "I produced valid test code"
- `test_red` (G_ver): Semantic promise — "My test fails on the buggy code." Runs inside the instance's Docker container against `base_commit`. If the test passes on buggy code, it's a false-positive test and gets rejected with feedback.

**Guard chain — ap_gen_patch**:
- `patch` (G_val): Structural promise — "I produced a clean diff"
- `test_green` (G_ver): Semantic promise — "My patch makes the test pass." Applies patch + runs generated test inside Docker.
- `full_eval` (G_ver): Regression promise — "Existing tests still pass." Runs full evaluation suite inside Docker.

**rmax = 6**: Doubled from arms 01-04 because execution-verified guards provide actionable feedback. A failed `test_red` tells the model "your test doesn't discriminate — it passes on buggy code" which enables meaningful retry, unlike a generic syntax error.

#### Arm 06: S1 TDD Behavior (prompt variant of Arm 05)

**Workflow**: Identical structure to Arm 05
**rmax**: 6
**Thesis**: A/B test of prompt design. The analysis prompt explicitly extracts "expected behavior", "actual behavior", and "trigger inputs" — framing the test generation as behavior specification rather than code-level assertion.

Same CompositeGuard chains as Arm 05. Only the prompt templates differ.

### 3.3 Progressive Decomposition Matrix

| Property | 01 | 02 | 03 | 04 | 05 | 06 |
|----------|-----|-----|-----|-----|-----|-----|
| Action pairs | 2 | 1 | 2 | 3 | 3 | 3 |
| Comprehension step | - | - | Y | Y | Y | Y |
| Test synthesis | - | - | - | Y | Y | Y |
| Static guards only | Y | Y | Y | Y | - | - |
| Execution guards | - | - | - | - | Y | Y |
| Behavior prompts | — | — | — | - | - | Y |
| rmax | 3 | 3 | 3 | 3 | 6 | 6 |
| require_git_apply | Y | Y | Y | Y | Y | Y |

---

## 4. Guard Taxonomy

### 4.1 Two-Phase Validation (Article Section 3)

Every guard operates in one of two phases:

**Phase 1 — Validation Guard (G_val)**: Structural promises. Computationally cheap (O(N) parse). Can run without Docker.

**Phase 2 — Verification Guard (G_ver)**: Semantic promises. Require execution inside the target environment (Docker). May be expensive.

### 4.2 Guard Catalogue

| Guard ID | Phase | Input | Validation | Cost |
|----------|-------|-------|------------|------|
| `localization` | G_val | JSON | File count within bounds, files exist in repo | O(1) |
| `analysis` | G_val | JSON | Schema valid (bug_type, root_cause, affected_files, fix_approach, confidence) | O(1) |
| `test_syntax` | G_val | Code | Parses in target language; contains test functions/assertions | O(N) |
| `patch` | G_val | Diff | `git apply` succeeds; patched files parse; no empty hunks | O(N) |
| `test_red` | G_ver | Code + Docker | Test *fails* on `base_commit` (buggy code). Confirms test discriminates. | O(exec) |
| `test_green` | G_ver | Diff + Code + Docker | Test *passes* after applying patch. Confirms fix satisfies test. | O(exec) |
| `full_eval` | G_ver | Diff + Docker | Full `fail_to_pass` + `pass_to_pass` suite passes. Regression check. | O(eval) |

### 4.3 CompositeGuard Semantics

A CompositeGuard is an ordered sequence of guards applied to a single action pair. Evaluation is short-circuit: the first guard failure triggers a retry with that guard's feedback. This is not merely "run all checks" — the ordering encodes the validation -> verification priority.

```python
class CompositeGuard:
    guards: list[Guard]  # Ordered: G_val guards first, G_ver guards last

    def evaluate(self, artifact, context) -> GuardResult:
        for guard in self.guards:
            result = guard.evaluate(artifact, context)
            if result.verdict != SUCCESS:
                return result  # Short-circuit with this guard's feedback
        return GuardResult(SUCCESS)
```

**Design principle**: Cheap structural checks (G_val) run before expensive execution checks (G_ver). This avoids wasting Docker execution on syntactically invalid artifacts.

**Retry feedback**: Each guard in the chain provides specific, actionable feedback:
- `test_syntax` failure: "SyntaxError at line 42: unexpected indent"
- `test_red` failure: "Test passed on buggy code — your test doesn't discriminate. The test must FAIL on the unmodified code."
- `patch` failure: "Search string not found in src/user/email.js — exact match required"
- `test_green` failure: "Test still fails after patch: AssertionError: expected True, got False"
- `full_eval` failure: "3 regression failures in pass_to_pass tests: test_user_login, test_admin_panel, test_api_auth"

---

## 5. Design Space for Future Arms

The current six arms explore one axis: progressive decomposition with escalating guard rigour. Several additional axes remain open, grouped below by the capability they test.

### 5.1 Localization-Augmented Arms

Current arms 02-06 give the model a repository file listing but no file *contents*. The model must infer which files contain the bug from the listing + problem statement. This is a known failure mode (see EXAMPLE_RUN.md: singleshot hallucinated file contents for all 4 attempts).

**Arm 07: S1 Direct + Localization**
```
ap_localize -> ap_analysis -> ap_patch
```
Add a localization step that identifies relevant files *before* analysis. The analysis generator receives file contents, not just paths. Tests whether grounding the comprehension step in actual code improves patch quality.

**Arm 08: S1 TDD + Localization (Verified)**
```
ap_localize -> ap_analysis -> ap_gen_test -> ap_gen_patch
```
Full Guard-Driven Synthesis with localization-informed analysis. The test generator sees the actual code that must change, enabling more targeted test construction. CompositeGuards on test (syntax + red) and patch (patch + green + full_eval).

**Arm 09: S1 TDD + Oracle Localization (Upper Bound)**
```
ap_analysis(+gold files) -> ap_gen_test -> ap_gen_patch
```
Uses the gold-patch file list as an oracle to provide perfect localization. Measures the ceiling: how much of the remaining failure is due to localization errors vs. comprehension/generation errors?

### 5.2 Specification-Enriched Arms (Pro-specific)

SWE-bench Pro provides `requirements` and `interface` fields that SWE-bench Verified lacks. These arms test whether explicitly injecting structured specification material into the test generator improves Guard-Driven Synthesis.

**Arm 10: S1 TDD + Spec-Aware Test (Verified)**
```
ap_analysis -> ap_gen_test(requirements+interface) -> ap_gen_patch
```
The test generator's prompt includes the `requirements` and `interface` fields verbatim. Tests whether richer specification material produces better discriminating tests.

**Arm 11: S1 TDD + Spec-Decomposed**
```
ap_analysis -> ap_spec_extract -> ap_gen_test -> ap_gen_patch
```
A new `ap_spec_extract` action pair takes the analysis and `requirements` fields and produces a structured test specification (preconditions, postconditions, invariants) before the test generator runs. The spec_extract guard validates the specification is internally consistent.

### 5.3 Iterative Repair Arms

Current arms treat each action pair as independent: if ap_gen_patch fails all retries, the workflow fails. These arms add cross-pair iteration.

**Arm 12: S1 TDD + Patch Iteration**
```
ap_analysis -> ap_gen_test -> ap_gen_patch -> [on full_eval fail] -> ap_refine_patch
```
If `full_eval` fails but `test_green` passed, the patch satisfies the generated test but breaks something else. A new `ap_refine_patch` action pair receives the regression failure details and the current patch, attempting a targeted fix. This is the "red loop" from the article applied at the workflow level.

**Arm 13: S1 TDD + Test-Patch Iteration**
```
ap_analysis -> ap_gen_test -> ap_gen_patch -> [on test_green fail] -> ap_refine_test -> ap_gen_patch
```
If the patch can't pass the generated test after rmax attempts, the test itself may be wrong (testing the wrong behavior, asserting an incorrect expected value). Backtrack to refine the test, then retry the patch. Tests whether controlled backtracking improves outcomes or just wastes budget.

### 5.4 Multi-Hypothesis Arms

Rather than generating one analysis and committing to it, generate multiple competing hypotheses and select the best.

**Arm 14: SC-Analysis + TDD**
```
ap_analysis xN -> select(consensus) -> ap_gen_test -> ap_gen_patch
```
Self-Consistency applied to the analysis step: generate N analyses, cluster by root cause hypothesis, select the consensus. Tests whether SC-Planning (from the article's Section 7.3) improves diagnostic accuracy.

**Arm 15: Multi-Test + Patch**
```
ap_analysis -> ap_gen_test xN -> select(best_discriminator) -> ap_gen_patch
```
Generate N candidate tests, run all through `test_red`, select the test with the strongest failure signal (e.g., most assertion failures, clearest error message). Tests whether test diversity improves constraint quality.

### 5.5 Language-Specific Arms

SWE-bench Pro's multi-language corpus enables language-stratified analysis.

**Arm 16: Language-Adaptive Guards**
```
Same as Arm 05, but guard_config varies by repo_language:
  Python: pytest runner, ast.parse validation
  Go:     go test runner, go vet validation
  JS/TS:  jest/mocha runner, esbuild/tsc validation
```
Tests whether language-specific guard implementations (beyond the current multi-lang subclass approach) capture more failures.

### 5.6 Feedback-Classified Backtracking Arms

Arms 01-16 are all *forward-only*: if a step exhausts its retry budget, the workflow fails. The framework's `DualStateAgent` retries within a single action pair, but there is no cross-pair feedback loop. The planning search design (see `docs/design/plans/plan_search_feedback_loop.md`) formalises this gap: guard feedback carries signal about *where in the pipeline* the root cause lies, making it a heuristic for search, not just a pass/fail gate.

The key insight is that **guards are not just validators — they are heuristic functions for the planning search**. A `test_green` failure saying "AssertionError: expected 200, got 404" tells you the patch is wrong (retry same step). But a `test_red` failure saying "ImportError: No module named 'foobar'" after 3 retries tells you the analysis misidentified the affected module (backtrack to analysis). The feedback *classifies* the appropriate backtrack depth.

This requires a new action pair and a new orchestration pattern.

#### New Action Pair: `ap_diff_review`

`ap_diff_review` is a *critique* action pair that reviews the generated patch as a code reviewer would. It receives the patch diff, the analysis, and the generated test as dependencies. Its output is a structured review: issues found (critical/minor), confidence that the patch is correct, and — crucially — a *failure classification* when issues are found.

```
Input:   patch diff + analysis + test (as dependencies)
Output:  { "verdict": "approve" | "revise" | "backtrack",
           "issues": [...],
           "backtrack_target": null | "ap_gen_patch" | "ap_gen_test" | "ap_analysis",
           "reasoning": "..." }
Guard:   review_schema (G_val) — validates JSON structure
```

The `backtrack_target` field is the heuristic signal. When `ap_diff_review` says "backtrack to ap_gen_test", it means the reviewer concluded that the test itself is testing the wrong thing — no patch can satisfy it correctly. When it says "backtrack to ap_analysis", the root cause hypothesis is wrong.

This separates *generation* from *critique*. LLMs are empirically better at reviewing code than generating it. The review step catches logical errors that structural guards (git apply, syntax parse) cannot detect, and semantic guards (test_green) may miss if the test itself is flawed.

**Guard for ap_diff_review**: A schema validation guard (G_val) that checks the review output is valid JSON with the required fields. The review itself is not executed — it is an LLM judgement, not a test execution. The *actionability* of the review comes from the backtracking orchestrator consuming the `backtrack_target` field.

#### Arm 17: S1 TDD + Diff Review (Forward Only)

**Workflow**: `ap_analysis` -> `ap_gen_test` -> `ap_gen_patch` -> `ap_diff_review`
**rmax**: 6
**Thesis**: Adding a review step after patch generation catches logical errors that structural and execution guards miss. Forward-only — if the review rejects, feedback loops back to `ap_gen_patch` for retry. The `backtrack_target` field is recorded but not acted on. This is the control arm for Arm 18.

```
+--------------+     +--------------+     +--------------+     +----------------+
| ap_analysis  |---->| ap_gen_test  |---->| ap_gen_patch |---->| ap_diff_review |
|              |     |              |     |              |     |                |
| G: analysis  |     | G: composite |     | G: composite |     | G: review_     |
|              |     |  (syntax+red)|     |  (patch+     |     |    schema      |
|              |     |              |     |   green+eval)|     |                |
+--------------+     +--------------+     +--------------+     +----------------+
```

**What it measures**: Does LLM-as-reviewer improve resolve rate over Arm 05 (which has no review step)? The review guard is cheap (one LLM call, no Docker), so the overhead is small. Even without backtracking, the review's rejection feedback ("your patch handles the error case but breaks the happy path because...") is more informative than test output alone.

#### Arm 18: S1 TDD + Diff Review + Selective Backtracking

**Workflow**: Same structure as Arm 17, but with a `BacktrackOrchestrator` wrapping the workflow.
**rmax**: 6 per step, backtrack_budget: 2 per step
**Thesis**: When `ap_diff_review` identifies the failure origin (via `backtrack_target`), the orchestrator backtracks to the indicated step with the review's reasoning as amended context. This is *informed* search — the backtrack depth is guided by the review heuristic rather than being fixed.

```
+--------------+     +--------------+     +--------------+     +----------------+
| ap_analysis  |<-+  | ap_gen_test  |<-+  | ap_gen_patch |---->| ap_diff_review |
|              |  |  |              |  |  |              |     |                |
| G: analysis  |  |  | G: composite |  |  | G: composite |     | G: review_     |
|              |  |  |  (syntax+red)|  |  |  (patch+     |     |    schema      |
|              |  |  |              |  |  |   green+eval)|     |                |
+--------------+  |  +--------------+  |  +--------------+     +----------------+
       ^          |         ^          |                              |
       |          |         |          |                              |
       +----------+---------+----------+--------- backtrack_target --+
```

**Backtrack semantics** (from `plan_search_feedback_loop.md` Section 4):

| `backtrack_target` | Action | Context amendment |
|--------------------|--------|-------------------|
| `null` (approve) | Accept patch, workflow succeeds | — |
| `"ap_gen_patch"` | Retry patch with review feedback | Review issues appended to retry context |
| `"ap_gen_test"` | Regenerate test, then re-attempt patch | "Previous test led to a patch that [review reasoning]. Generate a test that better captures the actual bug." |
| `"ap_analysis"` | Regenerate analysis, then cascade forward | "Previous analysis hypothesised [X] but review found [Y]. Revise the root cause." |

**Budget model** (adapted from `plan_search_feedback_loop.md` Section 5):

```
ap_analysis:    rmax=3, backtrack_budget=1  -> max 6 analysis calls
ap_gen_test:    rmax=3, backtrack_budget=2  -> max 9 test calls
ap_gen_patch:   rmax=6, backtrack_budget=0  -> max 6 patch calls per test
ap_diff_review: rmax=1, backtrack_budget=0  -> 1 review per patch attempt
```

Worst case: ~30 LLM calls + Docker executions. Common case: analysis passes, test passes red, patch passes green, review approves = 4 LLM calls + 2 Docker execs.

**What it measures**: Does informed backtracking (guided by diff review) improve over:
- Arm 05 (no backtracking, no review)?
- Arm 17 (review but no backtracking)?
- Arm 13 (fixed backtracking without review heuristic)?

The comparison between Arms 13 and 18 is particularly important: Arm 13 backtracks mechanically when `test_green` fails (always goes to test refinement). Arm 18 backtracks *selectively* based on the reviewer's diagnosis — sometimes the test is wrong, sometimes the analysis is wrong, and the reviewer's `backtrack_target` distinguishes these cases.

**Why ap_diff_review is the right heuristic source**: The plan search design doc (`plan_search_feedback_loop.md` Section 4.2) defines a heuristic function `h: (step_id, GuardResult, retry_count, history) -> backtrack_depth`. The original design uses rule-based pattern matching on guard feedback strings ("not parseable" -> depth 0, "not satisfiable" -> depth 1). This is fragile — guard feedback in code repair is diverse and unstructured. `ap_diff_review` replaces the rule-based heuristic with an LLM-based one: the reviewer reads the full context (analysis + test + patch + guard feedback) and produces a *reasoned* backtrack recommendation. The cost is one additional LLM call per failed attempt, but the backtrack decision is far more accurate than string matching.

#### Arm 19: S1 TDD + Backtracking (No Review, Rule-Based Heuristic)

**Workflow**: Same as Arm 05 but with rule-based backtracking.
**rmax**: 6 per step, backtrack_budget: 2
**Thesis**: Control arm for Arm 18. Uses the rule-based heuristic from `plan_search_feedback_loop.md` Section 4.1 instead of `ap_diff_review`. Backtrack depth is determined by pattern-matching guard feedback strings:

```python
def backtrack_heuristic(step_id, guard_result, retry_count, history):
    feedback = guard_result.feedback.lower()

    # Structural errors -> retry same step
    if "syntax error" in feedback or "not parseable" in feedback:
        return 0

    # Repeated identical failures -> escalate
    if same_feedback_repeated(history, guard_result, threshold=2):
        return min(retry_count, 2)

    # Import/module errors after multiple retries -> analysis wrong
    if retry_count >= 2 and ("importerror" in feedback or "modulenotfounderror" in feedback):
        return 2  # backtrack to analysis

    # Test discrimination failure -> test is wrong
    if "passed on buggy code" in feedback:
        return 0  # retry test (already at test step)

    # Patch doesn't fix the bug -> maybe test is wrong
    if "still fails after patch" in feedback and retry_count >= 3:
        return 1  # backtrack from patch to test

    # Default: retry at same level
    return 0
```

**What it measures**: Is rule-based backtracking sufficient, or does the LLM-based review (Arm 18) provide meaningfully better heuristics? If Arm 19 matches Arm 18, the review step is overhead. If Arm 18 >> Arm 19, the LLM's ability to reason about *why* the patch is wrong justifies the extra call.

### 5.7 Generated Workflow Arms (Meta-Level)

Arms 01-19 use static workflow definitions: every instance runs the same pipeline regardless of problem characteristics. But SWE-bench instances vary enormously — a one-line typo fix and a multi-file API refactor have different optimal strategies. The framework's Extension 06 (Generated Workflows) formalises this: workflow definitions are themselves artifacts that can be generated, validated, and executed.

This introduces a **two-level hierarchy**:
- **Meta-level**: A planning agent classifies the problem and generates a workflow specification
- **Object-level**: The generated workflow executes against the instance

The meta-level is itself an action pair: `ap_generate_workflow` with a guard that validates the generated workflow is structurally sound (all components resolvable, dependencies well-formed, budget within limits). This is the DS-PDDL schema from Extension 06 Definition 25.

#### New Action Pair: `ap_classify_problem`

Classifies the problem instance into a category that determines which workflow template to use.

```
Input:   problem_statement + repository file listing
Output:  { "category": "trivial_fix" | "single_file_bug" | "multi_file_bug" | "api_change" | "refactor",
           "estimated_complexity": 1-5,
           "reasoning": "..." }
Guard:   classification_schema (G_val) — validates JSON + category enum
```

#### New Action Pair: `ap_generate_workflow`

Generates a workflow specification (JSON) based on the problem classification. The workflow specification uses the same format as the static workflow configs (action_pairs, requires, guard, guard_config) but is generated per-instance.

```
Input:   problem classification + component registry (available generators, guards)
Output:  workflow JSON (same format as 01_baseline.json etc.)
Guard:   workflow_schema (G_val) — all action pairs reference registered components,
         dependencies are acyclic, total budget within B_max
```

The component registry (Definition 26) is the list of available generators and guards — the meta-level agent can only compose from existing primitives, not invent new ones. This constrains the generated workflow to be executable.

#### Arm 20: Adaptive Workflow (Classify + Generate + Execute)

**Workflow** (meta-level): `ap_classify_problem` -> `ap_generate_workflow` -> execute(generated_workflow)
**rmax**: 2 (meta-level), generated workflow's own rmax (object-level)
**Thesis**: Problem-adaptive workflow selection outperforms a fixed pipeline. Simple instances get lightweight workflows (singleshot or analysis -> patch). Complex instances get full TDD with execution verification. The meta-level overhead (2 LLM calls for classification + generation) is amortised by avoiding expensive pipelines on easy instances and providing richer pipelines for hard ones.

```
+--------------------+     +---------------------+     +----------------------+
| ap_classify_problem|---->| ap_generate_workflow|---->| execute(workflow)    |
|                    |     |                     |     |                      |
| G: classification_ |     | G: workflow_schema  |     | (object-level        |
|    schema          |     |  (components valid, |     |  execution with      |
|                    |     |   deps acyclic,     |     |  its own guards)     |
|                    |     |   budget in range)  |     |                      |
+--------------------+     +---------------------+     +----------------------+
```

**Example generated workflows by category**:

| Problem category | Expected generated workflow | Why |
|-----------------|---------------------------|-----|
| `trivial_fix` | `ap_singleshot` (rmax=2) | One-line fix, decomposition is overhead |
| `single_file_bug` | `ap_analysis -> ap_gen_patch` (rmax=3) | Comprehension helps but TDD overkill |
| `multi_file_bug` | `ap_analysis -> ap_gen_test -> ap_gen_patch` (rmax=6, CompositeGuard) | Full TDD with execution verification |
| `api_change` | `ap_localize -> ap_analysis -> ap_gen_test -> ap_gen_patch` (rmax=6) | Localization critical for multi-file |
| `refactor` | `ap_analysis -> ap_regression_spec -> ap_gen_patch` (rmax=4) | Regression focus, no new behaviour to test |

**What it measures**: Does per-instance workflow selection improve over best-fixed-pipeline (Arm 05)? The resolve rate comparison tells us whether adaptive selection helps. The token efficiency comparison tells us whether it saves budget on easy instances. The per-category breakdown reveals which problem types benefit most from adaptation.

**Connection to backtracking**: The generated workflow *can include backtracking configuration* if the meta-level agent includes it in the spec. An advanced variant would generate workflows with `backtrack_budget` fields, effectively letting the meta-level agent decide not just *which steps* to run but also *how much search* to allocate. This connects Arm 20 to the backtracking arms (17-19): the meta-level agent could choose to include `ap_diff_review` + backtracking for complex instances but skip it for simple ones.

#### Arm 21: Adaptive Workflow + Backtracking (Full Meta-Level)

**Workflow** (meta-level): `ap_classify_problem` -> `ap_generate_workflow` -> execute(generated_workflow_with_backtracking)
**Thesis**: The meta-level agent generates workflows that include backtracking configuration. For complex instances, the generated workflow includes `ap_diff_review` and `backtrack_budget` per step. For simple instances, no backtracking overhead.

This is the most ambitious arm — it combines:
- **Adaptive workflow selection** (Arm 20): right pipeline for the problem
- **Informed backtracking** (Arm 18): `ap_diff_review` as heuristic when included
- **Budget-aware search** (`plan_search_feedback_loop.md` Section 5): per-step backtrack budgets tuned by the meta-level agent

```
Meta-level:
  ap_classify_problem -> ap_generate_workflow(+backtrack_config)

Object-level (generated, example for multi_file_bug):
  ap_analysis -> ap_gen_test -> ap_gen_patch -> ap_diff_review
  with backtrack_budget: {ap_analysis: 1, ap_gen_test: 2, ap_gen_patch: 0}
```

**What it measures**: The ceiling question — how much of the gap between singleshot (Arm 02) and perfect is recoverable through adaptive orchestration? If Arm 21 significantly outperforms Arm 05 (best fixed pipeline) and Arm 18 (fixed pipeline + backtracking), the meta-level adds genuine value beyond just choosing the right fixed pipeline. If it doesn't, fixed pipelines with backtracking are sufficient and the meta-level is unnecessary complexity.

**Learning signal**: Every instance x workflow execution produces a trace: which workflow was generated, which steps succeeded/failed, which backtracks occurred, final resolve outcome. This trace is a training signal for improving the meta-level agent. Over many instances, the `ap_classify_problem` and `ap_generate_workflow` prompts can be refined based on which generated workflows actually led to resolves. This connects to Extension 04 (Learning Loop, Definitions 21-24): the refinement predicate selects traces where the generated workflow succeeded, and the training loss conditions on both the problem statement and the classification to improve future workflow generation.

### 5.8 Additional Action Pairs (Composable Across Arms)

The following action pairs are not tied to specific arms — they can be inserted into any pipeline to test their individual contribution.

#### `ap_context_gather` — Targeted File Reading

```
Input:   localization result (file list from ap_localize)
Output:  { "files": [{"path": "...", "content": "...", "relevant_lines": [...]}] }
Guard:   context_schema (G_val) — at least one file read, total content within token budget
Requires: ap_localize
```

Between localization and analysis, reads the actual file contents of localized files and injects them into context. Currently models get a file *listing* but never see contents — the most frequently observed failure mode. Unlike just adding localization (Arm 07), this explicitly extracts and bounds the content to fit within the context window.

#### `ap_regression_spec` — Regression Test Extraction

```
Input:   analysis artifact + repository file listing
Output:  { "test_framework": "...", "test_patterns": [...], "fixture_conventions": [...],
           "example_tests": [...] }
Guard:   spec_schema (G_val) — references actual test files from the repository
Requires: ap_analysis
```

Before writing the discriminating test, extracts existing test patterns from the repo: framework, assertion style, fixture conventions, naming patterns. Grounds the test generator in the project's idioms rather than generic pytest patterns. Particularly valuable for non-Python projects where test conventions vary widely (Go table-driven tests, Jest describe/it nesting, etc.).

#### `ap_root_cause_verify` — Analysis Verification via Counterfactual

```
Input:   analysis artifact
Output:  { "verification_test": "...", "prediction": "fail" | "pass",
           "actual_result": "fail" | "pass", "verified": true | false }
Guard:   composite(test_syntax, counterfactual_exec) — test parses AND execution
         matches prediction
Requires: ap_analysis
```

After analysis, generates a counterfactual question: "If the root cause hypothesis is correct, what specific test would demonstrate it?" Runs that test against the buggy code. If the test doesn't fail as predicted, the analysis is wrong — backtrack. This is a *verification guard on the analysis itself*, not just schema validation. The existing `AnalysisGuard` only checks JSON structure; this checks whether the hypothesis is empirically consistent.

### 5.9 Future Arm Summary Table

| Arm | Action Pairs | New Element | Hypothesis |
|-----|-------------|-------------|------------|
| 07 | localize -> analysis -> patch | Localization before comprehension | Grounding prevents hallucination |
| 08 | localize -> analysis -> test -> patch | Localization + Guard-Driven Synthesis | Best of both worlds |
| 09 | analysis(oracle) -> test -> patch | Perfect localization | Ceiling measurement |
| 10 | analysis -> test(+spec) -> patch | Requirements injection | Richer spec -> better tests |
| 11 | analysis -> spec_extract -> test -> patch | Explicit specification step | Structured spec intermediary |
| 12 | analysis -> test -> patch -> refine_patch | Cross-pair iteration (forward) | Regression repair |
| 13 | analysis -> test -> patch -> refine_test -> patch | Cross-pair backtracking (fixed) | Controlled re-specification |
| 14 | analysis xN -> test -> patch | SC-Planning on analysis | Hypothesis selection |
| 15 | analysis -> test xN -> patch | Multi-test generation | Test diversity |
| 16 | (05 with adaptive guards) | Language-specific guards | Language-aware validation |
| 17 | analysis -> test -> patch -> diff_review | LLM-as-reviewer, forward only | Critique catches logical errors |
| 18 | analysis -> test -> patch -> diff_review + backtrack | Review-guided selective backtracking | Informed search via LLM heuristic |
| 19 | (05 with rule-based backtracking) | Rule-based backtracking, no review | Baseline for backtracking value |
| 20 | classify -> generate_workflow -> execute | Per-instance adaptive workflow | Right pipeline for the problem |
| 21 | classify -> generate_workflow(+backtrack) -> execute | Adaptive workflow + backtracking | Full meta-level orchestration |

---

## 6. Evaluation Protocol

### 6.1 Primary Metric

**Resolve rate**: Percentage of instances where the generated patch passes both `fail_to_pass` and `pass_to_pass` test suites in the official evaluation harness.

### 6.2 Secondary Metrics

| Metric | What it measures | Capture method |
|--------|-----------------|----------------|
| **Guard pass rate by phase** | epsilon per action pair | Count PASS/FAIL per guard across instances |
| **Retry utilisation** | How many retries are actually used | Distribution of attempts before acceptance |
| **Guard feedback quality** | Does retry feedback improve outcomes | Compare attempt 1 vs attempt N success rates |
| **Decomposition overhead** | Token cost of multi-step vs single-shot | Sum tokens across action pairs per instance |
| **Execution time** | Wall clock per instance per arm | End-to-end timing |
| **Discriminator rate** | % of generated tests that fail on buggy code | `test_red` pass rate (Arms 05/06 only) |
| **Constraint collapse ratio** | Solve rate improvement from test->fix vs direct | Compare Arms 04/05 vs 03 per instance |

### 6.3 Statistical Design

- **Paired comparison**: Each instance evaluated by all arms — enables paired statistical tests
- **Stratification**: Report by language (Python/Go/JS/TS), repo, patch complexity (lines, files)
- **Confidence intervals**: Bootstrap 95% CI on resolve rates (following SWE-bench Pro methodology)
- **Effect size**: McNemar's test for paired binary outcomes between arms

### 6.4 Ablation Structure

The arms form a nested ablation:

```
02 (singleshot)
 +- + comprehension = 03 (s1_direct)
     +- + test synthesis = 04 (s1_tdd, static guards)
         +- + execution verification = 05 (s1_tdd_verified)
             +- + behaviour prompts = 06 (s1_tdd_behavior)
             +- + diff review (forward) = 17 (s1_tdd_review)
             |   +- + selective backtracking = 18 (review-guided backtrack)
             +- + rule-based backtracking = 19 (baseline backtrack)
```

Each transition isolates one variable. Arm 01 (baseline) provides an orthogonal comparison — decomposition via localization instead of comprehension. Arms 17-19 form a second ablation branch from Arm 05, isolating review value (17 vs 05), backtracking value (18 vs 17), and heuristic quality (18 vs 19). Arms 20-21 are orthogonal — they test meta-level orchestration and can incorporate any of the above as object-level components.

---

## 7. Artifact DAG Structure

Each instance x arm produces a content-addressed artifact DAG:

```
output/swe_bench_pro/
+-- results.jsonl                           # One JSON line per instance x arm
+-- artifact_dags/{instance_id}/{arm}/
|   +-- index.json                          # action_pairs: artifact IDs per action
|   +-- objects/{hash_prefix}/{artifact_id}.json
+-- predictions/
|   +-- 02_singleshot.json                  # Prediction file per arm
|   +-- 03_s1_direct.json
|   +-- ...
+-- eval_output/
    +-- eval_results.json                   # Official harness output
```

**index.json** maps action names to artifact ID lists. Rejected artifacts remain in the DAG with guard feedback — the full retry history is preserved. This enables post-hoc analysis of failure modes and retry effectiveness.

**Artifact content** includes: generator prompt, raw LLM response, extracted artifact, guard verdict, guard feedback message, timestamps, token counts.

---

## 8. Connection to AtomicGuard Theory

### 8.1 Article Section Mapping

| Experimental Element | Article Section | Concept |
|---------------------|-----------------|---------|
| Arms 02->05 progression | S3 Atomic Action Pair | Progressive guard complexity on (rho, a_gen, G) |
| CompositeGuard chains | S3 Two-phase validation | G_val -> G_ver pipeline |
| Retry with feedback | S4 Contingency / Red Loop | Context refinement on guard failure |
| Arms 04 vs 05 | S5.2 Guard-Driven Synthesis | Static vs execution-verified constraint collapse |
| Guard pass rates (epsilon) | S4 Promise fulfillment | Empirical epsilon estimation per action pair |
| SWE-bench Pro vs Verified | S5.1 Unknown Specifications | Specification richness vs bootstrap difficulty |
| Future Arms 14-15 | S7.3 SC-Planning | Self-Consistency for hypothesis/test selection |
| Arm 05 guard chain cost | S6 Complexity Cliff | G_val = O(N), G_ver = O(exec) |
| Arms 17-19 (backtracking) | S4 Contingency + plan_search_feedback_loop.md | Guards as heuristic functions for cross-pair search |
| ap_diff_review (Arm 18) | S4 Red Loop + plan_search_feedback_loop.md S4.2 | LLM-based heuristic replacing rule-based backtrack depth |
| Arms 20-21 (generated workflows) | Ext 06 Generated Workflows (Defs 25-26) | Workflow artifacts, component registry, two-level hierarchy |
| Arm 21 learning signal | Ext 04 Learning Loop (Defs 21-24) | Refinement predicate on workflow execution traces |

### 8.2 Predictions from Theory

1. **Arms 02 < 03 < 04 < 05**: Progressive decomposition + guard escalation should monotonically improve resolve rates
2. **04 ~ 05 on easy instances**: Static guards sufficient when tasks are well-specified (complexity below cliff)
3. **05 >> 04 on hard instances**: Execution verification provides decisive signal on complex multi-file patches
4. **epsilon(ap_gen_fix) < epsilon(ap_analysis)**: Coding tasks have lowest promise fulfillment rate (~0.35), confirming retry necessity
5. **Pro delta > Verified delta**: Guard-Driven Synthesis shows larger relative improvement on Pro (richer spec material)
6. **06 >= 05**: Behavior-focused prompts should produce better discriminating tests (higher test_red pass rate)

### 8.3 Predictions for Backtracking and Meta-Level Arms

7. **18 > 17 > 05**: Adding review improves over no review; adding backtracking improves over forward-only review
8. **18 > 19**: LLM-based backtrack heuristic (ap_diff_review) outperforms rule-based heuristic on diverse feedback
9. **19 > 05 on hard instances**: Even rule-based backtracking should help on multi-file patches where first attempts frequently fail
10. **Backtrack depth distribution**: Most backtracks should be depth 0-1 (retry or regenerate test); depth 2 (regenerate analysis) should be rare (<10% of backtracks)
11. **20 > 05 on token efficiency**: Adaptive workflows save budget on easy instances (singleshot for trivial fixes) while matching or beating resolve rate
12. **21 >= 20**: Adding backtracking configuration to generated workflows should not hurt and may help on the hardest instances
13. **ap_diff_review backtrack_target accuracy**: When the reviewer recommends backtracking to a specific step, the subsequent regeneration should succeed at a higher rate than blind retry — this validates that the heuristic is informative

### 8.4 What Would Disprove the Framework

- **02 >= 05**: If singleshot matches or beats the full TDD pipeline, decomposition adds cost without value
- **04 ~ 05 everywhere**: If execution guards never catch what static guards miss, the CompositeGuard is overhead
- **Retry loops don't converge**: If attempt N+1 success rate equals attempt 1 success rate, feedback doesn't help
- **Verified delta > Pro delta**: If Guard-Driven Synthesis helps more on easy tasks, the theory about specification richness is wrong
- **19 ~ 05**: If rule-based backtracking doesn't improve over fixed retry, cross-pair feedback is not actionable
- **18 ~ 19**: If the LLM review heuristic doesn't outperform rules, the extra LLM call is pure overhead
- **20 ~ 05**: If adaptive workflow selection doesn't improve over best-fixed-pipeline, the classification step is wasted
- **backtrack_target is random**: If ap_diff_review's backtrack recommendations don't correlate with successful recovery, the LLM cannot reliably diagnose failure origins

All outcomes are publishable. Negative results constrain the framework's applicability claims.

---

## 9. Practical Notes

### 9.1 Known Issues

- **`require_git_apply` oversight**: Arms 02-04 workflow configs previously set `require_git_apply: false`. Fixed to `true` for all arms. Without it, the guard accepts patches that won't apply cleanly, pushing failure to evaluation time rather than catching it during retry.
- **80-file cap**: Large repos silently truncate the file listing. The LLM has no indication the listing is incomplete. Future: append "(showing 80 of N files)" note.
- **Docker image availability**: `test_red`/`test_green`/`full_eval` guards require per-instance Docker images. Images auto-pull but initial pull is slow.

### 9.2 Resource Estimates

| Arm | LLM calls/instance | Docker execs/instance | Approx time/instance |
|-----|--------------------|-----------------------|---------------------|
| 02 | 1-3 | 0 | 10-30s |
| 03 | 2-6 | 0 | 15-60s |
| 04 | 3-9 | 0 | 20-90s |
| 05 | 3-18 | 2-12 | 1-10 min |
| 06 | 3-18 | 2-12 | 1-10 min |
| 17 | 4-24 | 2-12 | 1-12 min |
| 18 | 4-30 | 2-18 | 2-15 min |
| 19 | 3-24 | 2-18 | 1-12 min |
| 20 | 3-22 | 0-12 | 1-12 min |
| 21 | 4-32 | 0-18 | 2-18 min |

Full 731-instance run across core 6 arms: ~50-200 compute-hours depending on model and parallelism. Backtracking arms (17-19) add ~50% overhead. Meta-level arms (20-21) vary widely by generated workflow complexity.

### 9.3 Recommended Sampling Strategy for ISMIS

If full runs are infeasible within the submission timeline:
- **Stratified sample**: 50 instances, balanced across 4 languages and difficulty tiers
- **Run all 6 arms** on the sample for the ablation analysis
- **Full-set results** for Arms 02, 05 as the key comparison (singleshot vs full TDD verified)
- Report confidence intervals acknowledging sample size

---

## 10. Changelog

| Date | Change |
|------|--------|
| 2026-02-06 | v0.1 — Initial design document. 6 arms defined. Future arm design space (Arms 07-16) outlined. |
| 2026-02-06 | v0.2 — Added feedback-classified backtracking arms (17-19) with ap_diff_review as LLM-based heuristic. Added generated workflow arms (20-21) for adaptive meta-level orchestration. Added composable action pairs (ap_context_gather, ap_regression_spec, ap_root_cause_verify). Extended theory mapping and predictions for new arms. |
