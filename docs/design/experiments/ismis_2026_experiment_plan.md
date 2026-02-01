# ISMIS 2026 Experiment Plan

**Paper:** The Chaos and the Scaffold: Contingent Planning for LLMs
**Dataset:** [SWE-PolyBench](https://huggingface.co/datasets/AmazonScience/SWE-PolyBench) (full, Python subset)
**Model:** Qwen2.5-Coder-32B via HuggingFace Inference API

---

## Section 7 Overview

| #   | Experiment                          | Status       | Type        |
| --- | ----------------------------------- | ------------ | ----------- |
| 7.1 | G_plan Defect Detection             | Done         | Validation  |
| 7.2 | Bug Fix Strategy Comparison         | **Planned**  | Statistical |
| 7.3 | Incremental Execution Savings       | Planned      | Demo        |
| 7.4 | SC-Planning                         | If time      | Statistical |

---

## 7.2 Bug Fix Strategy Comparison on SWE-PolyBench

### Motivation

Bug fixes dominate real-world repositories. In the SWE-PolyBench Python subset,
162 of 199 instances (81%) are bug fixes. A scaffold architecture must demonstrate
value primarily on this category.

The published paper (arXiv:2512.20660) established the Action Pair and retry loop.
This experiment tests the **new** contribution: does the decomposed pipeline with
strategy selection improve outcomes, and does applying Guard-Driven Synthesis
(Section 5.2) to bug fixing outperform direct fixing?

### Dataset

**Source:** `AmazonScience/SWE-PolyBench` full dataset, `language == "python"`

| task_category | Count | Role in experiment                    |
| ------------- | ----- | ------------------------------------- |
| bug           | 162   | Primary — all three arms              |
| feature       | 29    | Secondary — report, don't depend on   |
| refactoring   | 7     | Tertiary — report, note sample size   |

**Fields used per instance:**
- `problem_statement` — input to pipeline
- `patch` — gold solution (reference only)
- `test_patch` — test changes from the PR
- `fail_to_pass` — tests that must pass after fix (success criterion)
- `pass_to_pass` — tests that must continue passing (regression check)
- `base_commit` — repo state before fix
- `repo` — repository identifier
- `Dockerfile` — instance-level evaluation environment

### Arms

Three arms, applied to every bug fix instance:

| Arm               | Pipeline                                                         | Action Pairs |
| ----------------- | ---------------------------------------------------------------- | ------------ |
| **Single-shot**   | problem_statement → generate fix → verify                        | 1            |
| **S1 (Direct)**   | problem_statement → g_analysis → generate fix → verify           | 2            |
| **S1-TDD**        | problem_statement → g_analysis → write_failing_test → fix_to_green → verify | 3  |

#### Arm 1: Single-shot (baseline)

```
Input:  problem_statement
Step 1: a_gen(problem_statement) → patch
Guard:  Run FAIL_TO_PASS tests
```

No decomposition. Direct generation of a fix from the issue description.

#### Arm 2: S1 — Direct Fix (decomposed)

```
Input:  problem_statement
Step 1: g_analysis(problem_statement) → analysis artifact
  Guard: G_analysis (structural validation)
Step 2: a_gen(problem_statement + analysis) → patch
  Guard: Run FAIL_TO_PASS tests
```

The analysis step classifies the bug, identifies likely affected components,
and characterises the defect. This narrows the search space for the fix
generation step.

#### Arm 3: S1-TDD — Reproduce then Fix (Guard-Driven Synthesis for bugs)

```
Input:  problem_statement
Step 1: g_analysis(problem_statement) → analysis artifact
  Guard: G_analysis (structural validation)
Step 2: a_gen_test(problem_statement + analysis) → failing test
  Guard: G_test (syntax validation — test must parse)
Step 3: a_gen_fix(problem_statement + analysis + failing_test) → patch
  Guard: Run FAIL_TO_PASS tests
```

The key difference: Step 2 generates a failing test that **reproduces** the bug.
This test acts as an executable specification (the "Beacon" from Section 5.2),
constraining the fix generation in Step 3. The generated test is an intermediate
artifact — final validation is always against the ground-truth FAIL_TO_PASS suite.

### Strategy Mapping to Paper Sections

| Arm     | Paper section              | What it validates                              |
| ------- | -------------------------- | ---------------------------------------------- |
| Single  | Baseline                   | ε without scaffold                             |
| S1      | Section 4.1 (Decomposition)| Search space compression via analysis step     |
| S1-TDD  | Section 5.2 (Guard-Driven) | Constraint Collapse — test bootstraps the spec |

### Metrics

**Primary:** ε̂ (promise fulfillment rate) — proportion of instances where the
generated patch passes all FAIL_TO_PASS tests.

**Secondary:**
- Pass-to-pass preservation rate (regression safety)
- Token consumption per arm (requires capturing `response.usage` from HF API)
- Token Efficiency Ratio: ε̂ / mean tokens consumed
- Wilson confidence intervals on all ε̂ values
- Fisher's exact test for pairwise arm comparisons
- Cohen's h effect size

**Reported per arm:**

```
| Arm        | n   | ε̂     | 95% CI        | Tokens (mean) | Efficiency |
| ---------- | --- | ----- | ------------- | ------------- | ---------- |
| Single     | 162 |       | [   -   ]     |               |            |
| S1         | 162 |       | [   -   ]     |               |            |
| S1-TDD     | 162 |       | [   -   ]     |               |            |
```

**Pairwise comparisons:**

```
| Comparison        | Δε    | p-value | Cohen's h | Interpretation |
| ----------------- | ----- | ------- | --------- | -------------- |
| S1 vs Single      |       |         |           |                |
| S1-TDD vs Single  |       |         |           |                |
| S1-TDD vs S1      |       |         |           |                |
```

### Evaluation Protocol

Each instance requires end-to-end execution:

1. Load instance from SWE-PolyBench (problem_statement, repo, base_commit)
2. Set up evaluation environment (Docker, using instance Dockerfile)
3. Checkout repo at base_commit
4. Run pipeline arm (generate analysis → generate test/fix → collect artifacts)
5. Apply generated patch to repo
6. Run FAIL_TO_PASS test suite
7. Run PASS_TO_PASS test suite (regression check)
8. Record: pass/fail, tokens consumed, artifacts generated

This follows SWE-PolyBench's own evaluation protocol via their
[evaluation harness](https://github.com/scaleapi/SWE-bench_Pro-os).

### Retry Policy

For the primary result, use **rmax=0** (single attempt per step) to isolate the
strategy effect from the retry effect. The retry mechanism is already validated
in the published paper.

Optional secondary run with rmax=2 to measure whether decomposition + retry
compounds, but this is lower priority.

---

## 7.1 G_plan Defect Detection (Done)

**Result:**

| Metric         | Minimal | Medium | Expansive |
| -------------- | ------- | ------ | --------- |
| Detection rate | 38%     | 100%   | 100%      |
| Complexity     | O(V+E)  | O(V×L) | O(R^K)    |

Medium rigor is necessary and sufficient for production use.

---

## 7.3 Incremental Execution Savings (Demo)

**Goal:** Demonstrate repository acts as dependency-aware cache.

**Protocol:** Run SDLC v2 workflow full, modify one requirement, run incremental,
compare steps executed and wall time. Uses local Ollama, no HuggingFace needed.

**Requires:** PR 4 (Incremental Execution support) and PR 9 (SDLC v2 workflow).

---

## 7.4 SC-Planning (If Time)

**Goal:** Test if Self-Consistency improves plan selection among G_plan-valid plans.

**Requires:** Plan clustering infrastructure not currently built.

---

## Infrastructure Requirements

### What exists on the current branch

- [x] HuggingFace Generator (`src/atomicguard/infrastructure/llm/huggingface.py`)
- [x] Workflow orchestration (`src/atomicguard/application/workflow.py`)
- [x] DualStateAgent with retry loop (`src/atomicguard/application/agent.py`)
- [x] SyntaxGuard, ImportGuard (`src/atomicguard/guards/static/`)
- [x] DynamicTestGuard (`src/atomicguard/guards/dynamic/test_runner.py`)
- [x] CompositeGuard (`src/atomicguard/guards/composite/base.py`)
- [x] Statistical analysis: Wilson CI, Fisher's exact, Cohen's h (`benchmarks/simulation.py`)
- [x] Result persistence: CSV/SQLite writer (`benchmarks/simulation.py`)
- [x] Visualization: matplotlib charts (`benchmarks/simulation.py`)

### Context Enrichment Gap (critical for decomposed pipelines)

The current Workflow passes dependency artifacts between steps, but **only the
artifact IDs reach the generator** — the actual content does not flow into the
prompt. Here is the full trace:

1. `Workflow.execute()` collects full `Artifact` objects as dependencies
2. `DualStateAgent._compose_context()` reduces them to ID pairs:
   `dependency_artifacts=tuple((k, v.artifact_id) for k, v in dependencies.items())`
3. `ActionPair.execute()` splits the flow:
   - **Guard** receives full `Artifact` objects via `guard.validate(artifact, **dependencies)` ✓
   - **Generator** receives `Context` which has only IDs, not content ✗
4. Neither `PromptTemplate.render()` nor `HuggingFaceGenerator._build_basic_prompt()`
   resolves dependency IDs into artifact content

**Impact on experiment:** In the S1 and S1-TDD arms, the analysis artifact from
step 1 must be visible to the generator in step 2 (and step 3). Without content
enrichment, the decomposed pipeline steps cannot see each other's outputs.

**Resolution options:**

| Option | Description | Effort |
| ------ | ----------- | ------ |
| A. PromptTemplate resolution | Add dependency content lookup to `PromptTemplate.render()` using `context.ambient.repository.get_artifact()` | Small (~5 LOC) |
| B. Context stores content | Change `Context.dependency_artifacts` to store `(key, artifact_id, content)` tuples | Small (model change + agent change) |
| C. Experiment-level workaround | Experiment runner manually composes enriched specifications, bypassing Workflow | None (but doesn't test the framework) |

**Option A is preferred.** It requires no model changes, the `ArtifactDAGInterface`
is already available on `context.ambient.repository`, and it makes dependency
content available to all generators via the standard prompt template path.

Option C is the fallback if we want to run the experiment before fixing the
framework, but it means we're testing prompt composition, not the Workflow
infrastructure itself.

### What needs to be built

| Component                          | Description                                             | Effort   | Blocking? |
| ---------------------------------- | ------------------------------------------------------- | -------- | --------- |
| **Context enrichment in PromptTemplate** | Resolve dependency artifact IDs into content for generator prompt | **Small** | **Yes** |
| SWE-PolyBench data loader          | Load dataset, filter Python, parse fields               | Small    | Yes       |
| SWE-PolyBench evaluation harness   | Docker setup, repo checkout, test execution per instance| Large    | Yes       |
| G_analysis guard                   | Validates analysis artifact structure (JSON schema)     | Small    | Yes       |
| G_test guard (syntax)              | Validates generated test parses (ast.parse)             | Exists   | No        |
| Pipeline prompt templates          | g_analysis, g_strategy, a_gen_test, a_gen_fix prompts   | Medium   | Yes       |
| Token tracking in HuggingFaceGenerator | Capture response.usage from HF chat_completion      | Small    | No        |
| Experiment runner                  | Orchestrates three arms across all instances            | Medium   | Yes       |
| Result analysis                    | Adapt simulation.py stats for this experiment format    | Small    | No        |

### The evaluation harness is the bottleneck

SWE-PolyBench provides per-instance Dockerfiles and test commands. The evaluation
loop (checkout repo → apply patch → run tests) is the most complex piece. Options:

1. **Use SWE-PolyBench's own evaluation code** from
   [amazon-science/SWE-PolyBench](https://github.com/amazon-science/SWE-PolyBench)
   — adapting their harness to run our pipeline instead of an agent.

2. **Use SWE-bench's evaluation harness** — SWE-PolyBench follows the SWE-bench
   evaluation protocol. The existing tooling at
   [swebench](https://github.com/princeton-nlp/SWE-bench) can be reused.

3. **Build minimal harness** — for each instance: docker build, apply patch,
   run test_command, parse results.

Option 1 or 2 is strongly preferred over building from scratch.

---

## PR Requirements

### For Experiment 7.2 (Bug Fix Strategy Comparison)

The experiment **requires context enrichment** (see gap analysis above). The
current Workflow passes dependency artifacts between steps, but only IDs reach
the generator — not content. This must be fixed for the S1 and S1-TDD arms.

**Minimum viable change:** Add dependency content resolution to
`PromptTemplate.render()` (~5 LOC). This is Option A from the gap analysis
and can be done on the current branch without any PRs.

**PR dependencies:**

| PR        | Required? | Content                                    | Value for 7.2                              |
| --------- | --------- | ------------------------------------------ | ------------------------------------------ |
| None      | **Yes**   | Context enrichment fix (Option A)          | Generators must see prior step outputs     |
| PR 2+3+4  | No        | Domain extensions, checkpoint, incremental | Extended Workflow capabilities, richer models for artifact tracking |
| PR 5      | No        | GuardResult stored in Artifact             | Cleaner data collection — each artifact records its guard outcome |
| PR 10     | No        | G_plan benchmark core                      | Pipeline prompt templates, G_plan validation logic, benchmark runner |
| PR 11     | No        | Pipeline decomposition + evaluation harness| Dataset adapters, evaluation harness, pipeline guards |

**Minimum viable set:** Current branch + context enrichment fix + new experiment code.

**Recommended set:** Merge grouped PR 2+3+4 → PR 5 → PR 10. This provides
the pipeline infrastructure and G_plan validation that the experiment builds on,
avoiding reimplementation. The context enrichment fix should be included in
PR 2+3+4 since it relates to how the Workflow passes context between steps.

### For Experiment 7.3 (Incremental Execution)

| PR   | Required | Reason                                 |
| ---- | -------- | -------------------------------------- |
| PR 4 | Yes      | Incremental execution support          |
| PR 9 | Yes      | SDLC v2 workflow (the demo scenario)   |

PR 4 is in the grouped PR 2+3+4. PR 9 depends on PR 6 (Composite Guards extension).

**Full chain:** PR 2+3+4 → PR 5 → PR 6 → PR 9.

---

## Merge Order (revised)

```
Phase 1: PR 2+3+4 (grouped — domain extensions, checkpoint, incremental)
Phase 2: PR 5     (GuardResult in Artifact)
Phase 3: PR 10    (G_plan benchmark core)     ← enables 7.2
Phase 4: PR 6     (Composite Guards)
Phase 5: PR 9     (SDLC v2)                   ← enables 7.3
Phase 6: PR 11    (evaluation harness)         ← enriches 7.2
Phase 7: PR 7, 8  (examples)
Phase 8: PR 12    (docs)
```

---

## Timeline

| Day | Task                                                        |
| --- | ----------------------------------------------------------- |
| 1   | Merge grouped PR 2+3+4 + PR 5. Set up SWE-PolyBench eval.  |
| 2   | Build pipeline prompts + G_analysis guard. Run single-shot. |
| 3   | Run S1 and S1-TDD arms. Collect results.                   |
| 4   | Statistical analysis, tables, update paper Section 7.       |

---

## Feature / Refactoring Instances (secondary)

The 29 feature and 7 refactoring Python instances run through the same pipeline
but with category-appropriate strategies:

| task_category | Strategy                                           | n   |
| ------------- | -------------------------------------------------- | --- |
| feature       | S2 — Write test → implement → verify               | 29  |
| refactoring   | S3 — Characterise behaviour → transform → verify   | 7   |

Results are reported alongside the bug fix results but with explicit caveats
about sample size. The natural category distribution (81% bug fix) is itself
a data point: the scaffold must work on the dominant category of real work.

---

## References

- SWE-PolyBench dataset: https://huggingface.co/datasets/AmazonScience/SWE-PolyBench
- SWE-PolyBench paper: https://arxiv.org/abs/2504.08703
- SWE-PolyBench evaluation: https://github.com/amazon-science/SWE-PolyBench
- Published paper: https://arxiv.org/abs/2512.20660
- Paper outline: docs/design/paper_guards/ (The Chaos and the Scaffold)
