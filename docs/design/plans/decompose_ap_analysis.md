# Plan: Decompose `ap_analysis` into Granular Action Pairs

**Status**: DRAFT
**Branch**: `claude/guard-driven-synthesis-design-Wksjt`
**Depends on**: Architecture cleanup Issues 0-6 (DONE), `escalation → escalate_feedback_to` rename (DONE)

---

## 1. Problem

`ap_analysis` is a monolithic action pair that asks a single LLM call to perform four distinct cognitive tasks:

1. **Localize** — identify files and functions to modify
2. **Classify** — determine bug type (off-by-one, missing null check, API misuse, etc.)
3. **Root-cause** — form a testable hypothesis about why the bug exists
4. **Plan fix** — describe a concrete fix approach + extract expected/actual behavior for TDD

When `ap_analysis` fails its guard, the feedback is coarse: "analysis rejected". The agent retries the entire analysis from scratch, even when only one sub-task was wrong (e.g., localization was correct but root-cause hypothesis was off). Worse, when downstream steps (`ap_gen_test`, `ap_gen_patch`) escalate via `escalate_feedback_to`, they can only point at the monolithic `ap_analysis` — there is no way to say "the localization was fine, re-do the root cause".

This wastes tokens, obscures failure attribution, and limits the effectiveness of informed backtracking (Extension 09).

---

## 2. Approach: Granular Action Pairs with Targeted Escalation

Decompose `ap_analysis` into two individually-guarded action pairs:

| Step | ID | Generator | Guard | What it produces |
|------|----|-----------|-------|------------------|
| 1 | `ap_localize` | `LocalizationGenerator` | `localization` | Files and functions to modify |
| 2 | `ap_characterize` | `CharacterizationGenerator` | `characterization` | Bug classification, root-cause hypothesis, fix approach, expected/actual behavior |

**Why two steps, not three or four?**

- Localization vs. characterization is the natural cognitive boundary. Localization is *where*; characterization is *what* and *why*.
- Localization has a cheap, checkable guard (files exist in repo, right count, right extensions). This is already implemented in workflow `01_baseline.json`.
- Characterization requires the localization as input — it reasons *about* the localized code, not the whole repo.
- Further splitting characterization (classify vs. root-cause vs. fix-plan) would create action pairs that share too much context and whose guards can't meaningfully validate independently. The root-cause hypothesis only makes sense in the context of the classification, and the fix approach only makes sense given the root cause.

### How this improves escalation

**Before** (monolithic `ap_analysis`):

```
ap_analysis ──→ ap_gen_test ──→ ap_gen_patch
                     │                │
            escalate_feedback_to:   escalate_feedback_to:
              [ap_analysis]           [ap_analysis, ap_gen_test]
```

When `ap_gen_test` fails with stagnation, it escalates to `ap_analysis`. The entire analysis is thrown away and regenerated — including localization that may have been correct.

**After** (granular):

```
ap_localize ──→ ap_characterize ──→ ap_gen_test ──→ ap_gen_patch
                      │                   │                │
             escalate_feedback_to:  escalate_feedback_to:  escalate_feedback_to:
               [ap_localize]        [ap_characterize]       [ap_characterize, ap_gen_test]
                                                            escalate_feedback_by_guard:
                                                              PatchGuard: [ap_characterize]
                                                              TestGreenGuard: [ap_characterize, ap_gen_test]
                                                              FullEvalGuard: [ap_localize, ap_characterize]
```

Now escalation is **targeted**:
- If `ap_characterize` stagnates → re-localize (maybe we targeted the wrong files)
- If `ap_gen_test` stagnates → re-characterize (hypothesis was wrong, but files may be fine)
- If `ap_gen_patch` stagnates on `TestGreenGuard` → re-characterize + re-test (fix approach was wrong)
- If `ap_gen_patch` stagnates on `FullEvalGuard` → re-localize + re-characterize (fundamental miss)

Each escalation path preserves correct upstream work and only regenerates what failed.

---

## 3. Expected Pipeline Progression

The existing workflow files form a progression from simple to sophisticated. The decomposition adds a new granularity axis orthogonal to the TDD/review axis:

```
                           Granularity axis →
                    Monolithic              Decomposed
                    ──────────              ──────────
    Single-shot │  02_singleshot           (N/A — nothing to decompose)
                │   └─ ap_singleshot
                │
Complexity  S1  │  03_s1_direct            07_s1_direct_granular
axis        │   │   └─ ap_analysis           └─ ap_localize
            │   │   └─ ap_patch              └─ ap_characterize
            │   │                            └─ ap_patch
            │   │
            ↓   │  04_s1_tdd               08_s1_tdd_granular
         S1-TDD │   └─ ap_analysis           └─ ap_localize
                │   └─ ap_gen_test           └─ ap_characterize
                │   └─ ap_gen_fix            └─ ap_gen_test
                │                            └─ ap_gen_fix
                │
   S1-TDD +     │  05_s1_tdd_verified      09_s1_tdd_verified_granular
   backtrack    │   └─ ap_analysis           └─ ap_localize
                │   └─ ap_gen_test           └─ ap_characterize
                │   └─ ap_gen_patch          └─ ap_gen_test
                │                            └─ ap_gen_patch
                │                            (with fine-grained escalate_feedback_to)
```

The experiment can now measure two independent effects:
1. **Decomposition effect**: Does breaking analysis into localize + characterize improve outcomes? (Compare column pairs at same row)
2. **Strategy effect**: Does TDD improve over direct? (Compare rows within same column)

---

## 4. New Workflow Definitions

### 4a. `07_s1_direct_granular.json`

```json
{
  "name": "S1 Direct Granular",
  "description": "Decomposed analysis: localize then characterize, then patch. Tests whether splitting analysis improves fix quality.",
  "rmax": 3,
  "action_pairs": {
    "ap_localize": {
      "generator": "LocalizationGenerator",
      "guard": "localization",
      "guard_config": {
        "require_files": true,
        "min_files": 1,
        "max_files": 5
      },
      "description": "Identify files and functions to modify"
    },
    "ap_characterize": {
      "generator": "CharacterizationGenerator",
      "guard": "characterization",
      "requires": ["ap_localize"],
      "description": "Classify bug, identify root cause, describe fix approach"
    },
    "ap_patch": {
      "generator": "PatchGenerator",
      "guard": "patch",
      "guard_config": {
        "require_git_apply": false,
        "require_syntax_valid": true
      },
      "requires": ["ap_localize", "ap_characterize"],
      "description": "Generate patch based on localization and characterization"
    }
  }
}
```

### 4b. `08_s1_tdd_granular.json`

```json
{
  "name": "S1 TDD Granular",
  "description": "Decomposed analysis + TDD: localize, characterize, write failing test, then fix.",
  "rmax": 3,
  "action_pairs": {
    "ap_localize": {
      "generator": "LocalizationGenerator",
      "guard": "localization",
      "guard_config": {
        "require_files": true,
        "min_files": 1,
        "max_files": 5
      },
      "description": "Identify files and functions to modify"
    },
    "ap_characterize": {
      "generator": "CharacterizationGenerator",
      "guard": "characterization",
      "requires": ["ap_localize"],
      "description": "Classify bug, identify root cause, extract expected/actual behavior"
    },
    "ap_gen_test": {
      "generator": "TestGenerator",
      "guard": "test_syntax",
      "requires": ["ap_localize", "ap_characterize"],
      "description": "Write a failing test that reproduces the bug"
    },
    "ap_gen_fix": {
      "generator": "PatchGenerator",
      "guard": "patch",
      "guard_config": {
        "require_git_apply": false,
        "require_syntax_valid": true
      },
      "requires": ["ap_localize", "ap_characterize", "ap_gen_test"],
      "description": "Generate patch to fix the bug and pass the test"
    }
  }
}
```

### 4c. `09_s1_tdd_verified_granular.json`

```json
{
  "name": "S1 TDD Verified Granular",
  "description": "Decomposed analysis + TDD + Docker verification + informed backtracking with fine-grained escalation routing.",
  "rmax": 6,
  "action_pairs": {
    "ap_localize": {
      "generator": "LocalizationGenerator",
      "guard": "localization",
      "guard_config": {
        "require_files": true,
        "min_files": 1,
        "max_files": 5
      },
      "description": "Identify files and functions to modify"
    },
    "ap_characterize": {
      "generator": "CharacterizationGenerator",
      "guard": "characterization",
      "requires": ["ap_localize"],
      "r_patience": 2,
      "e_max": 1,
      "escalate_feedback_to": ["ap_localize"],
      "description": "Classify bug, identify root cause, extract expected/actual behavior"
    },
    "ap_gen_test": {
      "generator": "TestGenerator",
      "guard": "composite",
      "guards": ["test_syntax", "test_red"],
      "requires": ["ap_localize", "ap_characterize"],
      "r_patience": 2,
      "e_max": 1,
      "escalate_feedback_to": ["ap_characterize"],
      "escalate_feedback_by_guard": {
        "TestRedGuard": ["ap_characterize"]
      },
      "description": "Generate test + verify it FAILS on buggy code (red phase)"
    },
    "ap_gen_patch": {
      "generator": "PatchGenerator",
      "guard": "composite",
      "guards": ["patch", "lint", "test_green", "full_eval"],
      "guard_config": {
        "require_git_apply": true,
        "require_syntax_valid": true
      },
      "requires": ["ap_localize", "ap_characterize", "ap_gen_test"],
      "r_patience": 3,
      "e_max": 2,
      "escalate_feedback_to": ["ap_characterize", "ap_gen_test"],
      "escalate_feedback_by_guard": {
        "PatchGuard": ["ap_characterize"],
        "LintGuard": ["ap_characterize"],
        "TestGreenGuard": ["ap_characterize", "ap_gen_test"],
        "FullEvalGuard": ["ap_localize", "ap_characterize"]
      },
      "description": "Generate patch + verify test PASSES + run full eval"
    }
  }
}
```

---

## 5. New Components Required

### 5a. `CharacterizationGenerator`

Focused generator that receives localization output and produces structured characterization:
- Bug classification (off-by-one, null-pointer, API misuse, etc.)
- Root-cause hypothesis referencing localized files
- Fix approach with concrete steps
- Expected/actual behavior (for TDD downstream)

**Key difference from `AnalysisGenerator`**: Receives `ap_localize` artifact as dependency. Does NOT do file discovery — that's already done.

### 5b. `characterization` guard

Validates the characterization artifact:
- Root-cause hypothesis is present and references at least one localized file
- Fix approach is concrete (not vague)
- Bug classification is from the defined taxonomy
- Expected/actual behavior is extractable (for TDD variants)

JSON schema validation similar to the existing `analysis` guard.

### 5c. Prompt template: `ap_characterize`

```json
{
  "ap_characterize": {
    "role": "You are a senior software engineer specializing in bug diagnosis.",
    "constraints": "Given the localized files and functions, characterize the bug.\n\nREQUIREMENTS:\n- Classify the bug type from: off_by_one, null_reference, missing_check, wrong_logic, api_misuse, concurrency, type_error, other\n- Root-cause hypothesis MUST reference specific localized files/functions\n- Fix approach must describe concrete code changes\n- Extract EXPECTED BEHAVIOR vs ACTUAL BEHAVIOR for test generation\n\nYou do NOT need to search for files — localization is provided as input.",
    "task": "Characterize the bug: classify type, identify root cause in the localized files, describe fix approach.",
    "feedback_wrapper": "CHARACTERIZATION REJECTED:\n{feedback}\n\nInstruction: Revise your characterization. Ensure root_cause_hypothesis references localized files and fix_approach is concrete."
  }
}
```

### 5d. Updated `ap_localize` prompt

The existing `ap_localize` prompt (from `prompts.json`) is already suitable. No changes needed — it focuses on file/function identification with a localization guard.

---

## 6. Context Enrichment Prerequisite

The decomposition requires **context enrichment** (identified in the experiment plan as a critical gap). Currently:

```
DualStateAgent._compose_context() → dependency_artifacts = tuple((k, v.artifact_id) for k, v in deps.items())
```

Only artifact **IDs** flow to generators, not content. For `ap_characterize` to see the localization output, and for `ap_gen_test` to see both localization and characterization, the generator must receive the actual artifact content.

**Resolution**: Option A from the experiment plan — resolve dependency artifact IDs into content in `PromptTemplate.render()`. This is ~5 LOC and unblocks all decomposed pipelines. Must be implemented before this plan.

---

## 7. Implementation Steps

### Step 0: Document the approach (this plan)

Write `docs/design/plans/decompose_ap_analysis.md` explaining:
- The problem with monolithic analysis
- The granular decomposition approach
- Pipeline progression diagram (monolithic vs. decomposed × simple vs. TDD)
- How `escalate_feedback_to` and `escalate_feedback_by_guard` enable targeted backtracking
- New workflow definitions with examples
- Expected experimental arms

This document serves as the design rationale for the decomposition and can be referenced from the experiment plan.

### Step 1: Implement context enrichment

Fix the dependency content gap in `DualStateAgent._compose_context()` or `PromptTemplate.render()` so generators see upstream artifact content, not just IDs.

**Files**: `src/atomicguard/application/agent.py` or prompt template rendering
**Tests**: Verify generator receives dependency content in multi-step workflow

### Step 2: Add `CharacterizationGenerator` and `characterization` guard

- Generator: Focused prompt that takes localization as input
- Guard: JSON schema validation for characterization structure

**Files**: New generator + guard classes, or config-driven via existing registry
**Tests**: Unit tests for guard validation, integration test for generator → guard flow

### Step 3: Add `ap_characterize` prompt template

Add to `prompts.json` alongside existing prompts.

**Files**: `examples/swe_bench_pro/prompts.json`, `examples/swe_bench_ablation/prompts.json`

### Step 4: Create granular workflow JSON files

- `07_s1_direct_granular.json`
- `08_s1_tdd_granular.json`
- `09_s1_tdd_verified_granular.json`

**Files**: `examples/swe_bench_ablation/workflows/`

### Step 5: Update experiment runner

Ensure the experiment runner handles workflows with 4 action pairs. Verify topological sort, dependency passing, and escalation routing work with the new step graph.

**Files**: `examples/swe_bench_common/config.py`, experiment runners

### Step 6: Add escalation tests for granular pipelines

Test that:
- `ap_characterize` stagnation escalates to `ap_localize`
- `ap_gen_test` stagnation escalates to `ap_characterize` (not `ap_localize`)
- `ap_gen_patch` guard-specific escalation routes correctly
- Cascade invalidation propagates through 4-step chain

**Files**: `tests/application/test_workflow_escalation.py`

### Step 7: Update experiment plan

Add granular arms to the experiment plan (Section 7.2). The experiment now has a 2×3 design:

| | Single-shot | S1 Direct | S1-TDD |
|---|---|---|---|
| **Monolithic** | Arm 1 (02) | Arm 2 (03) | Arm 3 (04) |
| **Granular** | — | Arm 4 (07) | Arm 5 (08) |

With optional verified variants (05/06/09) for the backtracking comparison.

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Extra LLM call adds latency/cost | Medium | Localization is cheap (short output). Measure token delta. |
| Localization → characterization handoff loses context | Low | Context enrichment (Step 1) ensures content flows. Test explicitly. |
| Characterization guard too strict/loose | Medium | Start with schema validation only. Tune thresholds from initial runs. |
| 4-step cascade invalidation is slow | Low | Cascade is rare (requires stagnation + escalation). Budget via `e_max`. |
| `ap_localize` already exists in 01_baseline | None | Reuse the same generator and guard. This is intentional convergence. |

---

## 9. Success Criteria

- All existing tests pass (285 core, 44 skipped)
- New workflow JSON files validate against `workflow.schema.json`
- Granular pipeline produces equivalent or better results than monolithic on a sample of SWE-PolyBench instances
- Escalation tests demonstrate targeted backtracking (e.g., re-characterize without re-localizing)
- Token consumption delta is measured and documented
