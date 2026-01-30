# Planning Workflow Decomposition Options

> **Context**: The current `workflow.json` has a single action pair (`g_plan_llm`) that takes a raw problem statement and generates a complete plan in one shot. This document explores multi-step decompositions.
>
> **Core tradeoff**: More steps → higher per-step epsilon (simpler task per LLM call, fewer retries) but more total latency. Fewer steps → faster per run but lower plan quality and more retries.

---

## Option A: Classify-then-Plan (minimal decomposition)

**Steps**: 2 action pairs
**Max LLM calls** (with rmax=3): 6
**Latency**: Low

```
g_analysis ──→ g_plan
```

The simplest useful decomposition. Classify the problem first, then generate a plan conditioned on the classification. The analysis step is high-epsilon (classification is easy for LLMs) and constrains the plan generator so it doesn't have to figure out what kind of problem it's looking at.

### workflow.json

```json
{
  "name": "Contingent Planning (Classify-then-Plan)",
  "description": "Two-step planning: classify the problem, then generate a strategy-appropriate plan.",
  "rmax": 3,
  "action_pairs": {
    "g_analysis": {
      "generator": "LLMJsonGenerator",
      "guard": "analysis_valid",
      "description": "Classify the problem type, language, and key signals from the problem statement"
    },
    "g_plan": {
      "generator": "LLMPlanGenerator",
      "guard": "plan_medium",
      "requires": ["g_analysis"],
      "description": "Generate a workflow plan conditioned on the problem analysis"
    }
  }
}
```

### prompts.json (g_analysis)

```json
{
  "g_analysis": {
    "role": "You are a software engineering triage specialist.",
    "constraints": "Analyze the problem and return a JSON object:\n{\n  \"problem_type\": \"bug_fix\" | \"feature\" | \"refactoring\" | \"performance\",\n  \"language\": \"python\" | \"java\" | \"javascript\" | \"typescript\" | \"unknown\",\n  \"severity\": \"low\" | \"medium\" | \"high\",\n  \"key_signals\": [\"list of error messages, stack traces, or feature keywords found\"],\n  \"affected_area\": \"brief description of what part of the system is involved\",\n  \"rationale\": \"one sentence explaining the classification\"\n}\n\nReturn ONLY the JSON object.",
    "task": "Analyze this problem and classify it.",
    "feedback_wrapper": "ANALYSIS REJECTED:\n{feedback}\n\nFix the analysis to address the issues above."
  }
}
```

### Guard: analysis_valid

Validates: parseable JSON, `problem_type` is one of the allowed values, `language` present, `key_signals` is a non-empty list. Lightweight — O(1) checks.

### Tradeoffs

| Pro | Con |
|-----|-----|
| Only 2 LLM calls on the happy path | Plan generator still does most of the work |
| Classification is high-epsilon (~95%+) | No codebase context for the planner |
| Simple to implement and test | Strategy selection is implicit in the plan prompt |
| Low latency | |

---

## Option B: Parallel Analysis + Plan

**Steps**: 3 action pairs (2 parallel + 1 sequential)
**Max LLM calls** (with rmax=3): 9
**Latency**: Medium (but analysis + recon run in parallel)

```
g_analysis ──┐
             ├──→ g_plan
g_recon    ──┘
```

Analysis (classify the problem) and reconnaissance (extract context from the problem description — mentioned files, stack traces, APIs, error patterns) run in parallel with no dependency between them. The plan generator receives both as inputs.

### workflow.json

```json
{
  "name": "Contingent Planning (Parallel Analysis)",
  "description": "Parallel problem analysis and context extraction, then plan generation from both.",
  "rmax": 3,
  "action_pairs": {
    "g_analysis": {
      "generator": "LLMJsonGenerator",
      "guard": "analysis_valid",
      "description": "Classify the problem type, language, and severity"
    },
    "g_recon": {
      "generator": "LLMJsonGenerator",
      "guard": "recon_valid",
      "description": "Extract contextual signals from the problem description: mentioned files, stack traces, APIs, test references"
    },
    "g_plan": {
      "generator": "LLMPlanGenerator",
      "guard": "plan_medium",
      "requires": ["g_analysis", "g_recon"],
      "description": "Generate a workflow plan conditioned on problem analysis and extracted context"
    }
  }
}
```

### prompts.json (g_recon)

```json
{
  "g_recon": {
    "role": "You are a software engineering investigator. Your job is to extract actionable context from a problem description.",
    "constraints": "Return a JSON object:\n{\n  \"mentioned_files\": [\"file paths or module names mentioned in the problem\"],\n  \"stack_traces\": [\"any stack traces or error outputs, summarized\"],\n  \"apis_involved\": [\"function names, class names, or APIs referenced\"],\n  \"test_references\": [\"any test files, test names, or testing patterns mentioned\"],\n  \"reproduction_steps\": [\"steps to reproduce if described, otherwise empty\"],\n  \"constraints_mentioned\": [\"any performance, compatibility, or design constraints\"]\n}\n\nExtract ONLY what is explicitly stated or directly implied. Do not infer or guess.\nReturn ONLY the JSON object.",
    "task": "Extract actionable context from this problem description.",
    "feedback_wrapper": "RECON REJECTED:\n{feedback}\n\nFix the extraction to address the issues above."
  }
}
```

### Guard: recon_valid

Validates: parseable JSON, all expected keys present, arrays (not strings) for list fields. Does NOT validate that files exist (we may not have the codebase). Lightweight — O(1).

### Tradeoffs

| Pro | Con |
|-----|-----|
| Analysis + recon run in parallel (same wall-clock as Option A) | 3 action pairs to maintain |
| Recon gives the planner grounded context (file names, stack traces) | Recon quality varies — sparse problem statements yield sparse recon |
| Plan generator gets structured inputs, not raw text | Slightly more complex dependency graph |
| Each pre-step is focused and high-epsilon | |

---

## Option C: Template Selection + Refinement (hybrid deterministic/LLM)

**Steps**: 3 action pairs (1 LLM + 1 deterministic + 1 LLM)
**Max LLM calls** (with rmax=3): 6 (deterministic step never retries)
**Latency**: Medium

```
g_analysis ──→ g_template (deterministic) ──→ g_plan_refine
```

Classify the problem, deterministically select a plan template from a catalog (no LLM — like `PlanGenerator` today), then have the LLM adapt the template to the specific problem. The middle step has epsilon=1.0 because it's deterministic.

### workflow.json

```json
{
  "name": "Contingent Planning (Template Refinement)",
  "description": "Classify problem, select a template deterministically, then LLM-refine for the specific problem.",
  "rmax": 3,
  "action_pairs": {
    "g_analysis": {
      "generator": "LLMJsonGenerator",
      "guard": "analysis_valid",
      "description": "Classify the problem type and language"
    },
    "g_template": {
      "generator": "TemplateSelectorGenerator",
      "generator_config": {
        "templates": {
          "bug_fix": "plans/bug_fix_template.json",
          "feature": "plans/feature_template.json",
          "refactoring": "plans/refactoring_template.json",
          "performance": "plans/performance_template.json"
        }
      },
      "guard": "plan_minimal",
      "requires": ["g_analysis"],
      "description": "Deterministically select a plan template based on problem classification (epsilon=1.0)"
    },
    "g_plan_refine": {
      "generator": "LLMPlanRefiner",
      "guard": "plan_medium",
      "requires": ["g_template", "g_analysis"],
      "description": "Adapt the template plan to the specific problem: adjust steps, add/remove as needed"
    }
  }
}
```

### prompts.json (g_plan_refine)

```json
{
  "g_plan_refine": {
    "role": "You are a workflow planner. You are given a template plan and a specific problem. Adapt the template to fit the problem.",
    "constraints": "You will receive:\n1. A problem analysis (type, language, key signals)\n2. A template plan (JSON) appropriate for this problem type\n\nYour job is to ADAPT the template, not generate from scratch:\n- Keep steps that are relevant\n- Remove steps that don't apply to this specific problem\n- Add steps if the problem requires something the template doesn't cover\n- Update step names and descriptions to be problem-specific\n- Ensure the DAG remains valid (all preconditions satisfiable)\n- Ensure retry budgets are reasonable\n\nReturn the adapted plan as a single JSON object following the same schema as the template.\nReturn ONLY the JSON object.",
    "task": "Adapt this template plan for the specific problem described.",
    "feedback_wrapper": "PLAN REFINEMENT REJECTED:\n{feedback}\n\nFix the plan to address the validation errors above. Keep it close to the template structure."
  }
}
```

### Plan templates (new files in plans/)

Each template is a pre-built plan skeleton for a problem type:

- `plans/bug_fix_template.json` — search → characterization test → fix → regression test
- `plans/feature_template.json` — design → test first → implement → validate
- `plans/refactoring_template.json` — ensure coverage → extract → verify no regression
- `plans/performance_template.json` — profile → benchmark → optimize → verify

### Tradeoffs

| Pro | Con |
|-----|-----|
| Deterministic template selection (epsilon=1.0, never fails) | Requires maintaining a template catalog |
| LLM refines rather than generates from scratch (easier, higher epsilon) | Templates may not cover unusual problem types |
| Strong structural guarantee — template is already valid | Refinement might break template validity (but G_plan catches this) |
| Cheapest: only 2 LLM calls, and refinement is constrained | Less creative — plans stay close to templates |
| Most testable — templates are fixed, refinement is bounded | Adding a new problem type requires a new template |

---

## Option D: Draft-Critique-Revise (reflective)

**Steps**: 3 action pairs (all LLM)
**Max LLM calls** (with rmax=3): 9
**Latency**: High

```
g_plan_draft ──→ g_critique ──→ g_plan_final
```

Generate a plan first (fast, might be flawed), then have a separate LLM call critique it (what's missing? what doesn't fit the problem?), then generate a revised plan incorporating the critique. This is a "reflection" pattern — the critique acts as a lightweight coach.

### workflow.json

```json
{
  "name": "Contingent Planning (Draft-Critique-Revise)",
  "description": "Generate a draft plan, critique it, then revise. The critique step acts as a lightweight coach.",
  "rmax": 3,
  "action_pairs": {
    "g_plan_draft": {
      "generator": "LLMPlanGenerator",
      "guard": "plan_minimal",
      "description": "Generate an initial plan draft (validated at Minimal level only — structural, not semantic)"
    },
    "g_critique": {
      "generator": "LLMJsonGenerator",
      "guard": "critique_valid",
      "requires": ["g_plan_draft"],
      "description": "Critique the draft plan: identify missing steps, mismatched strategy, unreachable goals"
    },
    "g_plan_final": {
      "generator": "LLMPlanGenerator",
      "guard": "plan_medium",
      "requires": ["g_plan_draft", "g_critique"],
      "description": "Generate a revised plan incorporating the critique feedback"
    }
  }
}
```

### prompts.json (g_critique)

```json
{
  "g_critique": {
    "role": "You are a plan reviewer for software development workflows.",
    "constraints": "You will receive a problem description and a draft workflow plan. Evaluate the plan and return a JSON critique:\n{\n  \"overall_assessment\": \"good\" | \"needs_revision\" | \"fundamentally_flawed\",\n  \"problem_type_match\": \"Does the plan strategy match the problem type? Explain.\",\n  \"missing_steps\": [\"steps that should be in the plan but are missing\"],\n  \"unnecessary_steps\": [\"steps that don't serve this specific problem\"],\n  \"ordering_issues\": [\"any dependency or sequencing problems\"],\n  \"guard_appropriateness\": [\"any guards that are wrong for their step\"],\n  \"specific_suggestions\": [\"concrete changes to make\"]\n}\n\nBe specific and actionable. Reference step IDs when possible.\nReturn ONLY the JSON object.",
    "task": "Critique this workflow plan for the given problem.",
    "feedback_wrapper": "CRITIQUE REJECTED:\n{feedback}\n\nProvide a valid critique addressing the issues above."
  }
}
```

### Guard: critique_valid

Validates: parseable JSON, `overall_assessment` is valid enum, arrays present for list fields, at least one field is non-empty (the critique says something). Lightweight.

### Tradeoffs

| Pro | Con |
|-----|-----|
| Draft uses Minimal guard (cheap, fast) — catches structural garbage early | 3 LLM calls minimum |
| Critique is domain-aware feedback before the expensive Medium validation | Critique quality varies — may be generic |
| Revision is targeted — the LLM knows what to fix | The revision step is the hardest (generate + incorporate feedback) |
| Maps to paper's Tier 1 (Coach) concept | Highest latency of all options |
| The critique itself is a testable artifact | Critique and final plan may not align (critique says X, plan ignores it) |

---

## Comparison Matrix

| | Option A | Option B | Option C | Option D |
|---|---|---|---|---|
| **Name** | Classify-then-Plan | Parallel Analysis | Template Refinement | Draft-Critique-Revise |
| **Steps** | 2 | 3 (2 parallel) | 3 (1 deterministic) | 3 |
| **LLM calls** (happy path) | 2 | 3 | 2 | 3 |
| **LLM calls** (worst case, rmax=3) | 6 | 9 | 6 | 9 |
| **Deterministic steps** | 0 | 0 | 1 (template select) | 0 |
| **Wall-clock latency** | Low | Medium (parallel) | Medium | High |
| **Plan quality** | Medium | High | High (constrained) | Highest |
| **Strategy awareness** | Implicit | Explicit (via recon) | Explicit (via template) | Implicit (via critique) |
| **Testability** | Good | Good | Best (templates fixed) | Good |
| **New generators needed** | 1 (LLMJsonGenerator) | 1 (LLMJsonGenerator) | 2 (TemplateSelector, PlanRefiner) | 1 (LLMJsonGenerator) |
| **New guards needed** | 1 (analysis_valid) | 2 (analysis_valid, recon_valid) | 1 (analysis_valid) | 1 (critique_valid) |
| **Best for** | Quick iteration, low cost | Rich problem context | Controlled, repeatable | Quality-critical plans |

---

## Combinations

These aren't mutually exclusive. The strongest configuration might combine elements:

**Option B + C** (Parallel Analysis + Template Refinement):
```
g_analysis ──┐
             ├──→ g_template (deterministic) ──→ g_plan_refine
g_recon    ──┘
```
4 steps, 3 LLM calls, 1 deterministic. Analysis and recon run in parallel. Template selected from analysis. Refinement uses both recon context and template.

**Option A + D** (Classify → Draft → Critique → Revise):
```
g_analysis ──→ g_plan_draft ──→ g_critique ──→ g_plan_final
```
4 steps, 4 LLM calls. Linear but each step is focused.

---

## Recommendation

**Start with Option A** (Classify-then-Plan) for minimum viable decomposition. It's 2 LLM calls, 1 new guard, 1 new generator, and already provides the key benefit: problem-type-aware planning.

**Graduate to Option C** (Template Refinement) if epsilon is too low — the deterministic template guarantees structural validity and the LLM only has to refine, not generate from scratch.

**Use Option D** (Draft-Critique-Revise) for the ISMIS paper's "feedback loop" experiment — the critique step directly demonstrates the paper's Tier 1 (Coach) concept.

**Option B** (Parallel Analysis) is the production choice — it extracts the most context and runs pre-steps in parallel.
