# Planning Workflow Decomposition

> **Status**: Implemented
>
> **Implementation**: `examples/advanced/g_plan_benchmark/`
>
> **Core tradeoff**: More steps per pipeline means higher per-step epsilon (simpler task per LLM call, fewer retries) but more total latency. Fewer steps means faster per run but lower plan quality and more retries.

---

## Decomposed Pipeline

Plan generation is decomposed into a 4-step pipeline where each step compresses the search space for the next. This is an instance of Extension 06 (Generated Workflows) — the planning workflow is itself a workflow.

```
g_analysis ──→ g_recon ──→ g_strategy ──→ g_plan_full
(classify)    (extract)    (select S1-S5)  (generate plan)
```

**Steps**: 4 action pairs
**Max LLM calls** (with rmax=3): 12
**LLM calls** (happy path): 4
**Latency**: Medium-High (sequential dependency chain)

### Step 1: Problem Analysis (`g_analysis`)

Classify the problem type (bug fix / feature / refactoring / performance), identify the language, estimate scope. This is the highest-epsilon step — classification is straightforward for LLMs.

- **Generator**: `LLMJsonGenerator`
- **Guard**: `AnalysisGuard` (7 predicates, O(1))
- **Output**: `{ problem_type, language, severity, key_signals, affected_area, rationale }`

### Step 2: Codebase Reconnaissance (`g_recon`)

Extract actionable context from the problem description: mentioned files, stack traces, APIs, test references, reproduction steps, constraints. Grounds subsequent steps in concrete codebase signals rather than abstract reasoning.

- **Generator**: `LLMJsonGenerator`
- **Guard**: `ReconGuard` (6 required list fields, at least one non-empty, O(1))
- **Requires**: `g_analysis`
- **Output**: `{ mentioned_files, stack_traces, apis_involved, test_references, reproduction_steps, constraints_mentioned }`

### Step 3: Strategy Selection (`g_strategy`)

Given the analysis from steps 1 and 2, select a strategy template from the S1-S5 vocabulary. This narrows the plan generator from "any plan shape" to a specific approach.

- **Generator**: `LLMJsonGenerator`
- **Guard**: `StrategyGuard` (validates strategy_id in vocabulary, O(1))
- **Requires**: `g_analysis`, `g_recon`
- **Output**: `{ strategy_id, strategy_name, rationale, key_steps, expected_guards, risk_factors }`

**Strategy vocabulary**:

| ID | Name | Suited For |
|----|------|-----------|
| `S1_locate_and_fix` | Locate and Fix | Bug fixes |
| `S2_tdd_feature` | TDD Feature | New features |
| `S3_refactor_safely` | Refactor Safely | Refactoring |
| `S4_profile_and_optimize` | Profile and Optimize | Performance |
| `S5_investigate_first` | Investigate First | Unclear problems |

### Step 4: Plan Generation (`g_plan_full`)

The LLM generates a concrete workflow plan, but now it has the problem classification, codebase context, and selected strategy as conditioning inputs. The search space is dramatically smaller than single-shot generation.

- **Generator**: `LLMPlanGenerator`
- **Guard**: `MediumPlanGuard` (O(V × L))
- **Requires**: `g_analysis`, `g_recon`, `g_strategy`
- **Output**: Workflow plan JSON

### Context Flow

Each step's output is injected into the next step's context via `Context.amend(delta_constraints=...)`, appearing in the `# CONTEXT` section of `PromptTemplate.render()`. Each injection is prefixed with a header (e.g., `## Problem Analysis (from g_analysis)`) for clarity.

### workflow.json

```json
{
  "g_analysis": {
    "generator": "LLMJsonGenerator",
    "guard": "analysis_valid",
    "description": "Classify the problem type, language, severity, and key signals"
  },
  "g_recon": {
    "generator": "LLMJsonGenerator",
    "guard": "recon_valid",
    "requires": ["g_analysis"],
    "description": "Extract actionable context: files, traces, APIs, test references"
  },
  "g_strategy": {
    "generator": "LLMJsonGenerator",
    "guard": "strategy_valid",
    "requires": ["g_analysis", "g_recon"],
    "description": "Select a resolution strategy (S1-S5)"
  },
  "g_plan_full": {
    "generator": "LLMPlanGenerator",
    "guard": "plan_medium",
    "requires": ["g_analysis", "g_recon", "g_strategy"],
    "description": "Generate a plan conditioned on analysis, recon, and strategy"
  }
}
```

---

## Pipeline Modes

Three pipeline modes are implemented in `demo.py` via `--pipeline`:

| Mode | Steps | Happy-path LLM Calls | Description |
|------|-------|---------------------|-------------|
| `single` | 1 | 1 | Single-shot generation (`g_plan_llm`) |
| `classify-then-plan` | 2 | 2 | `g_analysis → g_plan_conditioned` |
| `full` | 4 | 4 | `g_analysis → g_recon → g_strategy → g_plan_full` |

The `classify-then-plan` mode is a subset of the full pipeline — it uses only `g_analysis` before generating a plan. This provides a useful middle ground: problem-type-aware planning with only 2 LLM calls.

---

## Design Rationale

### Why Decomposition

A single LLM call must simultaneously: understand the problem type, extract relevant context, choose a strategy, and produce a valid DAG plan. Decomposition separates these concerns:

| Concern | Step | Guard |
|---------|------|-------|
| What kind of problem? | g_analysis | AnalysisGuard |
| What's in the codebase? | g_recon | ReconGuard |
| Which approach? | g_strategy | StrategyGuard |
| What's the plan? | g_plan_full | MediumPlanGuard |

Each step has its own guard, so failures are caught and retried at the step that produced them — not at the final plan validation where the root cause is obscured.

### Why Sequential (Not Parallel)

The dependency chain `g_analysis → g_recon → g_strategy → g_plan_full` is intentionally sequential. Each step's output constrains the next:

- Recon needs the problem type to know what signals to prioritize
- Strategy needs both analysis and recon to make an informed selection
- Plan generation needs all three as context

Parallel execution (e.g., running analysis and recon simultaneously) would remove the conditioning benefit. The latency cost is acceptable because each pre-step is a lightweight JSON extraction with an O(1) guard.

### Why S1-S5 Vocabulary

The strategy vocabulary is derived from the paper's guard catalog and real-world practice:

- **S1** (Locate-and-Fix): search → characterization test → patch → verify regression
- **S2** (TDD Feature): design tests → implement → validate
- **S3** (Refactor Safely): ensure coverage → extract/restructure → verify no regression
- **S4** (Profile and Optimize): profile → benchmark test → optimize → verify
- **S5** (Investigate First): reproduce → characterize → decide approach

These map to the `task_category` field in SWE-PolyBench (Bug Fix → S1, Feature → S2, Refactoring → S3), enabling strategy alignment scoring in benchmark evaluation.

---

## Future Directions

Two additional decomposition patterns remain available for experimentation:

### Template Refinement

Instead of LLM-generating a plan from scratch, deterministically select a plan template from a catalog based on the strategy, then have the LLM adapt it. The template selection step has epsilon=1.0 (deterministic), and the LLM only refines rather than generates from scratch.

```
g_analysis → g_template (deterministic) → g_plan_refine (LLM)
```

### Draft-Critique-Revise

Generate a draft plan (validated at Minimal level), critique it with a separate LLM call, then generate a revised plan incorporating the critique. This implements the paper's Tier 1 (Coach) concept.

```
g_plan_draft → g_critique → g_plan_final
```

Both can be combined with the current pipeline for deeper decomposition.
