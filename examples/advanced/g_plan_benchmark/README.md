# G_plan Validation Benchmark

Empirical validation of the G_plan taxonomy (Minimal / Medium / Expansive) and decomposed contingent planning pipeline from the ISMIS 2026 paper *Dynamic Neuro-Symbolic Control*.

## Overview

Plan generation is itself an Atomic Action Pair: `A_plan = ⟨ρ, a_gen_plan, G_plan⟩`. This benchmark implements G_plan as three `GuardInterface` instances at increasing rigor levels, measures their defect detection capability against plans with known injected defects, and provides a **decomposed planning pipeline** that breaks plan generation into individually guarded steps.

All validation uses real AtomicGuard infrastructure — `GuardInterface`, `Artifact`, `GuardResult`, and `GeneratorInterface` — not standalone simulation code.

## The G_plan Taxonomy

| Rigor Level | Predicates | Complexity | Guard Class |
|-------------|------------|------------|-------------|
| **Minimal** | `parseable ∧ is_dag ∧ guard_exists ∧ budget_defined` | O(V + E) | `MinimalPlanGuard` |
| **Medium** | Minimal + `reachable ∧ precond_satisfiable ∧ path_exists` | O(V × L) | `MediumPlanGuard` |
| **Expansive** | Medium + `∀π: terminates ∧ safe ∧ invariant_holds` | O(R^K) | `ExpansivePlanGuard` |

**Minimal** checks structural properties only: valid JSON, no dependency cycles (Kahn's algorithm), all guards in the catalog, and retry budgets > 0.

**Medium** adds semantic analysis: walks the topological order accumulating state tokens, verifies each step's preconditions are satisfiable by initial state + prior effects, and confirms goal tokens are reachable.

**Expansive** explores the state space with retry branching, physically manifesting the O(R^K) complexity cliff.

## Plan Representation

Plans bridge `workflow.json` (the format AtomicGuard uses) to a PDDL-style precondition/effect model:

- Each step's `guard_id` becomes its **effect token** (what it produces)
- Each step's `requires` entries become **precondition tokens** (what it needs)
- `initial_state` = `{intent_received}` (workflow entry point)
- `goal_state` = effect tokens of terminal steps

The `PlanDefinition.from_workflow_json()` loader constructs this representation from a real `workflow.json`, so the benchmark can validate actual AtomicGuard workflows.

## Defect Injection

Eight defect types target specific predicates in the taxonomy:

| Defect | Description | Caught By |
|--------|-------------|-----------|
| `cycle` | Circular dependency between steps | Minimal |
| `missing_guard` | Guard reference not in catalog | Minimal |
| `zero_retry` | Step with retry_budget = 0 | Minimal |
| `unreachable_goal` | Goal token no step produces | Medium |
| `unsatisfiable_precond` | Precondition nothing satisfies | Medium |
| `missing_initial` | Empty initial state | Medium |
| `budget_overflow` | Step budgets exceed total | Medium |
| `orphan_step` | Step with no path to goal | Medium |

The first three are structural defects (caught by Minimal). The remaining five are semantic — the plan looks structurally valid but can never converge. These require Medium's precondition/effect chain analysis.

## Results

```
Detection Summary (100 trials per defect):
  Minimal:   3/8 (38%)
  Medium:    8/8 (100%)
  Expansive: 8/8 (100%)
```

Medium rigor is necessary and sufficient for production use. Minimal misses critical semantic defects. Expansive adds exponential cost without additional detection for these defect classes.

### Complexity Cliff

```
   Validation Time by Plan Size (ms)
┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ Steps ┃ Minimal ┃ Medium ┃ Expansive ┃
┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│     5 │   0.022 │  0.049 │     0.112 │
│    10 │   0.039 │  0.091 │     0.455 │
│    20 │   0.077 │  0.300 │    41.935 │
│    50 │   0.160 │  0.411 │     0.413 │
│   100 │   0.320 │  0.924 │     0.909 │
└───────┴─────────┴────────┴───────────┘
```

The cliff appears at K=20 for Expansive (41.9ms vs 0.3ms for Medium). At K=50+ the safety bound (MAX_EXPLORATIONS=100,000) caps computation, demonstrating that without such a bound the cost would be intractable.

## Decomposed Planning Pipeline

Plan generation is decomposed into individually guarded steps, each compressing the search space for the next. This is an instance of Extension 06 (Generated Workflows) — the planning workflow is itself a workflow.

### Pipeline Architecture

```
g_analysis ──→ g_recon ──→ g_strategy ──→ g_plan_full
(classify)    (extract)    (select S1-S5)  (generate plan)
```

Each step is its own action pair with its own guard. If the problem classifier produces invalid output, it gets caught and retried before the planner runs. Context flows via `Context.amend(delta_constraints=...)`, enriching each subsequent prompt.

### Pipeline Steps

| Step | Generator | Guard | Purpose | Output |
|------|-----------|-------|---------|--------|
| `g_analysis` | `LLMJsonGenerator` | `AnalysisGuard` | Classify problem type, language, severity, key signals | Problem classification JSON |
| `g_recon` | `LLMJsonGenerator` | `ReconGuard` | Extract mentioned files, stack traces, APIs, test references, reproduction steps | Codebase reconnaissance JSON |
| `g_strategy` | `LLMJsonGenerator` | `StrategyGuard` | Select resolution strategy (S1-S5) with rationale and key steps | Strategy selection JSON |
| `g_plan_full` | `LLMPlanGenerator` | `MediumPlanGuard` | Generate plan conditioned on analysis, recon, and strategy | Workflow plan JSON |

### Pre-Step Guards

All pre-step guards are O(1) JSON validation:

- **AnalysisGuard** — 7 predicates: parseable JSON, valid `problem_type` (bug_fix / feature / refactoring / performance), `language` present, `key_signals` non-empty, `severity` valid, `affected_area` present, `rationale` present
- **ReconGuard** — Validates 6 required list fields (`mentioned_files`, `stack_traces`, `apis_involved`, `test_references`, `reproduction_steps`, `constraints_mentioned`), at least one non-empty
- **StrategyGuard** — Validates `strategy_id` in S1-S5 vocabulary, `strategy_name`, `rationale`, `key_steps` (non-empty list), `expected_guards` (non-empty list), `risk_factors` (list)

### Strategy Vocabulary

| ID | Name | Suited For |
|----|------|-----------|
| `S1_locate_and_fix` | Locate and Fix | Bug fixes |
| `S2_tdd_feature` | TDD Feature | New features |
| `S3_refactor_safely` | Refactor Safely | Refactoring |
| `S4_profile_and_optimize` | Profile and Optimize | Performance |
| `S5_investigate_first` | Investigate First | Unclear problems |

### Pipeline Modes

Three modes are available via `--pipeline`:

| Mode | Steps | LLM Calls (happy path) | Description |
|------|-------|------------------------|-------------|
| `single` | 1 | 1 | Single-shot plan generation (`g_plan_llm`) |
| `classify-then-plan` | 2 | 2 | `g_analysis → g_plan_conditioned` |
| `full` | 4 | 4 | `g_analysis → g_recon → g_strategy → g_plan_full` |

## Usage

```bash
# Validate a plan at all three rigor levels
uv run python -m examples.advanced.g_plan_benchmark.demo validate

# Validate the real sdlc_v2 workflow.json
uv run python -m examples.advanced.g_plan_benchmark.demo validate --from-workflow

# Run defect detection benchmark (Table 1 in paper)
uv run python -m examples.advanced.g_plan_benchmark.demo benchmark --trials 100

# Run defect detection against real workflow.json
uv run python -m examples.advanced.g_plan_benchmark.demo benchmark --from-workflow --trials 100

# Run complexity scaling benchmark (complexity cliff figure)
uv run python -m examples.advanced.g_plan_benchmark.demo complexity --trials 100

# Save results to JSON
uv run python -m examples.advanced.g_plan_benchmark.demo benchmark --output results.json

# Estimate epsilon for LLM plan generation (requires Ollama)
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon --trials 20

# Epsilon with specific model and host
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon \
    --trials 20 --host http://localhost:11434 --model qwen2.5-coder:14b

# Classify-then-plan pipeline
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon \
    --trials 20 --pipeline classify-then-plan

# Full decomposed pipeline
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon \
    --trials 20 --pipeline full --model qwen2.5-coder:14b

# HuggingFace backend
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon \
    --trials 20 --backend huggingface --model Qwen/Qwen2.5-Coder-32B-Instruct

# Verbose output (per-trial details + debug logging)
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon --trials 20 -v

# Save epsilon results to JSON
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon --trials 20 --output epsilon.json
```

## File Structure

```
g_plan_benchmark/
├── demo.py                      # CLI: validate, benchmark, complexity, epsilon
├── models.py                    # PlanDefinition, PlanStep (PDDL bridge)
├── defects.py                   # DefectType enum + inject_defect()
├── workflow.json                # AtomicGuard workflow config (all pipeline modes)
├── prompts.json                 # Prompt templates (deterministic + LLM + pipeline steps)
├── guards/
│   ├── minimal.py               # MinimalPlanGuard(GuardInterface) — O(V+E)
│   ├── medium.py                # MediumPlanGuard(GuardInterface) — O(V×L)
│   ├── expansive.py             # ExpansivePlanGuard(GuardInterface) — O(R^K)
│   ├── analysis.py              # AnalysisGuard — problem classification validation
│   ├── recon.py                 # ReconGuard — codebase reconnaissance validation
│   └── strategy.py              # StrategyGuard — strategy selection validation
├── generators/
│   ├── plan_generator.py        # PlanGenerator(GeneratorInterface) — deterministic
│   ├── llm_plan_generator.py    # LLMPlanGenerator(GeneratorInterface) — LLM-backed plan gen
│   └── llm_json_generator.py    # LLMJsonGenerator(GeneratorInterface) — LLM-backed JSON gen
└── plans/
    ├── sdlc_v2.json             # 6-step SDLC plan (derived from real workflow.json)
    └── simple.json              # 3-step linear plan
```

## Design Decisions

### Deterministic Generation for Guard Benchmarking

The benchmark tests **guard detection capability**, not generator quality. Using an LLM to generate plans would confound the experiment: you'd measure `P(defect | LLM) × P(detection | guard, defect)` instead of isolating the second term. Deterministic generation with injected defects gives a clean measurement.

The `PlanGenerator` follows the same pattern as `RulesExtractorGenerator` in `sdlc_v2` — a `GeneratorInterface` that produces artifacts without an LLM call. AtomicGuard treats deterministic generators as first-class.

### Why Not a Standalone Script

The original `g_plan_benchmark.py` reimplemented plan representation, guard logic, and validation from scratch without importing any AtomicGuard code. This benchmark uses the real framework:

| Component | AtomicGuard Integration |
|-----------|------------------------|
| Guards | `GuardInterface.validate() → GuardResult` |
| Generator | `GeneratorInterface.generate() → Artifact` |
| Plan content | `Artifact.content` (JSON string) |
| Plan loading | `PlanDefinition.from_workflow_json()` reads real configs |
| Defect injection | Operates on plan dicts before `Artifact` wrapping |

### Decomposed Pipeline Design

The decomposed pipeline breaks single-shot plan generation into individually guarded steps. Each step compresses the search space for the next:

1. **g_analysis** narrows from "any problem" to a classified problem type
2. **g_recon** grounds the plan in concrete codebase signals (files, traces, APIs)
3. **g_strategy** narrows from "any plan shape" to a specific strategy template (S1-S5)
4. **g_plan_full** generates a plan with all three as conditioning context

Each step uses `LLMJsonGenerator` (a thin subclass of `LLMPlanGenerator`) paired with its own O(1) guard. The guards catch invalid output before it pollutes downstream steps. Context flows via `Context.amend(delta_constraints=...)`, injecting each step's output into the prompt for the next step.

### LLM Epsilon Estimation

The `epsilon` command estimates the pass rate of LLM-generated plans against G_plan guards:

```
epsilon_hat = (plans passing G_plan) / (total generated)
```

This measures how reliably an LLM can produce structurally and semantically valid workflow plans. Results include:

- **epsilon-hat** per rigor level (Minimal / Medium / Expansive)
- **95% Wilson confidence intervals** for each estimate
- **E[attempts]** = 1/epsilon (expected retries to get a valid plan)
- **Per-step pass rates** for multi-step pipelines (analysis, recon, strategy)
- **Common failure modes** — frequency analysis of guard rejection reasons

The guards validate identically regardless of artifact source — deterministic or LLM-generated.
