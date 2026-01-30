# Benchmark Evaluation Framework

> **Status**: Planned
>
> **Depends on**: [huggingface_finetuning_pipeline.md](huggingface_finetuning_pipeline.md), G_plan benchmark (`examples/advanced/g_plan_benchmark/`)
>
> **Paper**: [arXiv:2512.20660](https://arxiv.org/abs/2512.20660) — "Managing the Stochastic: Foundations of Learning in Neuro-Symbolic Systems for Software Engineering"

---

## 1. Goal

Validate the AtomicGuard contingent planning framework end-to-end against established software engineering benchmarks. The evaluation should demonstrate that:

1. **G_plan guards** correctly validate plans for diverse, real-world problems (not just synthetic defects)
2. **Contingent planning** (problem-specific strategy selection) improves resolution rates compared to fixed plans
3. **The full pipeline** — from problem description through plan generation, plan validation, plan execution, and code evaluation — produces measurable improvements

This builds on the paper's existing experiments (Section 7: diagnostic probes, TDD workflow benchmark, G_plan defect detection) by scaling to real-world codebases.

---

## 2. Paper Experiments as Foundation

The paper (arXiv:2512.20660) establishes four experimental results:

| Paper Experiment | What It Measures | Current Status |
|---|---|---|
| **Diagnostic Probes** (Section 7.1) | Guard-validated retry loops improve first-pass reliability (13 LLMs, 3 probes, up to +66pp) | Complete (benchmarks/) |
| **TDD Workflow** (Section 7.2) | Multi-step workflow execution with guards | Complete (benchmarks/) |
| **G_plan Defect Detection** (Experiment 1) | Guards detect 8 defect types across 3 rigor levels | Complete (g_plan_benchmark/) |
| **Epsilon Estimation** (Experiment 4) | LLM plan generation pass rate with Wilson CIs | Complete (g_plan_benchmark/ epsilon command) |

**What's missing**: all of the above use synthetic or small-scale inputs. The benchmarks below test against real-world codebases with diverse problem types.

---

## 3. Target Benchmarks

### 3.1 SWE-PolyBench (Primary)

**Why**: Explicit `task_category` field with 40% bug / 40% feature / 20% refactor split. This directly maps to the contingent planning hypothesis — different problem types should produce different plans.

| Property | Value |
|---|---|
| Paper | [arXiv:2504.08703](https://arxiv.org/abs/2504.08703) |
| Instances | 2,110 (full), **500 (PB500, recommended)** |
| Languages | Java, JavaScript, TypeScript, Python |
| Task types | Bug Fix, Feature, Refactoring (explicit `task_category` field) |
| Data format | HuggingFace Parquet: `problem_statement`, `repo`, `base_commit`, `patch`, `test_patch`, `F2P`, `P2P`, `task_category`, `language`, `Dockerfile`, `test_command` |
| Evaluation | Docker-based: apply patch → run tests → check F2P/P2P |
| HuggingFace | [`AmazonScience/SWE-PolyBench_500`](https://huggingface.co/datasets/AmazonScience/SWE-PolyBench_500) |
| GitHub | [`amazon-science/SWE-PolyBench`](https://github.com/amazon-science/SWE-PolyBench) |

**Key fields for planning**:
- `problem_statement` → specification input to planner
- `task_category` → ground truth for strategy selection (Bug Fix / Feature / Refactoring)
- `language` → constrains which generators/guards apply
- `patch` → gold standard for evaluating execution quality

### 3.2 SWE-bench Verified (Secondary)

**Why**: Most widely cited benchmark. Provides comparability with the broader literature. The `difficulty` field enables analysis by problem complexity.

| Property | Value |
|---|---|
| Paper | [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) (ICLR 2024 Oral) |
| Instances | **500** (human-validated) |
| Languages | Python only |
| Task types | ~91% bug fixes, ~9% feature requests (no explicit field) |
| Difficulty | 4 levels: Easy (<15min, 39%), Medium (15min-1hr, 52%), Hard (1-4hr, 8%), Very Hard (>4hr, 1%) |
| Data format | HuggingFace: `instance_id`, `problem_statement`, `repo`, `base_commit`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `difficulty` |
| Evaluation | Docker-based: apply patch → run tests → check FAIL_TO_PASS / PASS_TO_PASS |
| HuggingFace | [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) |
| GitHub | [`SWE-bench/SWE-bench`](https://github.com/SWE-bench/SWE-bench) |
| Install | `pip install swebench` |

**Limitation**: No explicit `task_category` — would need LLM classification or manual annotation of a subset to separate bugs from features. Primarily useful for measuring overall resolution rate improvement from contingent planning, not per-category strategy analysis.

### 3.3 DevBench (SDLC Planning)

**Why**: The only benchmark that explicitly evaluates the **design phase** before implementation. The PRD → UML → Architecture → Implementation pipeline directly maps to AtomicGuard's workflow planning model.

| Property | Value |
|---|---|
| Paper | [arXiv:2403.08604](https://arxiv.org/abs/2403.08604) (COLING 2025) |
| Instances | 22 repositories, 5 tasks each (110 total task instances) |
| Languages | Python (10), Java (5), C/C++ (5), JavaScript (2) |
| Task types | Full SDLC: Software Design, Environment Setup, Implementation, Acceptance Testing, Unit Testing |
| Data format | GitHub-hosted per-repo directories with `docs/PRD.md`, `docs/UML_class.md`, `docs/UML_sequence.md`, `docs/architecture_design.md`, `repo_config.json` |
| Evaluation | Test execution (PyTest, JUnit, GTest, Jest) + LLM-as-Judge for design phase |
| GitHub | [`open-compass/DevBench`](https://github.com/open-compass/DevBench) |

**Key distinction**: DevBench evaluates building an entire codebase from a PRD, not fixing issues in existing code. The Software Design task (PRD → UML + architecture) is directly analogous to our planner generating a workflow plan from a specification.

---

## 4. Evaluation Architecture

### 4.1 Three-Layer Evaluation

The framework measures three distinct capabilities, building from the paper's experiments outward:

```
Layer 1: Plan Quality (G_plan guards)
    "Does the planner produce structurally and semantically valid plans?"
    Metric: epsilon (pass rate per guard level)
    Already implemented: g_plan_benchmark epsilon command

Layer 2: Strategy Appropriateness (new)
    "Does the planner choose the right strategy for the problem type?"
    Metric: strategy-category alignment score
    Requires: problem catalog with task_category labels

Layer 3: End-to-End Resolution (new)
    "Does planning improve actual code generation outcomes?"
    Metric: % resolved (benchmark-native), delta vs. no-planning baseline
    Requires: integration with benchmark evaluation harnesses
```

### 4.2 End-to-End Pipeline

```
                          ┌─────────────────────────────┐
                          │  Benchmark Dataset           │
                          │  (SWE-PolyBench / SWE-bench  │
                          │   Verified / DevBench)        │
                          └──────────┬──────────────────┘
                                     │
                                     │ problem_statement
                                     ▼
                          ┌─────────────────────────────┐
                          │  Plan Generation             │
          ┌──────────────▶│  LLMPlanGenerator            │
          │               │  (HuggingFace backend)       │
          │               └──────────┬──────────────────┘
          │                          │
          │                          │ plan (JSON)
          │                          ▼
          │               ┌─────────────────────────────┐
          │               │  Plan Validation             │
          │  retry        │  G_plan Guards               │
          │  (if budget   │  Minimal → Medium → Expansive│
          │   remains)    └──────────┬──────────────────┘
          │                          │
          │               ┌──────────┴──────────────┐
          │               │                         │
          │          PASS │                    FAIL │
          │               ▼                         │
          │    ┌──────────────────┐                 │
          │    │  Plan Execution   │                 │
          │    │  (coding agent    │                 │
          │    │   follows plan)   │                 │
          │    └────────┬─────────┘                 │
          │             │                           │
          │             │ model_patch               │
          │             ▼                           │
          │    ┌──────────────────┐                 │
          │    │  Benchmark       │                 │
          │    │  Evaluation      │                 │
          │    │  (Docker + tests)│                 │
          │    └────────┬─────────┘                 │
          │             │                           │
          │             ▼                           │
          │    resolved / not_resolved              │
          │                                         │
          └─────────────────────────────────────────┘
```

### 4.3 Experimental Conditions

For each benchmark, evaluate three conditions:

| Condition | Description | Measures |
|---|---|---|
| **A: No planning** (baseline) | Coding agent receives `problem_statement` directly, no plan | Baseline resolution rate |
| **B: Static plan** | Fixed workflow template applied to all problems (current AtomicGuard SDLC workflow) | Value of having *any* plan |
| **C: Contingent plan** | LLM planner generates problem-specific plan, validated by G_plan, then executed | Value of *adaptive* planning |

The primary claim is: **C > B > A** for resolution rate, with **C >> B** for diverse problem types (where a static plan is a poor fit for some categories).

---

## 5. Per-Benchmark Integration

### 5.1 SWE-PolyBench Integration

**Dataset loading**:
```python
from datasets import load_dataset
ds = load_dataset("AmazonScience/SWE-PolyBench_500")
```

**Problem → Plan input mapping**:
```python
for instance in ds["test"]:
    specification = instance["problem_statement"]
    task_category = instance["task_category"]  # "Bug Fix" | "Feature" | "Refactoring"
    language = instance["language"]
    # Feed specification to LLMPlanGenerator → get plan → validate with G_plan
```

**Strategy vocabulary** (what the planner should learn to select):

| Problem Category | Expected Strategy | Key Steps |
|---|---|---|
| Bug Fix | Locate → Characterize → Fix → Verify | search for error, write characterization test, fix, run regression |
| Feature | Design → TDD → Implement → Validate | design interface, write tests first, implement, acceptance tests |
| Refactoring | Characterize → Extract → Verify | ensure coverage, extract/restructure, verify no regression |

**Evaluation**: Feed model_patch to SWE-PolyBench Docker harness. Output predictions as JSONL:
```json
{"instance_id": "repo__name-123", "model_patch": "diff..."}
```

**Metrics**:
- Overall % resolved (conditions A/B/C)
- % resolved per `task_category` (Bug Fix / Feature / Refactoring)
- Strategy alignment: does the generated plan match the expected strategy for the category?
- Epsilon per category: plan pass rate at each guard level
- Wilson CIs on all rates

### 5.2 SWE-bench Verified Integration

**Dataset loading**:
```python
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Verified")
```

**Problem → Plan input mapping**:
```python
for instance in ds["test"]:
    specification = instance["problem_statement"]
    difficulty = instance["difficulty"]  # for stratified analysis
    hints = instance["hints_text"]  # optional additional context
```

**Evaluation**: Use the official `swebench` harness:
```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path predictions.json \
    --max_workers 8 \
    --run_id contingent_planning
```

**Metrics**:
- Overall % resolved (conditions A/B/C)
- % resolved by `difficulty` (Easy / Medium / Hard)
- Comparison with published leaderboard results
- Since there's no task_type field: use LLM classification on `problem_statement` to tag instances as bug/feature, then report per-type resolution rates

**Practical note**: SWE-bench Verified is Python-only and mostly bug fixes. The main value is comparability — if contingent planning improves even the bug-fix-dominated SWE-bench Verified, that demonstrates value; the per-category analysis comes from SWE-PolyBench.

### 5.3 DevBench Integration

DevBench requires a different integration pattern because it evaluates building from scratch, not patching.

**Problem → Plan input mapping**:
```python
# Each repo has a PRD that serves as the specification
prd_path = f"benchmark_data/{language}/{repo}/docs/PRD.md"
specification = open(prd_path).read()
# Feed PRD to LLMPlanGenerator → get workflow plan
```

**Two evaluation modes**:

**Mode 1 — Plan quality only** (Software Design task):
- Generate a plan from the PRD
- Compare the generated plan's structure against the gold `architecture_design.md`
- Use G_plan guards to validate structural/semantic correctness
- Use LLM-as-Judge (DevBench's native evaluation) to compare plan quality

**Mode 2 — End-to-end** (Plan → Implementation → Tests):
- Generate a plan from the PRD
- Validate with G_plan
- Execute the plan: each step maps to a generator call producing code
- Evaluate with DevBench's native test harness (PyTest/JUnit/GTest/Jest)

**Metrics**:
- Plan design quality (LLM-as-Judge pairwise comparison vs. baseline)
- Implementation pass rate (acceptance tests + unit tests)
- Coverage (statement coverage from unit tests)
- Per-language breakdown

**Mapping DevBench to AtomicGuard**:

| DevBench Phase | AtomicGuard Concept |
|---|---|
| PRD | Context.specification |
| Software Design (UML + architecture) | LLMPlanGenerator output (the plan) |
| Environment Setup | Action pair: ConfigExtractorGenerator + config guard |
| Implementation | Action pair: CoderGenerator + test guards |
| Acceptance/Unit Testing | Action pair: BDDGenerator/ADDGenerator + test guards |

---

## 6. Strategy Evaluation (Layer 2)

Beyond structural plan validity (G_plan) and end-to-end resolution (benchmark harness), we need to evaluate whether the planner **chose the right approach** for the problem.

### 6.1 Strategy Classification

Define a strategy taxonomy derived from the paper's guard catalog and real-world practice:

| Strategy ID | Name | Characteristic Steps | Suited For |
|---|---|---|---|
| `S1` | Locate-and-Fix | search → characterization test → patch → verify regression | Bug fixes |
| `S2` | TDD | design tests → implement → validate | New features |
| `S3` | Characterize-and-Refactor | ensure coverage → extract/restructure → verify no regression | Refactoring |
| `S4` | Design-First | architecture → interface tests → implement → integrate | Large features |
| `S5` | Performance | profile → benchmark test → optimize → verify | Optimization |

### 6.2 Strategy Alignment Score

For SWE-PolyBench (which has ground-truth `task_category`):

```
strategy_alignment = (plans where strategy matches category) / (total plans)
```

Mapping:
- `task_category = "Bug Fix"` → expected `S1`
- `task_category = "Feature"` → expected `S2` or `S4`
- `task_category = "Refactoring"` → expected `S3`

The strategy is inferred from the plan's steps (which generators and guards it uses, in what order). This requires a classifier that reads a plan and maps it to a strategy ID.

### 6.3 Strategy-Conditioned Resolution Rate

The key metric:

```
Δ_strategy = resolve_rate(contingent) - resolve_rate(static)
```

Hypothesis: Δ_strategy is largest for problem categories where the static plan is a poor fit (e.g., a TDD-style static plan applied to a bug fix, where locate-and-fix would be better).

---

## 7. Mapping Paper Experiments to Benchmarks

Each paper experiment scales naturally to the external benchmarks:

### Experiment 1: G_plan Defect Detection → Plan Quality on Real Problems

| Paper (current) | Benchmark extension |
|---|---|
| 8 synthetic defect types | Real `problem_statement` inputs from SWE-PolyBench/SWE-bench |
| 100 trials per defect | 500 instances per benchmark |
| Detection rate per guard level | Epsilon per guard level, per task_category |

### Experiment 2: Complexity Analysis → Plan Complexity vs Problem Difficulty

| Paper (current) | Benchmark extension |
|---|---|
| Vary K (plan size 5-100) | Plans generated for real problems have natural size variation |
| Timing measurements | Validation time vs. plan size for real plans |
| MAX_EXPLORATIONS cliff | Observe whether real plans hit the cliff |

### Experiment 3: TDD Workflow → End-to-End Resolution

| Paper (current) | Benchmark extension |
|---|---|
| 3 diagnostic probes (LRU Cache, Template, Password) | 500+ real-world instances |
| Fixed TDD workflow | Contingent plans adapted to problem type |
| Yi-Coder 9B, n=50 | Multiple models via HuggingFace Inference Providers |
| +66pp improvement | Δ_resolution = contingent - no_planning |

### Experiment 4: Epsilon Estimation → Per-Category Epsilon

| Paper (current) | Benchmark extension |
|---|---|
| Static specification | Real problem_statement from each benchmark instance |
| Single epsilon-hat | Epsilon per task_category (Bug Fix / Feature / Refactor) |
| Wilson CI | Same statistical methodology |

---

## 8. Implementation Phases

### Phase 1: Problem Adapter Layer

Build adapters that load benchmark datasets and produce AtomicGuard `Context` objects:

```
adapters/
├── swe_polybench.py    # Load PB500 → Context
├── swe_bench.py        # Load SWE-bench Verified → Context
└── devbench.py         # Load DevBench PRDs → Context
```

Each adapter:
- Loads the dataset (HuggingFace or GitHub)
- Extracts `problem_statement` → `Context.specification`
- Preserves metadata (`task_category`, `difficulty`, `language`, `instance_id`)
- Returns an iterator of `(instance_id, context, metadata)` tuples

### Phase 2: Dynamic Prompt with Strategy Vocabulary

Update `prompts.json` `g_plan_llm` entry to:
- Instruct the planner to analyze the problem type before generating steps
- Provide the strategy vocabulary (S1-S5) as options
- Include language-specific constraints when applicable

### Phase 3: Plan Execution Bridge

Connect a validated plan to a coding agent that executes it:
- Each plan step maps to a generator call (the existing AtomicGuard execution model)
- The coding agent produces a patch (unified diff)
- The patch is formatted for the benchmark's prediction format

### Phase 4: Benchmark Harness Integration

Wire the prediction output to each benchmark's evaluation:
- SWE-PolyBench: JSONL predictions → Docker harness → resolved/not-resolved
- SWE-bench Verified: JSON predictions → `swebench` harness → resolved/not-resolved
- DevBench: generated code → test harness → pass rate + coverage

### Phase 5: Analysis and Reporting

Aggregate results across conditions (A/B/C) and categories:
- Resolution rates with Wilson CIs
- Per-category strategy alignment
- Epsilon per guard level per category
- Comparison tables against published leaderboard results

---

## 9. Resource Requirements

| Benchmark | Instances | Docker Storage | Eval Time (est.) | Compute |
|---|---|---|---|---|
| SWE-PolyBench PB500 | 500 | ~1.2TB (with --delete-image: ~50GB) | ~8h @ 8 threads | x86_64, 16GB+ RAM |
| SWE-bench Verified | 500 | ~100GB (env images) | ~4h @ 8 threads | x86_64, 16GB+ RAM |
| DevBench | 110 (22 repos × 5 tasks) | Minimal (native test execution) | ~2h | Any |
| LLM inference (planning) | 500-1000 calls | N/A | Depends on model/provider | HuggingFace Inference |

Total: approximately one day of compute for a full evaluation run across all three benchmarks.

---

## 10. Expected Outputs

### 10.1 Primary Results Table

| Benchmark | Condition A (no plan) | Condition B (static plan) | Condition C (contingent plan) | Δ (C-A) |
|---|---|---|---|---|
| SWE-PolyBench PB500 | X% | Y% | Z% | Z-X pp |
| — Bug Fix | | | | |
| — Feature | | | | |
| — Refactoring | | | | |
| SWE-bench Verified | X% | Y% | Z% | Z-X pp |
| — Easy | | | | |
| — Medium | | | | |
| — Hard | | | | |
| DevBench (impl pass rate) | X% | Y% | Z% | Z-X pp |

### 10.2 Plan Quality Table (extending paper Experiment 1)

| Guard Level | SWE-PolyBench ε̂ | SWE-bench ε̂ | DevBench ε̂ |
|---|---|---|---|
| Minimal | | | |
| Medium | | | |
| Expansive | | | |

### 10.3 Strategy Alignment (SWE-PolyBench only)

| Task Category | Strategy S1 | Strategy S2 | Strategy S3 | Strategy S4 | Alignment |
|---|---|---|---|---|---|
| Bug Fix | X% | Y% | Z% | W% | X% |
| Feature | X% | Y% | Z% | W% | Y+W% |
| Refactoring | X% | Y% | Z% | W% | Z% |

---

## 11. See Also

- [huggingface_finetuning_pipeline.md](huggingface_finetuning_pipeline.md) — Fine-tuning pipeline for training on benchmark traces
- [04_learning_loop.md](../extensions/04_learning_loop.md) — Formal learning definitions
- [06_generated_workflows.md](../extensions/06_generated_workflows.md) — Generated workflows (the plans G_plan validates)
- `examples/advanced/g_plan_benchmark/` — Current G_plan benchmark implementation
- `benchmarks/` — Paper's diagnostic probe and TDD workflow benchmarks

### External References

- [SWE-PolyBench (Amazon)](https://github.com/amazon-science/SWE-PolyBench) — [arXiv:2504.08703](https://arxiv.org/abs/2504.08703)
- [SWE-bench (Princeton NLP)](https://github.com/SWE-bench/SWE-bench) — [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- [DevBench (OpenCompass)](https://github.com/open-compass/DevBench) — [arXiv:2403.08604](https://arxiv.org/abs/2403.08604)
