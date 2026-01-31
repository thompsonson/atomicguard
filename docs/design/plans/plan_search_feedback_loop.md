# Plan Search with Feedback Loop

> **Status**: Research / Design
>
> **Depends on**: [planning_workflow_decomposition.md](planning_workflow_decomposition.md) (implemented), `evaluation/` harness (implemented)
>
> **Paper**: ISMIS 2026 — "The Chaos and the Scaffold: Contingent Planning for LLMs"
>
> **Prior work**: [IA Series: AI Search](https://matt.thompson.gr/2025/04/24/ia-series-n-ai-search.html), [IA Series: Search Algorithms](https://matt.thompson.gr/2025/04/24/ia-series-n-search-algorithms.html)

---

## 1. Motivation

The current planning pipeline is **forward-only**. Each step runs once:

```
g_analysis → g_recon → g_strategy → g_plan_full
```

If `g_plan_full` fails validation (Medium guard rejects the plan), the trial fails. The guard feedback — which explains *why* the plan failed — is discarded. The system never:

- Retries plan generation with the guard's feedback
- Revisits strategy selection after learning the chosen strategy produced an invalid plan
- Reclassifies the problem after downstream steps reveal the initial analysis was wrong

Within a single action pair, `DualStateAgent` already retries with accumulated `feedback_history` (up to `rmax` attempts). But **across pipeline steps**, there is no feedback mechanism. This leaves value on the table: a plan that fails Medium guard with "preconditions not satisfiable" is a signal that the strategy may be wrong, not just that the plan text needs reformatting.

This is fundamentally a **search problem**. The IA Series on [AI Search](https://matt.thompson.gr/2025/04/24/ia-series-n-ai-search.html) and [Search Algorithms](https://matt.thompson.gr/2025/04/24/ia-series-n-search-algorithms.html) established the framework for reasoning about AI problems as search through state spaces. The planning pipeline maps directly onto this framework: the state space is the set of possible pipeline outputs, the operators are generation and backtracking, and the guard feedback provides the heuristic signal that makes the search informed rather than blind.

---

## 2. Framing as Classical AI Search

The decomposed planning pipeline is an instance of a classical AI search problem. This section maps the pipeline onto the standard search formalism established in the [IA Series on AI Search](https://matt.thompson.gr/2025/04/24/ia-series-n-ai-search.html).

### 2.1 The Five Components

Every search problem is defined by five components. For plan generation:

| Search component | Pipeline mapping |
|---|---|
| **State space** | The set of all possible tuples `(A, R, S, P)` where each element is either `None` or a generated artifact |
| **Initial state** | `(None, None, None, None)` — no pipeline step has run |
| **Goal test** | All four guards pass: `AnalysisGuard(A) ∧ ReconGuard(R) ∧ StrategyGuard(S) ∧ MediumPlanGuard(P)` |
| **Operators** | `generate(step)`, `retry(step, feedback)`, `backtrack(target, failure_summary)` |
| **Path cost** | Number of LLM calls consumed (each operator application = 1 call) |

The **state space** has structure: it is a layered DAG where level *i* depends on all levels *j < i*. This is not a flat state space — the dependency ordering constrains which operators apply at each point.

### 2.2 Informed vs. Uninformed Search

The [Search Algorithms](https://matt.thompson.gr/2025/04/24/ia-series-n-search-algorithms.html) post distinguishes **uninformed** search (blind exploration using only the problem structure) from **informed** search (using a heuristic to guide expansion).

In this pipeline:

- **Uninformed** = Retry at the same level on failure, backtrack one level when retries are exhausted. The system knows the tree structure but not *why* something failed.
- **Informed** = Guard feedback provides a heuristic signal. A `MediumPlanGuard` failure saying "preconditions not satisfiable" tells you the strategy is likely wrong (backtrack depth 1), while "not parseable as JSON" says the plan just needs reformatting (depth 0). The heuristic function `h(feedback) → backtrack_depth` converts guard output into search guidance.

This is the key insight: **guards are not just validators — they are heuristic functions for the planning search**. Each guard failure carries signal about *where in the pipeline* the root cause lies, making the search informed.

### 2.3 Search Algorithm Selection

From the classical algorithm taxonomy:

| Algorithm | Applicable? | Notes |
|---|---|---|
| **Depth-first search** | Yes (default) | Natural fit: exhaust cheap options (plan retry) before expensive ones (analysis redo). Memory-efficient — only tracks current path. |
| **Breadth-first search** | Possible but expensive | Would generate all analyses before any recon. Wasteful: most first analyses are good enough. |
| **Iterative deepening** | Yes (future) | Start with small budgets, increase on failure. Gets DFS memory efficiency with BFS completeness guarantees. |
| **A\* / best-first** | Yes (with guard scores) | If guards returned a numeric quality score (not just pass/fail), A\* could prioritize the most promising partial paths. Currently guards are binary, so A\* degenerates to DFS. |
| **Beam search** | Yes (research variant) | Maintain top-K candidates at each level. Useful for comparing strategies in parallel. K× more expensive than DFS. |
| **Hill climbing** | Effectively what within-step retry does | Retry with feedback is local search — improve the current artifact without backtracking. |

**DFS is chosen as the default** because of the cost gradient: deeper nodes are cheaper to regenerate (plan retries reuse all prior context), so exhausting depth-first minimizes expected LLM calls.

### 2.4 The Two Levels of Search

The paper title is "The Chaos and the Scaffold: Contingent Planning for LLMs." This design introduces search at two levels:

1. **Object level** (existing): The generated plan itself is a contingent plan — a DAG with branches for guard success/failure. Each step has a retry budget. The `ExpansivePlanGuard` validates that this object-level search space (the plan's execution traces) reaches the goal.

2. **Meta level** (this design): The *planning process* is a search — a search for a valid plan. Guard feedback guides this meta-search. The search tree is over *pipeline outputs*, not over plan execution traces.

The scaffold (guards + pipeline structure) constrains the chaos (LLM generation) at both levels. The meta-level search is itself a contingent plan:

- "If plan generation fails with structural errors, retry plan"
- "If plan generation fails with convergence errors, revise strategy"
- "If strategy revision also fails, reconsider reconnaissance"
- "If nothing works within budget, report unsolvable"

This is a **plan for planning** — and it could itself be represented as a DAG validated by guards, closing the recursive loop.

---

## 3. The Planning Space as a Search Tree

Each pipeline execution traces a path through a search tree. The tree has four levels corresponding to the four pipeline steps:

```
                          ∅  (root)
                          │
                     g_analysis
                    ╱    │    ╲
                  A₁     A₂    A₃  ← analysis candidates
                  │
              g_recon
             ╱       ╲
           R₁         R₂  ← recon candidates
           │
       g_strategy
       ╱       ╲
     S₁         S₂  ← strategy candidates
     │            │
  g_plan       g_plan
  ╱    ╲          │
P₁     P₂       P₃  ← plan candidates
✗      ✗         ✓   ← guard validation
```

**Nodes** are generated artifacts (an analysis, a recon report, a strategy, a plan).
**Edges** are either:
- **Retry** (sibling): regenerate at the same level with guard feedback
- **Forward** (child): proceed to the next pipeline step
- **Backtrack** (up): return to an ancestor level with downstream failure information

A successful evaluation finds a root-to-leaf path where every node passes its guard. The current pipeline performs a single root-to-leaf traversal with no branching.

### 3.1 State Representation

The formal state representation from Section 2.1 applies here concretely. Each node in the tree represents a partial state:

- Level 0 (root): `(None, None, None, None)`
- Level 1 (after analysis): `(A_i, None, None, None)`
- Level 2 (after recon): `(A_i, R_j, None, None)`
- Level 3 (after strategy): `(A_i, R_j, S_k, None)`
- Level 4 (leaf): `(A_i, R_j, S_k, P_l)` — apply goal test

### 3.2 Operators

| Operator | Precondition | Effect |
|---|---|---|
| `generate(step)` | All prior steps satisfied | Produces candidate artifact at `step` |
| `retry(step, feedback)` | Step attempted, guard failed, budget remaining | Produces new candidate with feedback in context |
| `backtrack(target_step, failure_summary)` | A descendant of `target_step` failed | Clears all steps below `target_step`, amends `target_step`'s context with failure summary |

### 3.3 Why DFS

As established in Section 2.3, DFS is chosen because of the cost gradient:

1. **Cost gradient**: Retrying the deepest level (plan generation) is cheapest — it reuses all accumulated context from analysis/recon/strategy. Backtracking to strategy requires regenerating strategy + plan. Backtracking to analysis requires regenerating everything.
2. **Most failures are local**: A structurally invalid plan (bad JSON, missing fields) doesn't indicate a wrong strategy. Retrying at the plan level with guard feedback is the correct response.
3. **Memory efficiency**: DFS only needs the current path, not the full frontier.

This is the within-step analogue of hill climbing: retry with feedback is local search at a single tree level. DFS with backtracking extends this to the full tree when local search fails.

---

## 4. Feedback as Heuristic

Not all guard failures carry the same signal. The key design decision is: **given a failure at step N, how far should we backtrack?**

### 4.1 Failure Classification

Guard feedback can be classified by what it implies about the *cause* of failure:

| Guard | Feedback pattern | Implied cause | Backtrack to |
|---|---|---|---|
| MinimalPlanGuard | "not parseable as JSON" | Formatting error | Retry g_plan (depth 0) |
| MinimalPlanGuard | "retry_budget <= 0" | Invalid step structure | Retry g_plan (depth 0) |
| MediumPlanGuard | "preconditions {X} not satisfiable" | Missing context or wrong step ordering | g_strategy (depth 1) |
| MediumPlanGuard | "goal tokens unreachable" | Plan doesn't solve the problem | g_strategy (depth 1) |
| MediumPlanGuard | "total_retry_budget exceeds R_max" | Strategy overcommitted | g_strategy (depth 1) |
| ExpansivePlanGuard | "dead-end: no path reaches goal" | Fundamental structural problem | g_recon (depth 2) |
| ExpansivePlanGuard | "MAX_EXPLORATIONS exceeded" | Plan too complex to verify | g_strategy (depth 1) |
| Repeated identical failures | Same feedback across rmax retries | Analysis misclassification | g_analysis (depth 3) |

### 4.2 Heuristic Function

```
h: (step_id, GuardResult, retry_count, history) → backtrack_depth
```

The simplest implementation is rule-based:

```python
def backtrack_heuristic(step_id, guard_result, retry_count, history):
    """Determine how far to backtrack after a failure.

    Returns 0 for "retry same step", 1 for "backtrack one level", etc.
    Returns -1 for "give up" (budget exhausted, no useful backtrack).
    """
    feedback = guard_result.feedback.lower()

    # Structural formatting errors → retry at same level
    if "not parseable" in feedback or "missing required field" in feedback:
        return 0

    # Repeated identical failures → deeper backtrack
    if _same_feedback_repeated(history, guard_result, threshold=2):
        return min(retry_count, 3)  # escalate backtrack depth

    # Convergence failures → strategy was wrong
    if "not satisfiable" in feedback or "unreachable" in feedback:
        return 1  # backtrack to strategy

    # Complexity explosion → simplify strategy or improve recon
    if "MAX_EXPLORATIONS" in feedback:
        return 1

    # Dead-end exploration → recon may have missed something
    if "dead-end" in feedback and "no path" in feedback:
        return 2  # backtrack to recon

    # Default: retry at same level
    return 0
```

A future version could use an LLM call to classify the failure, but that consumes search budget and should be reserved for high-stakes backtracking decisions (e.g., "should I redo analysis?").

---

## 5. Budget Model

### 5.1 Hierarchical Budget

Each step has two budgets:

- **`rmax`**: maximum retries at this level (existing concept)
- **`backtrack_budget`**: maximum times this step can be revisited from a descendant's failure

```python
@dataclass(frozen=True)
class StepBudget:
    rmax: int = 3              # retries within this step
    backtrack_budget: int = 2  # times this step can be revisited
```

The total search budget is bounded by:

```
B_max = Σ_i (rmax_i × (1 + backtrack_budget_i))
```

But in practice, feedback-guided search terminates much earlier.

### 5.2 Example Budget Allocation

```
g_analysis:  rmax=2, backtrack_budget=1  → max 4 analysis calls
g_recon:     rmax=2, backtrack_budget=2  → max 6 recon calls
g_strategy:  rmax=2, backtrack_budget=3  → max 8 strategy calls
g_plan:      rmax=3, backtrack_budget=0  → max 3 plan calls per strategy
```

**Worst case**: 4 + 6 + 8 + (3 × 8) = 42 LLM calls.
**Common case**: Analysis passes first try, recon passes, strategy needs one revision after plan feedback, plan passes on retry = 5 LLM calls.

### 5.3 Termination Conditions

The search terminates when:

1. **Success**: A root-to-leaf path is found where all guards pass
2. **Budget exhausted**: Total LLM calls exceed `B_max`
3. **All branches pruned**: Every reachable node has been tried and failed
4. **Backtrack ceiling**: Analysis backtrack budget exhausted (nowhere left to go)

Condition 3 is the "paths that don't find a solution" meta-problem. The system reports this differently from budget exhaustion — it means the problem may be unsolvable by this pipeline, not that more budget would help.

---

## 6. Context Amendment on Backtrack

The feedback loop closes through `Context.amend()`. When backtracking from step N to ancestor step K, the ancestor's context is amended with a **failure summary** describing what went wrong downstream:

```python
def _build_backtrack_context(
    ancestor_context: Context,
    failed_step: str,
    failed_guard_result: GuardResult,
    path_history: list[tuple[str, str]],  # (step_id, artifact_content)
) -> Context:
    """Amend ancestor context with downstream failure information."""
    summary_parts = [
        f"## Previous Attempt Failed",
        f"",
        f"A downstream step ({failed_step}) failed validation:",
        f"",
        f"**Guard feedback**: {failed_guard_result.feedback}",
        f"",
        f"The following path was attempted:",
    ]
    for step_id, content_preview in path_history:
        preview = content_preview[:200] + "..." if len(content_preview) > 200 else content_preview
        summary_parts.append(f"- {step_id}: {preview}")

    summary_parts.append("")
    summary_parts.append("Generate a different output that avoids this failure mode.")

    return ancestor_context.amend(
        delta_constraints="\n".join(summary_parts)
    )
```

This reuses the existing `Context.amend()` mechanism. The LLM sees the failure in its constraints section and can adjust its output accordingly.

### 6.1 Feedback Wrapper Integration

The existing `PromptTemplate.feedback_wrapper` already handles within-step retry feedback. For cross-step backtrack feedback, the failure information arrives via `delta_constraints` rather than `feedback_history`. This keeps the two feedback channels distinct:

- **`feedback_history`**: "Your previous output for *this step* was rejected because..."
- **`delta_constraints`**: "A later step failed because the output of *this step* led to..."

Both appear in the rendered prompt but serve different purposes.

---

## 7. Architecture

### 7.1 Component: PlanSearchOrchestrator

```python
@dataclass(frozen=True)
class SearchConfig:
    """Configuration for feedback-guided plan search."""
    step_budgets: dict[str, StepBudget]  # per-step rmax + backtrack
    max_total_calls: int = 30            # hard ceiling on LLM calls
    target_guard_level: str = "medium"   # which guard level = success

@dataclass
class SearchNode:
    """A node in the search tree."""
    step_id: str
    attempt: int
    context: Context
    artifact: Artifact | None = None
    guard_result: GuardResult | None = None
    children: list["SearchNode"] = field(default_factory=list)
    parent: "SearchNode | None" = None

@dataclass
class SearchResult:
    """Outcome of plan search."""
    succeeded: bool
    path: list[SearchNode]          # root-to-leaf winning path, or best failed path
    total_calls: int                # LLM calls consumed
    tree: SearchNode                # full search tree (for analysis)
    termination_reason: str         # "success" | "budget_exhausted" | "all_pruned"
```

### 7.2 Search Algorithm (Pseudocode)

```
function PLAN_SEARCH(specification, config):
    root ← new SearchNode(step="root", context=make_context(specification))
    calls ← 0

    function DFS(node, step_index):
        if step_index >= len(STEPS):
            return VALIDATE_PLAN(node)  # leaf: run g_plan guards

        step ← STEPS[step_index]
        budget ← config.step_budgets[step]
        backtracks_remaining ← budget.backtrack_budget

        for attempt in 1..budget.rmax:
            if calls >= config.max_total_calls:
                return BUDGET_EXHAUSTED

            calls += 1
            artifact, guard_result ← EXECUTE_STEP(step, node.context)
            child ← new SearchNode(step, attempt, node.context, artifact, guard_result)
            node.children.append(child)

            if guard_result.passed:
                child.context ← node.context.amend(delta=artifact.content)
                result ← DFS(child, step_index + 1)

                if result == SUCCESS:
                    return SUCCESS

                if result == BUDGET_EXHAUSTED:
                    return BUDGET_EXHAUSTED

                # Descendant failed — decide whether to backtrack here
                if backtracks_remaining > 0:
                    backtracks_remaining -= 1
                    depth ← HEURISTIC(child.failed_step, child.guard_result, attempt)
                    if depth == 0:
                        continue  # retry this step
                    elif depth > 0:
                        return BACKTRACK(depth - 1)  # propagate up
                else:
                    return ALL_PRUNED
            else:
                # This step's guard failed — retry with feedback
                node.context ← AMEND_WITH_FEEDBACK(node.context, guard_result)

        return ALL_PRUNED  # exhausted retries at this level

    return DFS(root, 0)
```

### 7.3 Relationship to Existing Components

| Existing component | Role in search |
|---|---|
| `DualStateAgent` | Executes a single node (generate + validate + retry within rmax). Could be used as-is for each step, or the search orchestrator could manage retries directly for finer control. |
| `Workflow` | Currently forward-only. The orchestrator replaces `Workflow.execute()` for search-enabled pipelines. |
| `Context.amend()` | Creates branch contexts on backtrack. No changes needed. |
| `GuardResult.feedback` | Input to the heuristic function. No changes needed. |
| `Artifact` + DAG | Records the full search tree. Each node's artifact has `previous_attempt_id` linking retries. |
| `PromptTemplate.feedback_wrapper` | Wraps within-step retry feedback. Cross-step backtrack feedback goes through `delta_constraints`. |
| `ExperimentRunner` | Evaluation harness. Would need a new pipeline mode (e.g., `"search"`) that uses the orchestrator instead of the linear pipeline. |

---

## 8. Measurable Outcomes

### 8.1 New Metrics

| Metric | Definition | What it shows |
|---|---|---|
| `ε_search(B)` | Pass rate given search budget B | How search improves over single-shot |
| `E[calls]` | Expected LLM calls to find a valid plan | Efficiency of feedback-guided search |
| `backtrack_rate` | Fraction of trials requiring backtrack | How often the heuristic is needed |
| `backtrack_depth_dist` | Distribution of backtrack depths | Where failures typically originate |
| `ε_search(B) - ε_single` | Improvement from search over single-shot | Core paper result |

### 8.2 Publishable Results

1. **ε vs budget curve**: Plot `ε_search(B)` for B = 1, 5, 10, 20, 30. Shows diminishing returns and the budget where search saturates.
2. **Backtrack depth analysis**: Per problem category, how deep does search typically backtrack? Hypothesis: bug fixes mostly retry at plan level (depth 0), while features require strategy revision (depth 1).
3. **Heuristic value**: Compare feedback-guided backtracking vs. blind DFS (always backtrack one level). If the heuristic helps, it validates the guard feedback as a useful signal.

### 8.3 Integration with Evaluation Harness

The existing `ExperimentRunner` runs `problem × pipeline × trial`. Adding search means:

```python
# In ExperimentConfig
pipelines: list[str]  # add "search" alongside "single", "classify-then-plan", "full"

# SearchConfig embedded in ExperimentConfig for the "search" pipeline
search_config: SearchConfig | None = None
```

The scorecard gains new columns:
- `search_epsilon`: Overall pass rate with search
- `avg_calls`: Mean LLM calls per trial
- `backtrack_rate`: Fraction of trials that backtracked
- `delta_vs_full`: `ε_search - ε_full` (improvement over linear full pipeline)

---

## 9. Design Decisions

### 9.1 Chosen: DFS with feedback heuristic

**Rationale**: Minimizes LLM calls for the common case (plan retry succeeds). The heuristic prevents wasted retries when the failure clearly originates upstream. DFS naturally exhausts cheap options before expensive ones.

**Alternative considered**: Beam search (maintain top-K candidates at each level). More exploratory but K× more expensive. Appropriate for research experiments comparing strategies, not for production planning. Could be added as a search variant later.

### 9.2 Chosen: Rule-based heuristic first

**Rationale**: Guard feedback strings are structured enough for pattern matching. An LLM-based classifier would consume search budget and add latency. Start with rules; if they prove insufficient, upgrade to LLM-based classification later.

### 9.3 Chosen: Separate orchestrator, not modifying Workflow

**Rationale**: `Workflow` is the production orchestrator for executing validated plans. The search orchestrator is a *planning-time* component that finds valid plans. Mixing search into `Workflow` would complicate the execution model. Keeping them separate maintains the distinction between "finding a plan" and "executing a plan."

### 9.4 Open question: Should failed branches influence subsequent branches?

When backtracking from strategy S₁ to try S₂, should S₂'s context include information about S₁'s failure? The current design says **yes** — `_build_backtrack_context()` appends the failure summary. But this accumulates context, and after multiple backtracks the prompt may become cluttered. A **context window** that only keeps the most recent N failures may be needed.

---

## 10. Implementation Order

1. **SearchConfig + SearchNode + SearchResult** data models (pure dataclasses, no dependencies)
2. **backtrack_heuristic()** rule-based classifier
3. **PlanSearchOrchestrator.search()** core DFS loop
4. **_build_backtrack_context()** context amendment on backtrack
5. **Integration with ExperimentRunner** ("search" pipeline mode)
6. **Metrics collection** (calls, backtrack depth, backtrack rate)
7. **Tests**: unit tests for heuristic, integration tests for search with mock generators

---

## 11. See Also

**Prior work:**
- [IA Series: AI Search](https://matt.thompson.gr/2025/04/24/ia-series-n-ai-search.html) — Framing AI problems as search through state spaces
- [IA Series: Search Algorithms](https://matt.thompson.gr/2025/04/24/ia-series-n-search-algorithms.html) — Survey of search algorithms (DFS, BFS, A\*, iterative deepening, beam search)

**Design documents:**
- [planning_workflow_decomposition.md](planning_workflow_decomposition.md) — The implemented linear pipeline this extends
- [benchmark_evaluation_framework.md](benchmark_evaluation_framework.md) — Evaluation harness that would run search experiments

**Source code:**
- `src/atomicguard/application/agent.py` — `DualStateAgent` within-step retry loop (hill climbing)
- `src/atomicguard/application/workflow.py` — `Workflow` forward-only orchestrator (single path, no backtracking)
- `examples/advanced/g_plan_benchmark/evaluation/runner.py` — `ExperimentRunner`
