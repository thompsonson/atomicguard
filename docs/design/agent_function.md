# Dual-State Agent: Agent Function

## Percepts and Actions Summary

**Percepts:**

- Specification Ψ (from user)
- Artifact a (from LLM)
- Dependency artifacts (scoped from repository R)
- Guard result (v, φ, fatal) (from guard execution)

**Actions:**

| Category | Actions |
|----------|---------|
| **External (Actuators)** | QUERY-LLM, REPLY-TO-USER, ESCALATE-TO-USER |
| **Sensing** | EXECUTE-GUARD (with scoped dependencies) |
| **Internal** | ADVANCE-STATE, REFINE-CONTEXT |

---

## Percept Sequence Tables

### 1. Happy Path (Guard passes first attempt)

| Percept Sequence | Action |
|------------------|--------|
| [Ψ: "Implement LRU Cache"] | QUERY-LLM |
| [Ψ, a₁: "class LRUCache..."] | EXECUTE-GUARD(a₁, deps={}) |
| [Ψ, a₁, (⊤, ∅)] | ADVANCE-STATE |
| [Ψ, a₁, (⊤, ∅), sw: {g₁ ↦ ⊤}] | REPLY-TO-USER(a₁) |

### 2. Refinement Loop (Guard fails, then passes)

| Percept Sequence | Action |
|------------------|--------|
| [Ψ: "Implement LRU Cache"] | QUERY-LLM |
| [Ψ, a₁: "class LRUCache..."] | EXECUTE-GUARD(a₁, deps={}) |
| [Ψ, a₁, (⊥, φ₁: "Missing get method")] | REFINE-CONTEXT |
| [Ψ, a₁, (⊥, φ₁), H: {(a₁, φ₁)}] | QUERY-LLM |
| [Ψ, a₂: "class LRUCache with get..."] | EXECUTE-GUARD(a₂, deps={}) |
| [Ψ, a₂, (⊤, ∅)] | ADVANCE-STATE |
| [Ψ, a₂, (⊤, ∅), sw: {g₁ ↦ ⊤}] | REPLY-TO-USER(a₂) |

### 3. Rmax Exhaustion (All retries fail)

| Percept Sequence | Action |
|------------------|--------|
| [Ψ: "Implement prime validator"] | QUERY-LLM |
| [Ψ, a₁] | EXECUTE-GUARD(a₁, deps={}) |
| [Ψ, a₁, (⊥, φ₁: "Sum is 12, not prime")] | REFINE-CONTEXT |
| [Ψ, H: {(a₁, φ₁)}] | QUERY-LLM |
| [Ψ, a₂, (⊥, φ₂: "Sum is 14, not prime")] | REFINE-CONTEXT |
| [Ψ, H: {(a₁, φ₁), (a₂, φ₂)}] | QUERY-LLM |
| [Ψ, a₃, (⊥, φ₃: "Sum is 10, not prime")] | REFINE-CONTEXT |
| [Ψ, H: {...}, retries = Rmax] | REPLY-TO-USER(failure, provenance) |

### 3.5. Fatal Escalation (⊥_fatal - Non-recoverable)

| Percept Sequence | Action |
|------------------|--------|
| [Ψ: "Execute code in sandbox"] | QUERY-LLM |
| [Ψ, a₁: "import os; os.system(...)"] | EXECUTE-GUARD(a₁, deps={}) |
| [Ψ, a₁, (⊥_fatal, φ: "Security: os.system forbidden")] | ESCALATE-TO-USER(a₁, φ) |

**Note**: On `⊥_fatal`, the agent:

1. Stores artifact in DAG for provenance
2. Raises `EscalationRequired` immediately (no retry)
3. Workflow returns `WorkflowStatus.ESCALATION`
4. Human must review artifact and feedback

### 4. TDD Workflow (Guard uses dependency artifact)

| Percept Sequence | Action |
|------------------|--------|
| [Ψ: "Write tests for LRU Cache"] | QUERY-LLM |
| [Ψ, a_test: "def test_lru..."] | EXECUTE-GUARD(a_test, deps={}) |
| [Ψ, a_test, (⊤, ∅), sw: {g_test ↦ ⊤}] | ADVANCE-STATE |
| [Ψ: "Implement LRU Cache"] | QUERY-LLM |
| [Ψ, a_impl: "class LRUCache..."] | EXECUTE-GUARD(a_impl, deps={test: a_test}) |
| [Ψ, a_impl, (⊥, φ: "test_get failed")] | REFINE-CONTEXT |
| [...] | QUERY-LLM |
| [Ψ, a_impl₂, (⊤, ∅), sw: {g_test ↦ ⊤, g_impl ↦ ⊤}] | REPLY-TO-USER(a_impl₂) |

---

## State Representation

**Workflow State (Sworkflow):** Deterministic, observable

```
sw : G → {⊥, ⊤}
```

**Environment State (Senv):** Contains stochastic artifacts

```
senv = ⟨a, C⟩
where C = ⟨E, Clocal, Hfeedback⟩
E = ⟨R, Ω⟩ (Ambient Environment)
```

**Key Invariant (Definition 4):** Workflow state stable on guard failure

```
T(sw, ⊥) = sw   # No state change on failure
```

**Remark (Semantic Agency)**: The action `QUERY-LLM` may invoke an autonomous semantic agent (ReAct, CoT, tool-use loop) rather than a single inference step. The agent's internal reasoning trajectory is contained within `Senv` and is opaque to the workflow. The Dual-State Agent observes only the final artifact `a` and validates it via `EXECUTE-GUARD`.

---

## Agent Function Definition

```
function DUAL-STATE-AGENT(percept) returns action
    persistent: sw,        workflow state (σ : G → {⊥, ⊤})
                senv,      environment state ⟨artifact, context⟩
                retries,   current retry count
                Rmax,      maximum retries
                Ψ,         specification (immutable)
                Hfeedback, feedback history
                R,         artifact repository (DAG)

    # Initialize on specification input
    if percept contains specification:
        Ψ ← specification
        sw ← initial_state()           # all guards ⊥
        Hfeedback ← ∅
        retries ← 0
        C ← compose_context(E, Ψ, Hfeedback)
        return QUERY-LLM(C)

    # Process LLM artifact
    elif percept contains artifact:
        a ← artifact
        R.store(a)                     # append to DAG
        deps ← extract_dependencies(current_step)
        return EXECUTE-GUARD(a, deps)

    # Process guard result
    elif percept contains (v, φ, fatal):
        if v = ⊤:
            # Definition 8: Advance workflow state
            sw ← sw[gid ↦ ⊤]
            Hfeedback ← ∅
            retries ← 0

            if is_goal_state(sw):
                return REPLY-TO-USER(a)
            else:
                C ← compose_context(E, Ψ, Hfeedback)
                return QUERY-LLM(C)

        elif fatal:  # v = ⊥_fatal
            # Definition 6: Non-recoverable failure - escalate immediately
            return ESCALATE-TO-USER(a, φ)

        else:  # v = ⊥ (retryable)
            # Check retry budget
            if retries ≥ Rmax:
                return REPLY-TO-USER(failure, Hfeedback)

            # Definition 5: Refine context, workflow stable
            Hfeedback ← Hfeedback ∪ {(a, φ)}
            retries ← retries + 1
            C ← compose_context(E, Ψ, Hfeedback)
            return QUERY-LLM(C)

function compose_context(E, Ψ, Hfeedback) returns C
    # Definition 3: Hierarchical context composition
    Clocal ← ⟨Ψ, current_artifact⟩
    return ⟨E, Clocal, Hfeedback⟩

function extract_dependencies(step) returns deps
    # Extract scoped artifacts from R based on step configuration
    deps ← {}
    for dep_id in step.guard_artifact_deps:
        deps[dep_id] ← R.get_version(dep_id)
    return deps

function is_goal_state(sw) returns boolean
    # All required guards satisfied
    return ∀g ∈ Sgoal : sw(g) = ⊤
```

---

## Agent Type Selection

**Goal-Based Agent** — The agent maintains explicit goal state (Sgoal) and selects actions to achieve guard satisfaction. Not utility-based (no cost optimization beyond Rmax bound).
