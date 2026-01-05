# Domain Notation

This document defines the formal symbols and definitions used throughout the AtomicGuard framework. It serves as the foundational reference for understanding the agent function and program documentation.

> **Source**: Symbol table adapted from [Dual-State Domain](https://matt.thompson.gr/2025/12/30/dual-state-domain.html). Formal definitions from the research paper.

---

## Symbol Reference

### State Space

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Composite | S | System State | S = S_workflow × S_env (Def. 1) |
| Control | S_workflow | Workflow State | {σ \| σ: G → {⊥, ⊤}} — Deterministic guard truth assignments |
| Information | S_env | Environment State | A × C — Stochastic artifact and context history |

### Workflow Components

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Static | G | Guard Set | {g₁, …, gₙ} — Unique guard identifiers |
| Dynamic | σ | Current State | σ: G → {⊥, ⊤} — Specific truth assignment at step t |
| Function | T | Transition Function | T(s_w, ⊤) → s_{w+1}; T(s_w, ⊥) → s_w (Invariant on failure) |

### Environment: Global

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Persistent | R | Versioned Repository | Append-only DAG: R = {(a₀ … aₖ) \| aᵢ ∈ A} (Def. 2) |
| Static | Ω | Global Constraints | Invariant safety rules accessible to all steps |
| Container | E | Ambient Environment | ⟨R, Ω⟩ — Read-only access to ancestors |

### Environment: Local

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Static | Ψ | Static Specification | Requirements, tests, and constraints for the current step |
| Mutable | aₖ | Candidate Artifact | The artifact currently under validation (output of a_gen) |
| Container | C_local | Local Context | ⟨Ψ, aₖ⟩ — The active scope for the current node |
| Transient | H | Feedback History | [(aₖ, φₖ), …] — Accumulates rejections for the current step |

### Action Pair

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Tuple | A | Action Pair | ⟨ρ, a_gen, G⟩ (Def. 6) |
| Function | ρ | Precondition | ρ: S_workflow → {0, 1} — Gating function |
| Function | a_gen | Generator | a_gen: C → A — Stochastic generative function |
| Function | G | Guard | G: A × C → (v, φ) — Deterministic verification |

### Planning

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Config | R_max | Retry Limit | Finite budget per node (Budget constraint) |
| Status | v | Verdict | v ∈ {⊤, ⊥_retry, ⊥_fatal} |
| Output | φ | Diagnostic Feedback | φ ∈ Σ* — Error trace or compiler message |

---

## Formal Definitions

### Definition 1: State Space Decomposition

The system state space S is decomposed into an observable workflow space and an opaque environment space:

```
S = S_workflow × S_env
```

- **Workflow State (S_workflow)**: Set of all truth assignments to guard functions
- **Environment State (S_env)**: Cartesian product of artifact space and context space (A × C)

### Definition 2: Artifact Space & Versioning

A Versioned Repository R is defined as a Directed Acyclic Graph (DAG) where nodes represent artifact versions:

```
R = {(a₀, …, aₖ) | aᵢ ∈ A}
```

Every generative action creates a new node rather than overwriting the previous state, preserving failure history for learning.

### Definition 3: Hierarchical Context Composition

The context C conditioning the generator and available to the guard:

```
C_total = ⟨E, C_local, H_feedback⟩
```

Where:

- **E = ⟨R, Ω⟩**: Ambient Environment (repository + global constraints)
- **C_local = ⟨Ψ, aₖ⟩**: Local Context (specification + current artifact)
- **H_feedback**: Accumulated guard rejections for this node

### Definition 4: Workflow Stability

The workflow state s_w is invariant under guard failure:

```
T(s_w, ⊥) = s_w
```

Progress in S_workflow occurs exclusively upon guard satisfaction.

### Definition 5: Context Refinement

While workflow state remains stable on failure, context evolves to capture error signal:

```
C_{k+1} = ⟨Ψ, Hₖ ∪ {(aₖ, φₖ)}⟩
```

This ensures the planner remains at the same node while the generator's conditioning changes monotonically.

### Definition 6: Action Pair

An action is defined as a tuple representing the sequence of verification and execution:

```
A = ⟨ρ, a_gen, G⟩
```

Where:

- **ρ**: Precondition (Entry Gate) — determines applicability
- **a_gen**: Generator (Execution) — produces artifact from context
- **G**: Guard (Exit Gate) — validates artifact, returns verdict + feedback

### Definition 7: System Dynamics

The evolution of system state upon executing action A:

1. **Generation**: a' ~ a_gen(C_t)
2. **Sensing**: ⟨v, φ⟩ = G(a', C_t)
3. **State Update**: Based on verdict v (advance on ⊤, refine on ⊥)

### Definition 8: Workflow Transition Function

```
T(s_w, v) = {
    s_w[g_id ↦ ⊤]  if v = ⊤
    s_w              if v = ⊥
}
```

### Definition 9: Guard-Based Planning Problem

```
P = ⟨S_workflow, A, s_w0, C_init, S_goal, R_max⟩
```

Where:

- **s_w0**: Initial control state (all guards ⊥)
- **C_init**: Initial specification context
- **S_goal**: Set of satisfying goal states
- **R_max**: Maximum retry budget per node

---

## See Also

- [domain_definitions.md](domain_definitions.md) — Foundational and architectural definitions
- [domain_ubiquitous_language.md](domain_ubiquitous_language.md) — DDD terms and architectural concepts
- [agent_function.md](agent_function.md) — Agent percepts and actions
- [agent_program.md](agent_program.md) — Implementation details
