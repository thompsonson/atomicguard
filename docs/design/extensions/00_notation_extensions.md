# Notation Extensions

This document defines additional symbols introduced by formal framework extensions. These extend the base notation in [domain_notation.md](../agent_design_process/domain_notation.md).

> **Note**: Extensions are designed to preserve existing system dynamics (Definitions 1-9). They introduce no new actuators or state transitions.

---

## Relationship to Paper

The base paper defines artifacts `a ∈ A` as simple content. These extensions introduce **repository items** `r ∈ ℛ` that wrap artifacts with configuration snapshots (Ψ, Ω, W_ref) and provenance metadata. This distinction is critical:

| Symbol | Scope | Description |
|--------|-------|-------------|
| `a` | Paper | Base artifact — content produced by generator |
| `r` | Extension | Repository item — artifact wrapped with configuration |

---

## Versioned Environment Extension (Definitions 10-16)

Symbols for repository items, workflow references, and human-in-the-loop. This is the foundational extension that other extensions build upon.

### Repository Item

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Tuple | r | Repository Item | r = ⟨a, Ψ, Ω, W_ref, H, source, metadata⟩ (Def. 10) |
| Space | ℛ | Repository | ℛ = {r₀, r₁, ..., rₖ} — append-only DAG of items |
| Field | r.a | Artifact Content | Base artifact a ∈ A (paper Definition 2) |
| Enum | source | Item Source | source ∈ {GENERATOR, HUMAN} — Origin of artifact (Def. 10) |

### Workflow Reference

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Hash | W_ref | Workflow Reference | W_ref = hash(W) — Content-addressed pointer (Def. 11) |
| Function | resolve | Resolution Function | resolve: W_ref → W — Implementation-defined (Def. 11) |

**Note**: W_ref is a NEW concept not present in the original paper. The paper's Planning Problem (Definition 9) treats workflow structure as implicit.

### Configuration Evolution

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Operator | ⊕ | Amendment Operator | Composition of configuration with delta (Def. 12) |
| Delta | Δ_Ψ | Specification Amendment | Change to apply to specification |
| Delta | Δ_Ω | Constraint Amendment | Change to apply to global constraints |

### Context Derivation

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Function | C(r) | Context Derivation | C(r) = ⟨E(r), C_local(r), H(r)⟩ (Def. 13) |

### Checkpoint and Resume

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Pointer | Checkpoint | Checkpoint | r_id ∈ ℛ — Pointer to resumable state (Def. 14) |
| Function | Resume | Resume Function | Resume: ℛ × r_id → Execution (Def. 15) |

### Human Generator

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Function | a_human | Human Generator | a_human: (C, Δ_Ψ, Δ_Ω) → a (Def. 16) |

**Reference**: [01_versioned_environment.md](01_versioned_environment.md)

---

## Extraction Extension (Definitions 17-18)

Symbols for artifact extraction (read-only queries over ℛ). Builds on repository items (Definition 10).

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Function | E | Extraction Function | E: ℛ × Φ → 2^ℛ — Pure query over repository (Def. 17) |
| Predicate | Φ | Filter Predicate | Φ: r → {⊤, ⊥} — Selection criteria (Def. 18) |
| Output | 2^ℛ | Powerset of Items | Subset of repository items matching filter |

**Reference**: [02_artifact_extraction.md](02_artifact_extraction.md)

---

## Multi-Agent Extension (Definitions 19-20)

Symbols for multi-agent workflows sharing ℛ as common state. Builds on repository items (Definition 10) and extraction (Definitions 17-18).

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| System | MAS | Multi-Agent System | MAS = ⟨{Ag₁, ..., Agₙ}, ℛ, G⟩ (Def. 19) |
| Set | {Ag₁, ..., Agₙ} | Agent Set | Collection of n agents sharing repository |
| State | σᵢ | Agent-Local State | σᵢ: G → {⊥, ⊤} — Agent i's workflow state (Def. 20) |
| Belief | B_Agᵢ(s_w) | Agent Belief | Agent i's belief about workflow state |

**Reference**: [03_multi_agent_workflows.md](03_multi_agent_workflows.md)

---

## Learning Loop Extension (Definitions 21-24)

Symbols for training data extraction and policy updates. Builds on repository items (Definition 10) and extraction (Definitions 17-18).

### Training Predicates and Traces

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Predicate | Φ_refinement | Refinement Predicate | Φ_refinement(r) = ⊤ iff r.status = ACCEPTED ∧ ∃r' ∈ provenance(r) : r'.status = REJECTED (Def. 21) |
| Set | τ | Training Trace | τ = E(ℛ, Φ_training) — Extracted training data (Def. 22) |
| Predicate | Φ_training | Training Filter | Policy choice: Φ_refinement, Φ_status(ACCEPTED), etc. (Def. 22) |

### Reward and Policy

| Scope | Symbol | Name | Formal Definition / Notes |
|-------|--------|------|---------------------------|
| Function | R_sparse | Sparse Reward Signal | R_sparse: r → {-1, +1} — From guard verdict (Def. 23) |
| Function | L(θ) | Policy Update Loss | L(θ) = -E_τ[log π_θ(r.a \| r.Ψ, r.H)] (Def. 24) |
| Policy | π_θ | Generator Policy | Parameterized by weights θ |

### Common Filter Predicates

| Symbol | Definition | Use Case |
|--------|------------|----------|
| Φ_refinement | retry→success chains | Learn from corrections |
| Φ_source(s) | r.source = s | Filter by GENERATOR or HUMAN |
| Φ_status(s) | r.status = s | Filter by ACCEPTED or REJECTED |
| Φ_timestamp(>t) | r.created_at > t | Recent traces only |

**Reference**: [04_learning_loop.md](04_learning_loop.md), [05_learning_implementation.md](05_learning_implementation.md)

---

## Definition Summary

| Definition | Name | Extension |
|------------|------|-----------|
| Def. 10 | Repository Item | Versioned Environment |
| Def. 11 | Workflow Reference | Versioned Environment |
| Def. 12 | Configuration Amendment | Versioned Environment |
| Def. 13 | Context Derivation | Versioned Environment |
| Def. 14 | Checkpoint | Versioned Environment |
| Def. 15 | Resume Function | Versioned Environment |
| Def. 16 | Human Generator | Versioned Environment |
| Def. 17 | Extraction Function | Artifact Extraction |
| Def. 18 | Filter Predicate | Artifact Extraction |
| Def. 19 | Multi-Agent System | Multi-Agent Workflows |
| Def. 20 | Agent-Local Workflow State | Multi-Agent Workflows |
| Def. 21 | Refinement Predicate | Learning Loop |
| Def. 22 | Training Trace | Learning Loop |
| Def. 23 | Sparse Reward Signal | Learning Loop |
| Def. 24 | Policy Update | Learning Loop |

---

## Theorem Summary

| Theorem | Name | Extension |
|---------|------|-----------|
| Thm. 3 | Extraction Invariance | Artifact Extraction |
| Thm. 4 | Resume Preserves System Dynamics | Versioned Environment |
| Thm. 5 | Human-in-the-Loop Preserves System Dynamics | Versioned Environment |
| Thm. 6 | Belief Convergence | Multi-Agent Workflows |
| Thm. 7 | System Dynamics Preservation | Multi-Agent Workflows |
| Thm. 8 | Cross-Workflow Dependency Resolution | Multi-Agent Workflows |
| Thm. 9 | Training Trace Completeness | Learning Loop |
| Thm. 10 | Learning Loop Preserves System Dynamics | Learning Loop |

Note: Theorems 1-2 are in the base paper. Extension theorems start at 3.

---

## Extension Dependency Graph

```
┌─────────────────────────────────────────┐
│  Versioned Environment (Def 10-16)      │  ← Foundation
│  - Repository Item r (wraps artifact a) │
│  - W_ref content addressing             │
│  - Checkpoint/Resume/Human              │
└─────────────────────────────────────────┘
              │
              ├──────────────────────────────┐
              ▼                              ▼
┌──────────────────────────┐    ┌────────────────────────────┐
│  Artifact Extraction     │    │  Multi-Agent Workflows     │
│  (Def 17-18)             │    │  (Def 19-20)               │
│  Queries repository      │    │  Shares repository         │
│  items r                 │    │  items r                   │
└──────────────────────────┘    └────────────────────────────┘
              │
              ▼
┌──────────────────────────┐
│  Learning Loop           │
│  (Def 21-24)             │
│  - Training traces τ     │
│  - Reward R_sparse       │
│  - Policy update L(θ)    │
└──────────────────────────┘
              │
              ▼
┌──────────────────────────┐
│  Coach (FUTURE)          │
│  - Dense reward R_dense  │
│  - Semantic feedback     │
└──────────────────────────┘
```

---

## Key Notation Distinctions

### Artifact vs Repository Item

```
Paper:     a ∈ A                    # Simple artifact (content only)
Extension: r = ⟨a, Ψ, Ω, W_ref, H, source, metadata⟩  # Wrapped with context
```

### History is Emergent

The Markov property means each repository item contains all state needed for the next step. History is implicit in the ℛ DAG structure:

```
ℛ = {r₀, r₁, ..., rₖ}    # History is the sequence of items
```

No separate `Ψ_history`, `Ω_history`, or `W_history` concepts are needed.

---

## See Also

- [../agent_design_process/domain_notation.md](../agent_design_process/domain_notation.md) — Base notation (Definitions 1-9)
- [01_versioned_environment.md](01_versioned_environment.md) — Versioned environment formal extension
- [02_artifact_extraction.md](02_artifact_extraction.md) — Extraction formal extension
- [03_multi_agent_workflows.md](03_multi_agent_workflows.md) — Multi-agent formal extension
- [04_learning_loop.md](04_learning_loop.md) — Learning loop formal extension
- [05_learning_implementation.md](05_learning_implementation.md) — Learning loop implementation guide
