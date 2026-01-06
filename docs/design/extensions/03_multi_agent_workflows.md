# Multi-Agent Workflows

This document formally extends the Dual-State Framework to characterize multi-agent systems sharing ℛ as a common substrate for coordination, proving this requires NO modification to system dynamics.

> **Depends on**: [01_versioned_environment.md](01_versioned_environment.md) (repository items, Definitions 10-16), [02_artifact_extraction.md](02_artifact_extraction.md) (extraction, Definitions 17-18)
>
> **See also**: [00_notation_extensions.md](00_notation_extensions.md) (symbols), [../agent_design_process/domain_notation.md](../agent_design_process/domain_notation.md) (base notation)

---

## Relationship to Paper

This extension builds on Definitions 1-9 from the base paper. The paper defines artifacts `a ∈ A` as simple content. This extension uses **repository items** `r ∈ ℛ` (Definition 10) that wrap artifacts with configuration snapshots. Multiple agents share the same ℛ, enabling coordination through the shared DAG without explicit message passing.

---

## 1. Motivation

The single-agent Dual-State Framework (Definitions 1-9) provides guarantees for one agent executing one workflow. Natural extensions arise:

- **Specialization**: Different agents for different tasks (test generation vs. implementation)
- **Parallelism**: Multiple agents working on independent action pairs concurrently
- **Modularity**: Agents as composable units with defined interfaces

The key insight from the paper (Section 8.4):

> "The workflow state S_workflow serves as a **fully observable blackboard**. A downstream Implementation Agent does not need to query an upstream Specification Agent for status; it simply executes the relevant sensing action on the shared artifact."

This section formalizes how multiple agents can share ℛ without explicit message passing or consensus protocols.

---

## 2. Formal Definitions

### Definition 19: Multi-Agent System

A **Multi-Agent System** (MAS) in the Dual-State Framework is a tuple:

```
MAS = ⟨{Ag₁, ..., Agₙ}, ℛ, G⟩
```

Where:

- **{Ag₁, ..., Agₙ}** is a finite set of agents
- **ℛ** is the shared Versioned Repository containing repository items (Definition 10)
- **G** is the shared Guard Library (deterministic validation functions)

Each agent Agᵢ:

- Executes its own workflow with local state σᵢ
- Reads from and writes repository items to the shared ℛ
- Uses guards from the shared library G

### Definition 20: Agent-Local Workflow State

Each agent Agᵢ maintains a local workflow state:

```
σᵢ: G → {⊥, ⊤}
```

This is the agent's **belief** about which guards have been satisfied, derived by evaluating guards on repository items in ℛ.

### Remark (State Derivability)

Unlike distributed systems requiring explicit state synchronization, each σᵢ is **derivable** from ℛ:

```
σᵢ(g) = G_g(r_g.a, C)    where r_g = latest accepted item for guard g in ℛ
```

Agents do not exchange state directly — they compute it by querying the shared repository.

---

## 3. Theorems

### Theorem 6: Belief Convergence (Shared Truth via Guards)

**Statement**: If two agents Ag₁ and Ag₂ evaluate the same guard G on the same repository item r ∈ ℛ, they arrive at identical verdicts.

```
∀ Ag₁, Ag₂, r ∈ ℛ:  G(r.a, C) evaluated by Ag₁ = G(r.a, C) evaluated by Ag₂
```

**Proof**:

1. Guards are deterministic functions (Definition 6): G: A × C → (v, φ)
2. The repository item r is immutable once stored in ℛ (Definition 2: append-only)
3. Context C is derived from ℛ (Definition 13), which is shared
4. Therefore, identical inputs yield identical outputs

∎

**Corollary (Shared Truth)**: All agents observing the same repository item reach consensus on workflow state without explicit coordination:

```
B_Ag₁(s_w) ∩ B_Ag₂(s_w) → {G(r.a)}
```

Where B_Agᵢ(s_w) is agent i's belief about workflow state.

### Theorem 7: System Dynamics Preservation

**Statement**: The multi-agent extension does not modify Definition 7 (System Dynamics). Each agent executes standard single-agent dynamics independently.

**Proof**:

Definition 7 specifies state evolution for a single agent:

1. Generation: a' ~ a_gen(C_t)
2. Sensing: ⟨v, φ⟩ = G(a', C_t)
3. State Update: Based on verdict v

For multiple agents:

- Each Agᵢ executes steps 1-3 on its local workflow
- Writes to ℛ are append-only (no conflicts)
- Reads from ℛ observe consistent snapshots
- No new state spaces, transitions, or actuators are introduced

The multi-agent case is simply **parallel composition** of single-agent dynamics.

∎

### Theorem 8: Cross-Workflow Dependency Resolution

**Statement**: Agent Agⱼ can consume repository items produced by Agᵢ without direct communication, using extraction (Definition 17).

```
Agᵢ produces r with action_pair_id = "g_test", status = ACCEPTED
Agⱼ extracts: E(ℛ, Φ_action_pair("g_test") ∧ Φ_status(ACCEPTED)) → {r}
Agⱼ uses r.a as dependency
```

**Proof**: Follows from Theorem 3 (Extraction Invariance) — extraction is a read-only query that does not modify state.

∎

---

## 4. Concurrency Properties

### Property 1: Read-Read Commutativity

Multiple concurrent reads (extractions, guard evaluations) do not conflict:

```
E₁(ℛ, Φ₁) ∥ E₂(ℛ, Φ₂) → No conflict
G₁(r.a, C) ∥ G₂(r.a, C) → No conflict
```

**Rationale**: ℛ is append-only. Reads observe immutable data.

### Property 2: Write Serialization

Concurrent writes (repository item stores) are serialized by the repository:

```
store(r₁) ∥ store(r₂) → Both succeed, order determined by repository
```

**Rationale**: Append-only semantics. No overwrites, no conflicts. Both items are preserved with distinct IDs and timestamps.

### Property 3: Read-After-Write Visibility

A repository item written by Agᵢ becomes visible to Agⱼ after the write completes:

```
Agᵢ: store(r) at t₀
Agⱼ: E(ℛ, Φ) at t₁ > t₀ → includes r (if Φ(r) = ⊤)
```

**Rationale**: Append-only semantics guarantee monotonic visibility.

### Property 4: Monotonic Read

Once an agent observes a repository item, it cannot "disappear":

```
If r ∈ E(ℛ, Φ) at t₁, then r ∈ E(ℛ, Φ) at t₂ for all t₂ > t₁
```

**Rationale**: ℛ is append-only. Deletions are not permitted.

---

## 5. Coordination Patterns

### 5.1 Blackboard Pattern

ℛ serves as a shared blackboard where agents post repository items and observe others' contributions:

```
Agent A (Specification):
  Writes: specification artifact → stored as repository item r₁
  Guard: SpecificationGuard validates

Agent B (Test Generation):
  Reads: specification from ℛ (extracts r₁.a)
  Writes: test artifact → stored as repository item r₂
  Guard: SyntaxGuard validates

Agent C (Implementation):
  Reads: specification + tests from ℛ (extracts r₁.a, r₂.a)
  Writes: implementation artifact → stored as repository item r₃
  Guard: TestGuard validates against tests
```

No direct communication between agents — coordination emerges from shared ℛ.

### 5.2 Producer-Consumer

Agents form implicit producer-consumer relationships via repository item dependencies:

```
Producer (Agᵢ):
  action_pair_id: "g_test"
  Produces: test artifact → stored as r

Consumer (Agⱼ):
  guard_artifact_deps: {"test": "g_test"}
  Extracts: accepted repository item from ℛ
  Uses: r.a as input to implementation generation
```

### 5.3 Fork-Join (Parallel Action Pairs)

Independent action pairs can execute in parallel:

```
Fork:
  Ag₁ executes action_pair "g_unit_tests"
  Ag₂ executes action_pair "g_integration_tests"
  Ag₃ executes action_pair "g_arch_tests"
  (All run concurrently, no dependencies between them)

Join:
  Ag₄ waits for all three via extraction filter:
  E(ℛ, Φ_status(ACCEPTED) ∧ Φ_action_pair ∈ {"g_unit_tests", "g_integration_tests", "g_arch_tests"})
  Proceeds when |result| = 3
```

---

## 6. Failure Modes

### 6.1 Deadlock Freedom

**Claim**: Multi-agent workflows using shared ℛ are deadlock-free.

**Rationale**:

- No locks are held during read operations
- Writes are non-blocking (append-only)
- Agents do not wait for each other — they wait for repository items in ℛ
- R_max bounds prevent infinite waiting on any single action pair

### 6.2 Starvation

Possible if an upstream agent never produces required repository items. Mitigations:

- Timeouts on extraction waits
- ⊥_fatal escalation when dependencies unavailable
- Human intervention via HumanGuard

### 6.3 Inconsistent Views

Agents may see temporarily inconsistent views of ℛ during concurrent writes. This is acceptable because:

- Guards are deterministic — eventual consistency of verdicts is guaranteed
- Append-only semantics ensure monotonic progress
- No rollbacks or overwrites can cause confusion

---

## 7. Summary

1. **Multi-agent workflows require NO new system dynamics.** Each agent executes Definition 7 independently.

2. **Coordination is emergent.** Agents communicate through the shared DAG (ℛ), not explicit messages.

3. **Guards provide deterministic consensus.** Same repository item + same guard = same verdict (Theorem 6).

4. **Concurrency is safe.** Append-only semantics eliminate read-write conflicts.

5. **The extension is minimal.** Only Definition 19 (MAS) and Definition 20 (Agent-Local State) are added, building on repository items (Definition 10).

---

## See Also

- [01_versioned_environment.md](01_versioned_environment.md) — Repository items (Definitions 10-16)
- [02_artifact_extraction.md](02_artifact_extraction.md) — Extraction function (Definitions 17-18)
- [00_notation_extensions.md](00_notation_extensions.md) — Symbol reference for MAS, σᵢ, B_Agᵢ
- [../agent_design_process/domain_notation.md](../agent_design_process/domain_notation.md) — Base notation (Definitions 1-9)
- Paper Section 8.4 — Multi-Agent Shared Truth proposition
