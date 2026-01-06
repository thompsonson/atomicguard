# Artifact Extraction

This document formally extends the Dual-State Framework to characterize artifact extraction as a read-only query operation on the Versioned Repository (ℛ), proving it preserves system dynamics without requiring a new actuator.

> **Depends on**: [01_versioned_environment.md](01_versioned_environment.md) (repository items, Definitions 10-16)
>
> **See also**: [00_notation_extensions.md](00_notation_extensions.md) (symbols), [../agent_design_process/domain_notation.md](../agent_design_process/domain_notation.md) (base notation), [../agent_design_process/agent_program.md](../agent_design_process/agent_program.md) (implementation)

---

## Relationship to Paper

This extension builds on Definitions 1-9 from the base paper. The paper defines artifacts `a ∈ A` as simple content. This extension introduces **repository items** `r ∈ ℛ` (Definition 10) that wrap artifacts with configuration snapshots. The extraction function queries repository items, not base artifacts.

---

## 1. Motivation

The Dual-State Framework defines three core operations that modify system state:

1. **Generation** (`a_gen: C → A`) — Produces artifacts, stored as items in ℛ
2. **Sensing** (`G: A × C → (v, φ)`) — Projects onto S_workflow via guard evaluation
3. **State Transition** (`T(s_w, v)`) — Advances workflow state on guard satisfaction

A natural question arises: **How do we retrieve repository items from ℛ for external use?**

Use cases include:

- Cross-workflow item sharing (Agent A's output becomes Agent B's input)
- Downstream pipeline integration (feeding artifacts to build systems, test runners)
- Reporting and provenance analysis
- Human review of generated artifacts

The existing framework provides point queries (`get_artifact(id)`, `get_provenance(id)`) but lacks formal characterization of bulk extraction with filtering.

---

## 2. Formal Definition

### Definition 17: Extraction Function

An **Extraction Function** is a pure query over the Versioned Repository:

```
E: ℛ × Φ → 2^ℛ
```

Where:

- **ℛ** is the Versioned Repository (Definition 2) — the append-only DAG of repository items (Definition 10)
- **Φ** is a **Filter Predicate** specifying extraction criteria
- **2^ℛ** is the powerset of repository items (the result is a subset of all items in ℛ)

### Definition 18: Filter Predicate

A **Filter Predicate** Φ is a boolean function over repository item fields:

```
Φ: r → {⊤, ⊥}
```

Common filter predicates include:

- `Φ_status(s)`: `r.status = s` (e.g., ACCEPTED, REJECTED)
- `Φ_action_pair(id)`: `r.action_pair_id = id`
- `Φ_workflow(wf)`: `r.workflow_id = wf`
- `Φ_compound`: `Φ₁ ∧ Φ₂ ∧ ... ∧ Φₙ` (conjunction of predicates)

### Extraction Semantics

The extraction function returns all repository items satisfying the predicate:

```
E(ℛ, Φ) = {r ∈ ℛ | Φ(r) = ⊤}
```

---

## 3. Theorem: Extraction Preserves System Dynamics

**Theorem 3 (Extraction Invariance):** The extraction function E does not modify system state. For any extraction E(ℛ, Φ):

```
S' = S    (where S = ⟨S_workflow, S_env⟩)
```

**Proof:**

1. **ℛ is unchanged:** Extraction performs no write operations. The DAG ℛ remains identical before and after E invocation.

2. **S_workflow is unchanged:** The workflow state transitions only via T(s_w, v) where v is a guard verdict (Definition 8). Extraction does not invoke any guard, hence no transition occurs.

3. **S_env is unchanged:** The environment state S_env is unchanged. Extraction:
   - Does not create new repository items (ℛ unchanged)
   - Does not modify context (C unchanged)
   - Does not append to feedback history H (H unchanged)

4. **Context refinement is not triggered:** Definition 5 specifies context evolution: `C_{k+1} = ⟨Ψ, H_k ∪ {(a_k, φ_k)}⟩`. Extraction produces no artifact a_k and no feedback φ_k, hence C remains invariant.

∎

**Corollary 3.1 (Idempotence):** Extraction is idempotent:

```
E(ℛ, Φ) = E(E(ℛ, Φ), Φ) = E(ℛ, Φ)
```

**Corollary 3.2 (Referential Transparency):** For fixed ℛ and Φ, repeated extraction yields identical results:

```
∀t₁, t₂: E_t₁(ℛ, Φ) = E_t₂(ℛ, Φ)    (assuming no concurrent writes to ℛ between t₁ and t₂)
```

---

## 4. Classification: Query vs. Action

The framework distinguishes three operation types:

| Category | Examples | Modifies S? | Formal Type |
|----------|----------|-------------|-------------|
| **Actuator** | Generator (`a_gen`) | Yes (appends to ℛ) | C → A |
| **Sensor** | Guard (`G`) | Yes (projects to S_workflow) | A × C → (v, φ) |
| **Query** | Extractor (`E`) | No | ℛ × Φ → 2^ℛ |

### Key Distinction

Extractors are fundamentally different from Generators and Guards:

1. **Generators** are stochastic and produce artifacts that become repository items
2. **Guards** are deterministic and trigger state transitions
3. **Extractors** are deterministic read-only projections

An Extractor cannot be an "Actuator" because it produces no effect on the system state. It is analogous to a database SELECT query — it retrieves but does not modify.

### Architectural Position

```
┌─────────────────────────────────────────────────┐
│                  Agent Boundary                  │
├─────────────────────────────────────────────────┤
│  Actuators (Write)     │  Queries (Read)        │
│  ├── Generator (a_gen) │  ├── Extractor (E)     │
│  └── (Side Effects)    │  ├── get_item()        │
│                        │  └── get_provenance()  │
├─────────────────────────────────────────────────┤
│  Sensors (Project)                              │
│  └── Guard (G) → S_workflow                     │
└─────────────────────────────────────────────────┘
```

---

## 5. Multi-Agent Implications

When multiple agents share ℛ as a common substrate for coordination, extraction becomes a critical primitive.

### 5.1 Concurrent Read Safety

**Property (Read-Read Commutativity):** Multiple concurrent extractions do not conflict:

```
E₁(ℛ, Φ₁) ∥ E₂(ℛ, Φ₂) → No conflict
```

Since ℛ is append-only (Definition 2), reads observe a consistent snapshot. No locking required.

### 5.2 Read-Write Ordering

**Property (Read-After-Write Consistency):** If agent A stores repository item r at time t₀, then agent B's extraction at t₁ > t₀ will include r (assuming the filter matches):

```
store(r) at t₀ → E(ℛ, Φ) at t₁ includes r    (if Φ(r) = ⊤)
```

This follows from the append-only property: once written, repository items are permanently visible.

### 5.3 Cross-Workflow Item Sharing

Agents can consume repository items from other workflows via extraction:

```
Agent A (workflow_id: wf_1):
  Produces repository item r₁ with action_pair_id: "g_impl"
  Guard validates: status → ACCEPTED

Agent B (workflow_id: wf_2):
  Extracts: E(ℛ, Φ_action_pair("g_impl") ∧ Φ_status(ACCEPTED))
  Receives: {r₁, ...}
  Uses r₁.a as dependency input
```

This enables **loose coupling** between agents: they communicate through the shared DAG rather than direct message passing.

### 5.4 Eventual Consistency Guarantees

For a multi-agent system:

1. **Strong Consistency within Workflow:** A workflow's own items are immediately visible after `store()`
2. **Eventual Consistency across Workflows:** Other workflows observe items after filesystem/index sync
3. **Monotonic Read:** Once a repository item is observed, it cannot "disappear" (append-only)

---

## 6. Interface Extension

### Proposed Addition to ArtifactDAGInterface

```python
def extract(
    self,
    filter: Callable[[RepositoryItem], bool],
    *,
    limit: int | None = None,
    order_by: str = "created_at",
    descending: bool = True,
) -> list[RepositoryItem]:
    """
    Extract repository items matching filter predicate.

    Implements E: ℛ × Φ → 2^ℛ (Definition 17).

    This is a read-only operation that does not modify S_workflow or S_env.

    Args:
        filter: Predicate function Φ: r → {⊤, ⊥}
        limit: Maximum items to return (pagination)
        order_by: Sort key (created_at, action_pair_id, etc.)
        descending: Sort direction

    Returns:
        List of repository items where Φ(r) = ⊤
    """
    pass
```

### Common Filter Factories

```python
# Pre-built predicates for common use cases
def by_status(status: ArtifactStatus) -> Callable[[RepositoryItem], bool]:
    return lambda r: r.status == status

def by_action_pair(action_pair_id: str) -> Callable[[RepositoryItem], bool]:
    return lambda r: r.action_pair_id == action_pair_id

def by_workflow(workflow_id: str) -> Callable[[RepositoryItem], bool]:
    return lambda r: r.workflow_id == workflow_id

def accepted() -> Callable[[RepositoryItem], bool]:
    return by_status(ArtifactStatus.ACCEPTED)

# Compound predicates
def and_(*predicates: Callable[[RepositoryItem], bool]) -> Callable[[RepositoryItem], bool]:
    return lambda r: all(p(r) for p in predicates)
```

---

## 7. Relationship to Existing Operations

| Operation | Returns | Definition | Extraction Equivalent |
|-----------|---------|------------|----------------------|
| `get_item(id)` | Single item | Point query | `E(ℛ, λr. r.id = id)[0]` |
| `get_provenance(id)` | Chain list | Traversal | `E(ℛ, λr. r ∈ chain(id))` |
| `get_by_action_pair(id)` | All for action | Filter | `E(ℛ, Φ_action_pair(id))` |
| `get_accepted(id)` | Single accepted | Filter + status | `E(ℛ, Φ_action_pair(id) ∧ Φ_status(ACCEPTED))[0]` |
| `get_by_workflow(wf)` | All for workflow | Filter | `E(ℛ, Φ_workflow(wf))` |

The extraction function E generalizes all existing read operations.

---

## 8. Summary

1. **Extraction does NOT require a new actuator.** It is a read-only query function over ℛ.

2. **System dynamics are preserved.** Extraction does not invoke T(s_w, v), does not modify S_env, and does not trigger context refinement.

3. **Multi-agent coordination is enabled.** The append-only DAG provides a consistent shared substrate for cross-workflow item sharing.

4. **The formal extension is minimal.** Definition 17 (Extraction Function) and Definition 18 (Filter Predicate) extend the framework, building on repository items (Definition 10).

---

## See Also

- [01_versioned_environment.md](01_versioned_environment.md) — Repository items (Definitions 10-16)
- [00_notation_extensions.md](00_notation_extensions.md) — Symbol table including E, Φ
- [03_multi_agent_workflows.md](03_multi_agent_workflows.md) — Multi-agent extension using extraction (Definitions 19-20)
- [../agent_design_process/domain_notation.md](../agent_design_process/domain_notation.md) — Base notation (Definitions 1-9)
- [../agent_design_process/agent_program.md](../agent_design_process/agent_program.md) — Implementation of ArtifactDAGInterface
