# Versioned Environment

This document formally extends the Dual-State Framework to embed configuration versioning within repository items stored in ℛ, enabling checkpoints, resume, and human-in-the-loop workflows without additional state management.

> **See also**: [00_notation_extensions.md](00_notation_extensions.md) (symbols), [../agent_design_process/domain_notation.md](../agent_design_process/domain_notation.md) (base notation)

---

## Relationship to Paper

This extension builds on Definitions 1-9 from the base paper. The paper defines artifacts `a ∈ A` as simple content. This extension introduces **repository items** `r ∈ ℛ` that wrap artifacts with configuration snapshots (Ψ, Ω, W_ref) and provenance metadata. The base system dynamics (Definition 7) are preserved — repository items simply carry additional context that enables checkpointing, extraction, and multi-agent coordination.

**Note**: The workflow reference (W_ref) is a NEW concept not present in the original paper. The paper's Planning Problem (Definition 9) treats workflow structure as implicit; this extension makes it explicit and content-addressable.

---

## 1. Motivation

The base framework (Definition 1) defines environment state as:

```
S_env = A × C
```

This formulation treats configuration (Ψ, Ω) as external to the artifact. However, practical workflows require:

1. **Checkpointing**: Pause and resume execution without losing context
2. **Reproducibility**: Know exactly what configuration produced each artifact
3. **Human amendment**: Allow humans to modify configuration mid-workflow
4. **Audit trail**: Track how configuration evolved over time
5. **Workflow integrity**: Ensure the workflow structure hasn't changed since checkpoint

The key insight: **if configuration is stored with each repository item in ℛ, these capabilities emerge naturally**.

### Remark (History is Emergent)

The Markov property means each repository item contains all state needed for the next step. Configuration "history" is not stored explicitly — it's the ℛ DAG itself. We don't need separate `Ψ_history`, `Ω_history`, or `W_history` concepts. The history is implicit: `ℛ = {r₀, r₁, ..., rₖ}`.

---

## 2. Repository Items

### Definition 10: Repository Item

A **repository item** is a tuple that wraps a base artifact with configuration context:

```
r = ⟨a, Ψ, Ω, W_ref, H, source, metadata⟩
```

Where:

- **a ∈ A**: The base artifact (content produced by a generator)
- **Ψ**: Specification snapshot at generation time
- **Ω**: Global constraints snapshot at generation time
- **W_ref**: Content-addressed reference to workflow structure (Definition 11)
- **H**: Feedback history for this attempt
- **source ∈ {GENERATOR, HUMAN}**: Origin of the artifact
- **metadata**: Timestamps, IDs, provenance links

The repository ℛ is an append-only DAG of items: `ℛ = {r₀, r₁, ..., rₖ}`.

### Remark (Configuration Immutability)

Once stored, a repository item's configuration snapshot (Ψ, Ω, W_ref) is immutable. This ensures:

1. Reproducibility: Re-running with the same item ID uses identical configuration
2. Auditability: Configuration changes are explicit, not implicit

### Remark (Markov Property)

The repository item satisfies the Markov property for workflow execution. Given the current item r and the resolved workflow W, the next state is conditionally independent of all prior items. The workflow W acts as a static parameter (the "rules"), while (Ψ, Ω, H) capture all dynamic state needed for the current step.

---

## 3. Workflow Reference

### Definition 11: Workflow Reference

A **workflow reference** is a content-addressed pointer to a workflow structure:

```
W_ref = hash(W)
```

Where:

- **W**: The workflow structure (action pairs, dependencies, preconditions)
- **hash**: A cryptographic hash function (e.g., SHA-256)

### Axiom (Integrity)

For all W_ref stored in repository items:

```
∀ W_ref stored in r ∈ ℛ:
    hash(resolve(W_ref)) = W_ref
```

### Remark (Implementation-Defined Resolution)

The **resolve** function is implementation-defined. Valid implementations include:

| Storage Strategy | resolve(W_ref) |
|------------------|----------------|
| W stored in ℛ | Lookup in ℛ by content hash |
| W stored in Git | `git cat-file` by hash |
| W stored in filesystem | Read file, verify hash |
| W stored inline | Extract W_content from item metadata |

The formal framework is agnostic to storage location—only the integrity axiom must hold.

### Remark (Content = Identity)

Following best practices from Git, IPFS, and other content-addressable systems:

- The hash IS the version identifier (no separate version counter)
- Identical workflow content produces identical W_ref
- Workflow changes produce new W_ref values

---

## 4. Configuration Evolution

### Definition 12: Configuration Amendment

Configuration evolves monotonically through amendment operations:

```
Ψ_{k+1} = Ψ_k ⊕ Δ_Ψ    (specification amendment)
Ω_{k+1} = Ω_k ⊕ Δ_Ω    (constraint amendment)
```

Where ⊕ denotes a composition operation (e.g., merge, override, patch).

**Properties:**

- Each version is immutable once stored
- Amendments create new versions, preserving the prior version
- The amendment Δ may be empty (no change from prior version)

### Remark (Monotonic Growth)

Configuration history only grows. There is no "rollback" operation—instead, a new version is created that reverts to prior values. This preserves the complete audit trail.

---

## 5. Context Derivation

### Definition 13: Context Derivation Function

Context C is **derived from** a repository item's stored configuration:

```
C(r) = ⟨E(r), C_local(r), H(r)⟩
```

Where:

- **E(r) = ⟨ℛ, r.Ω⟩**: Ambient environment using item's constraint version
- **C_local(r) = ⟨r.Ψ, r.a⟩**: Local context using item's specification and artifact
- **H(r) = r.H**: Stored feedback history

### Remark (No External State)

This derivation means context is fully recoverable from any repository item in ℛ. There is no external "current context" that must be synchronized—ℛ is the single source of truth.

---

## 6. Checkpoint and Resume

### Definition 14: Checkpoint

A **checkpoint** is a pointer into ℛ:

```
Checkpoint = r_id ∈ ℛ
```

A checkpoint is **not** a separate snapshot. The full execution state is derivable from the referenced item:

- Configuration (Ψ, Ω) stored in the item
- Workflow reference (W_ref) stored in the item
- Feedback history (H) stored in the item
- Workflow position derivable from item's action_pair_id
- Prior items accessible via provenance links

### Definition 15: Resume Function

The **resume function** continues execution from a stored item state:

```
Resume: ℛ × r_id → Execution

Resume(ℛ, id) =
    r = ℛ[id]
    W = resolve(r.W_ref)
    if hash(W) ≠ r.W_ref:
        ⊥_integrity_error
    Execute(C(r), W)
```

Resume reconstructs context from the repository item, verifies workflow integrity, and continues the standard execution loop (Definition 7: System Dynamics).

### Theorem 4: Resume Preserves System Dynamics

**Statement**: Resuming from a checkpoint is semantically equivalent to having never paused.

**Proof**:

1. Let r_k be the repository item at checkpoint.
2. Context C(r_k) is derived from r_k's stored (Ψ, Ω, H).
3. Workflow W is resolved from r_k.W_ref with integrity verification.
4. The next generation step uses this context: a' ~ a_gen(C(r_k))
5. This is identical to Definition 7 step 1.
6. Sensing and state update follow Definition 7 steps 2-3.

No special "resume logic" is required—the standard dynamics apply.

∎

---

## 7. Human-in-the-Loop

### Definition 16: Human Generator

The **human generator** is a special generator with configuration amendment capability:

```
a_human: (C, Δ_Ψ, Δ_Ω) → a
```

Where:

- **C**: Current context (derived from prior repository item)
- **Δ_Ψ**: Optional specification amendment
- **Δ_Ω**: Optional constraint amendment
- **a**: Resulting artifact

**Properties:**

1. Subject to guard validation (preserves Definition 7)
2. When stored, creates repository item r with `source = HUMAN`
3. Configuration amendments (Δ_Ψ, Δ_Ω) are captured in the resulting repository item

### Remark (Amendment Types)

Human amendments fall into three categories:

| Type | Description | Effect |
|------|-------------|--------|
| **CONTENT** | Human provides artifact content | r.a = human_provided |
| **SPECIFICATION** | Human modifies Ψ | r.Ψ = Ψ_prev ⊕ Δ_Ψ |
| **CONSTRAINT** | Human modifies Ω | r.Ω = Ω_prev ⊕ Δ_Ω |

Amendments may be combined (e.g., provide content AND modify specification).

### Theorem 5: Human-in-the-Loop Preserves System Dynamics

**Statement**: Human-provided artifacts flow through the same Definition 7 dynamics as generator-produced artifacts.

**Proof**:

1. **Generation**: a' = a_human(C, Δ_Ψ, Δ_Ω) produces artifact a'
2. **Storage**: a' is wrapped into repository item r' with amended configuration
3. **Sensing**: ⟨v, φ⟩ = G(r'.a, C') where C' = C(r')
4. **State Update**: Based on verdict v (advance on ⊤, refine on ⊥)

The only differences from standard dynamics:

- source = HUMAN instead of GENERATOR
- Configuration may be amended (stored in r'.Ψ, r'.Ω)

Neither difference affects the structure of Definition 7.

∎

---

## 8. Memoization Implications

### Remark (Automatic Cache Invalidation)

With configuration stored per-item, memoization (Remark 3 in the base paper) becomes:

```
hash(r.Ψ, r.Ω, r.W_ref) = hash(r'.Ψ, r'.Ω, r'.W_ref) ⟹ cache_hit
```

Cache invalidation is automatic:

- Same (Ψ, Ω, W_ref) → cached result is valid
- Different configuration → new generation required

No explicit "invalidate cache when config changes" logic is needed.

---

## 9. Summary

1. **Repository items wrap artifacts** with configuration snapshots (Definition 10)

2. **Workflow references are content-addressed** with implementation-defined storage (Definition 11)

3. **Configuration evolves monotonically** through amendments (Definition 12)

4. **Context is derived** from repository items, not stored externally (Definition 13)

5. **Checkpoints are trivial** — just item pointers (Definition 14)

6. **Resume verifies workflow integrity** via content hash (Definition 15, Theorem 4)

7. **Human amendments** are formalized as a special generator (Definition 16, Theorem 5)

---

## Future Extensions

- **W evolution tracking**: Parent links between workflow versions (w.parent_ref) for tracking workflow changes across executions
- **S_workflow Derivation**: Compute workflow state entirely from ℛ, eliminating separate state tracking

---

## See Also

- [00_notation_extensions.md](00_notation_extensions.md) — Symbol table including r, W_ref, source
- [02_artifact_extraction.md](02_artifact_extraction.md) — Extraction function for querying ℛ (Definitions 17-18)
- [03_multi_agent_workflows.md](03_multi_agent_workflows.md) — Multi-agent coordination via shared ℛ (Definitions 19-20)
- [../agent_design_process/domain_notation.md](../agent_design_process/domain_notation.md) — Base notation (Definitions 1-9)
