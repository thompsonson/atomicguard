# Dual-State Agent: PEAS and Environment Analysis

## Environment Specification: PEAS Analysis

| Element | Description |
|---------|-------------|
| **Performance** | Return verified artifact within Rmax retries, or fail with provenance |
| **Environment** | User + LLM (generator) + Artifact DAG (git-backed) |
| **Actuators** | LLM queries, user responses |
| **Sensors** | User input, guard functions G (with scoped artifact access) |

### Percepts

| Percept | Type | Source |
|---------|------|--------|
| Specification Ψ | Immutable | User input |
| Artifact a | Mutable | LLM generation |
| Dependency artifacts | Immutable | Repository R (scoped by workflow) |
| Guard result (v, φ) | Tuple | Guard execution |
| Workflow state sw | Map G → {⊥, ⊤} | Internal |
| Retry count | Integer | Internal |

### Actions

| Category | Actions |
|----------|---------|
| **External (Actuators)** | QUERY-LLM, REPLY-TO-USER (includes failure with provenance) |
| **Sensing** | EXECUTE-GUARD |
| **Internal** | ADVANCE-STATE, REFINE-CONTEXT |

### Action Detail

| Action | Effect | Paper Reference |
|--------|--------|-----------------|
| QUERY-LLM | Produce artifact a' ∼ agen(C) | Definition 7.1 |
| EXECUTE-GUARD | Evaluate (v, φ) = G(a', deps) where deps scoped from R | Definition 7.2 |
| ADVANCE-STATE | sw[gid ↦ ⊤], clear Hfeedback | Definition 8, Equation 5 |
| REFINE-CONTEXT | Hk+1 = Hk ∪ {(a', φ)} | Definition 5, Equation 6 |
| REPLY-TO-USER | Return artifact or failure with provenance | Algorithm 1 |

---

## Environment Analysis

| Property | Classification | Justification |
|----------|---------------|---------------|
| **Observable** | Partially | Agent observes Sworkflow via guards; Senv (LLM internals) opaque. Guards project environment onto observable state (Proposition 1). |
| **Deterministic** | Hybrid | Guards deterministic (Lemma 1); generator stochastic (Assumption 1). Core contribution: deterministic control over stochastic generation. |
| **Static** | Static | Specification Ψ fixed during execution. Environment doesn't change while agent deliberates. |
| **Discrete** | Discrete | Binary guard outcomes {⊥, ⊤}. Finite workflow states. Discrete artifacts (strings/code). |
| **Agents** | Single | One agent executing sequential generate-verify loop. |
| **Known** | Partially | Guard semantics known; generator is black-box oracle with ε-capability assumption. |

**Key Insight**: The dual-state separation (Definition 1) exists precisely because observability differs—Sworkflow is fully observable, Senv is not.

**Remark (Guard Input Scoping)**: While Definition 6 provides guards access to the full context C, well-designed guards accept only minimal required inputs. The Workflow extracts specific artifacts from R and passes them explicitly, preserving guard simplicity and testability.
