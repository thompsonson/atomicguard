# Formal Framework Extensions

This directory contains formal extensions to the Dual-State Domain framework presented in [arXiv:2512.20660](https://arxiv.org/abs/2512.20660).

These extensions preserve the base system dynamics (Definitions 1-9) while introducing additional concepts for practical agent implementations: repository items with configuration snapshots, read-only extraction queries, and multi-agent coordination through shared state.

---

## Extensions

### [00 — Notation Extensions](00_notation_extensions.md)

Symbol reference for all extension definitions. Start here to understand the notation conventions, particularly the distinction between base artifacts (`a ∈ A`) and repository items (`r ∈ ℛ`).

### [01 — Versioned Environment](01_versioned_environment.md)

**Definitions 10-16.** The foundational extension. Introduces repository items that wrap artifacts with configuration snapshots (Ψ, Ω, W_ref), enabling checkpointing, resume, and human-in-the-loop workflows without external state management.

### [02 — Artifact Extraction](02_artifact_extraction.md)

**Definitions 17-18.** Formalizes read-only queries over the repository. Proves extraction preserves system dynamics (Theorem 3) and enables cross-workflow artifact sharing.

### [03 — Multi-Agent Workflows](03_multi_agent_workflows.md)

**Definitions 19-20.** Extends the framework to multiple agents sharing the same repository. Coordination emerges from the shared DAG — no explicit message passing or consensus protocols required.

---

## Reading Order

1. **Notation** — Symbol reference
2. **Versioned Environment** — Foundation (repository items, W_ref)
3. **Extraction** — Queries over repository items
4. **Multi-Agent** — Agents sharing repository items

---

## Relationship to Paper

The base paper defines artifacts `a ∈ A` as simple content. These extensions introduce **repository items** `r ∈ ℛ` that wrap artifacts with configuration snapshots and provenance metadata. The base system dynamics (Definition 7) are preserved — repository items simply carry additional context.

| Concept | Paper | Extension |
|---------|-------|-----------|
| Artifact | `a ∈ A` | `r = ⟨a, Ψ, Ω, W_ref, H, source, metadata⟩` |
| History | Implicit | Emergent from ℛ DAG |
| Workflow | Implicit in P | Explicit via W_ref |
