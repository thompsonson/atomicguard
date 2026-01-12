# Formal Framework Extensions

This directory contains formal extensions to the Dual-State Domain framework presented in [arXiv:2512.20660](https://arxiv.org/abs/2512.20660).

These extensions preserve the base system dynamics (Definitions 1-9) while introducing additional concepts for practical agent implementations: repository items with configuration snapshots, read-only extraction queries, multi-agent coordination, and continuous learning from workflow traces.

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

### [04 — Learning Loop](04_learning_loop.md)

**Definitions 21-24, Theorems 9-10.** Formalizes continuous learning from workflow execution traces. Defines training trace extraction, sparse reward signals from guard verdicts, and policy updates for fine-tuning the generator. Includes analysis of what the model learns and potential issues with mitigations.

### [05 — Learning Implementation](05_learning_implementation.md)

**Practical guide.** Implementation guidance for the Learning Loop extension using Unsloth and LoRA adapters. Covers dataset extraction, prompt formatting, training configuration, filtering strategies, incremental training, and evaluation.

### [06 — Generated Workflows](06_generated_workflows.md)

**Definitions 25-32, Theorems 11-13.** Treats workflows as generated artifacts, enabling agents to determine their own execution path. Introduces the Planner ActionPair, two-level execution, and configurable escalation policies.

### [07 — Incremental Execution](07_incremental_execution.md)

**Definitions 33-37.** Enables skipping unchanged action pairs based on configuration fingerprints. Supports efficient re-execution when only parts of the specification change.

### [08 — Composite Guards](08_composite_guards.md)

**Definitions 38-43, Theorem 14.** Formalizes guard composition patterns for combining multiple validation checks into a single guard function. Enables fail-fast sequential execution, parallel concurrent validation, and nested composition while preserving system dynamics.

---

## Reading Order

1. **Notation** — Symbol reference
2. **Versioned Environment** — Foundation (repository items, W_ref)
3. **Extraction** — Queries over repository items
4. **Multi-Agent** — Agents sharing repository items
5. **Learning Loop** — Training from workflow traces
6. **Learning Implementation** — Practical fine-tuning guide
7. **Generated Workflows** — Dynamic workflow generation
8. **Incremental Execution** — Skip unchanged action pairs
9. **Composite Guards** — Guard composition patterns

---

## Relationship to Paper

The base paper defines artifacts `a ∈ A` as simple content. These extensions introduce **repository items** `r ∈ ℛ` that wrap artifacts with configuration snapshots and provenance metadata. The base system dynamics (Definition 7) are preserved — repository items simply carry additional context.

| Concept | Paper | Extension |
|---------|-------|-----------|
| Artifact | `a ∈ A` | `r = ⟨a, Ψ, Ω, W_ref, H, source, metadata⟩` |
| History | Implicit | Emergent from ℛ DAG |
| Workflow | Implicit in P | Explicit via W_ref |
| Learning | Section 6.1 (conceptual) | Definitions 21-24 (formal) |

---

## Dependency Graph

```
┌─────────────────────────────────────────┐
│  Versioned Environment (Def 10-16)      │  ← Foundation
└─────────────────────────────────────────┘
              │
              ├──────────────────────────────┬────────────────────────┐
              ▼                              ▼                        │
┌──────────────────────────┐    ┌────────────────────────────┐       │
│  Artifact Extraction     │    │  Multi-Agent Workflows     │       │
│  (Def 17-18)             │    │  (Def 19-20)               │       │
└──────────────────────────┘    └────────────────────────────┘       │
              │                                                       │
              ├───────────────────────────────────────────────────────┤
              ▼                                                       │
┌──────────────────────────┐                                         │
│  Learning Loop           │                                         │
│  (Def 21-24)             │                                         │
└──────────────────────────┘                                         │
              │                                                       │
              ▼                                                       │
┌──────────────────────────┐    ┌────────────────────────────┐       │
│  Implementation Guide    │    │  Generated Workflows       │◄──────┘
│  (Unsloth/LoRA)          │    │  (Def 25-32)               │
└──────────────────────────┘    └────────────────────────────┘
                                              │
                                              ▼
                                ┌────────────────────────────┐
                                │  Incremental Execution     │
                                │  (Def 33-37)               │
                                └────────────────────────────┘

┌─────────────────────────────────────────┐
│  Composite Guards (Def 38-43)           │  ← Independent (Base Paper only)
│  Guard composition patterns             │
└─────────────────────────────────────────┘
```
