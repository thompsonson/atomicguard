# AtomicGuard Architecture

This document explains the architectural principles behind AtomicGuard and why they matter for building reliable AI agents.

## Four Aspects

AtomicGuard addresses compounded hallucinations through four complementary aspects:

| Aspect | What It Solves |
|--------|----------------|
| 🛡️ **Safety** | Errors caught immediately via guard validation |
| 💾 **State** | Full context preserved for debugging and resume |
| 🌐 **Scale** | Multiple agents without coordination complexity |
| 📈 **Improvement** | System learns from its own failures |

The core principle is **Bounded Indeterminacy**: the LLM generates content, but a deterministic state machine controls the logic. Goals are decomposed into small, measurable tasks—each validated before the workflow advances.

| Layer | Controller | Nature |
|-------|------------|--------|
| **Content** | LLM (Generator) | Stochastic |
| **Logic** | State Machine (Workflow) | Deterministic |
| **Validation** | Guards | Deterministic |

This transforms the problem from:

- ❌ "Hope the agent gets it right" (unbounded search over complex goals)
- ✅ "Ensure each step converges" (bounded validation of small, measurable tasks)

---

## The 4 Aspects in Detail

### 🛡️ Safety: Atomic Action Pairs

Every generation is wrapped in a **guard transaction**:

```
ActionPair = ⟨Generator, Guard⟩
```

The workflow state **never advances** unless the guard passes. This is the "atomic" in AtomicGuard:

| Guard Result | Meaning | Action |
|--------------|---------|--------|
| ⊤ (pass) | Generation is valid | Advance workflow |
| ⊥_retry | Generation failed, recoverable | Retry with feedback |
| ⊥_fatal | Unrecoverable failure | Escalate to human |

**Why it matters**: Errors are caught immediately, before they can compound. The workflow state remains clean.

### 💾 State: Versioned Environments

Every artifact is stored with its **configuration snapshot**:

```
RepositoryItem = ⟨artifact, specification, constraints, workflow_ref, history, source⟩
```

This enables:

- **Checkpointing**: Pause and resume without losing context
- **Time travel**: Inspect any prior state
- **Reproducibility**: Know exactly what configuration produced each artifact
- **Audit trail**: Track how configuration evolved

**Why it matters**: Treat agent memory like `git`. Every state is recoverable, every change is traceable.

### 🌐 Scale: Emergent Coordination

Multiple agents coordinate via a **shared DAG** (Directed Acyclic Graph):

```
Agent A writes → Repository ← Agent B reads
```

This is the **Blackboard Pattern**:

- No message buses or coordination protocols
- Agents read from and write to the shared repository
- One agent's output is another agent's input
- Coordination emerges from the data structure, not explicit communication

**Why it matters**: Scaling to multiple agents doesn't require complex distributed systems infrastructure. The append-only DAG provides natural consistency guarantees.

### 📈 Improvement: The Learning Loop

Every guard verdict is a **training signal**:

| Verdict | Signal |
|---------|--------|
| ⊤ (accepted) | Positive example |
| ⊥ (rejected) | Negative example with feedback |

The system extracts **Training Traces** from execution history:

- Successful (specification → artifact) pairs for supervised fine-tuning
- Guard feedback for reinforcement learning
- Retry chains showing correction patterns

**Why it matters**: Runtime failure is training data. The system improves from its own mistakes.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  WORKFLOW (Deterministic FSM)                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Step 1: ⟨TestGenerator, SyntaxGuard⟩                   ││
│  │  Step 2: ⟨ImplGenerator, TestGuard⟩  [requires: Step 1] ││
│  │  Step 3: ⟨DocGenerator, FormatGuard⟩ [requires: Step 2] ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  EXECUTION LOOP (per step)                                  │
│                                                             │
│  1. Generate: artifact ← Generator(context)                │
│  2. Validate: result ← Guard(artifact)                     │
│  3. Branch:                                                 │
│     - ⊤: Store artifact, advance to next step              │
│     - ⊥_retry: Add feedback to context, retry (up to rmax) │
│     - ⊥_fatal: Checkpoint, escalate to human               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  REPOSITORY (Append-only DAG)                               │
│                                                             │
│  All artifacts stored with full context snapshot            │
│  Provenance links track retry chains                        │
│  Enables checkpoint/resume, extraction, learning            │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparison to Other Approaches

| Approach | How it handles errors | AtomicGuard difference |
|----------|----------------------|------------------------|
| **ReAct** | Retry with reasoning | Guards provide deterministic validation, not just LLM self-reflection |
| **Chain-of-Thought** | Hope reasoning prevents errors | Errors are caught by guards, not hoped away |
| **AutoGPT-style** | Let agent decide next action | Workflow structure is predetermined, only content is generated |
| **LangGraph** | Graph-based workflow | Similar structure, but AtomicGuard adds guard transactions and learning loop |

---

## See Also

- [Paper: Managing the Stochastic](https://arxiv.org/abs/2512.20660) — Full formal framework
- [Extensions](extensions/README.md) — Formal definitions (Definitions 10-32)
- [Getting Started](../getting-started.md) — Quick start guide
- [Decision Log](decisions/decisions.md) — Architectural decisions and rationale
