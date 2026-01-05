# Domain Ubiquitous Language

This document establishes the shared vocabulary for the AtomicGuard framework, combining Domain-Driven Design (DDD) terminology with the mathematical domain definitions from the research paper.

---

## Framework-Specific Terms

### Core Concepts

| Term | Definition | Related Symbol |
|------|------------|----------------|
| **Artifact** | An immutable generated output with version tracking. The concrete result of a generation step. | a, aₖ |
| **Guard** | A deterministic verification function that validates artifacts against specifications. Returns verdict + feedback. | G |
| **Generator** | A stochastic function that produces artifacts from context. May be a single LLM call or an autonomous agent. | a_gen |
| **ActionPair** | An atomic transaction coupling a generator with a guard. Both succeed together or both fail together. | A = ⟨ρ, a_gen, G⟩ |
| **Workflow** | An orchestrator that executes sequential ActionPairs, managing state transitions and dependencies. | — |
| **Context** | The hierarchical composition of ambient environment, local specification, and feedback history. | C |
| **Specification** | The static requirements, tests, and constraints for a specific step. | Ψ |

### State Concepts

| Term | Definition | Related Symbol |
|------|------------|----------------|
| **Workflow State** | The deterministic, observable state tracking guard satisfaction. A finite state machine. | S_workflow, σ |
| **Environment State** | The stochastic, partially opaque state containing artifacts and context. | S_env |
| **Dual-State** | The architectural pattern separating control (workflow) from information (environment). | S = S_workflow × S_env |
| **Guard Satisfaction** | The condition where a guard returns ⊤ (pass), enabling workflow state advancement. | σ(g) = ⊤ |

### Execution Concepts

| Term | Definition | Related Symbol |
|------|------------|----------------|
| **Refinement Loop** | The retry cycle where guard failure triggers context enrichment with feedback. | H_{k+1} = Hₖ ∪ {(aₖ, φₖ)} |
| **Feedback History** | The accumulated sequence of rejected artifacts and their diagnostic messages. | H |
| **Retry Budget** | The maximum number of generation attempts allowed per workflow node. | R_max |
| **Escalation** | The action of surfacing a fatal failure to human oversight when retries cannot resolve it. | ⊥_fatal |

### Verdict Types

| Term | Symbol | Description |
|------|--------|-------------|
| **Pass** | ⊤ | Guard validation succeeded; advance workflow state |
| **Retry** | ⊥_retry | Guard validation failed; refine context and retry |
| **Fatal** | ⊥_fatal | Non-recoverable failure; escalate to human immediately |

---

## DDD Building Blocks

### Tactical Patterns Used

| Pattern | Usage in AtomicGuard |
|---------|---------------------|
| **Entity** | `Artifact` — has identity (artifact_id, version), mutable through versioning |
| **Value Object** | `GuardResult`, `Context` — immutable, defined by attributes |
| **Repository** | `ArtifactDAGInterface` — provides access to artifact history |
| **Factory** | `GeneratorInterface` — creates artifacts from context |
| **Service** | `Workflow`, `DualStateAgent` — orchestrates domain logic |

### Strategic Patterns Applied

| Pattern | Application |
|---------|-------------|
| **Bounded Context** | Separation between Workflow (control) and Environment (information) |
| **Anti-Corruption Layer** | Guards act as ACL between stochastic generation and deterministic workflow |
| **Ubiquitous Language** | This document; shared vocabulary across code, docs, and paper |

---

## Architectural Definitions

### From the Research Paper

| Term | Definition | Source |
|------|------------|--------|
| **Control Boundary** | The boundary defining what the agent can modify. Components outside constitute the environment. | Sutton & Barto, 1998 |
| **Goal-Based Agent** | A rational decision function treating stochastic generations as percepts, not actions. Control flow is deterministic. | Russell & Norvig, 1995 |
| **Neuro-Symbolic System** | Architecture integrating neural generation with symbolic verification. LLM issues promises; guards verify fulfillment. | Framework definition |
| **Promise Theory** | A model where autonomous agents issue promises regarding intended behavior. Consumer verifies fulfillment. | Burgess, 2015 |
| **Bounded Rationality** | Agents satisfice (select first valid solution) rather than optimize for global maximum. | Simon, 1955 |
| **Weak Agency** | Software exhibiting autonomy, reactivity, and pro-activeness without implying consciousness. | Wooldridge & Jennings, 1995 |

### Dual-State Architecture

The core architectural pattern separating:

1. **Observable World (Agent/Workflow)**
   - Precondition checking (ρ)
   - Context composition (C)
   - State transitions (T)

2. **Opaque World (Environment)**
   - Generator execution (a_gen)
   - Artifact production (a)
   - Internal reasoning (hidden from workflow)

The **Control Boundary** lies between context composition and generator execution.

---

## Guard Terminology

### Guard Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **Syntax Guards** | Validate structural correctness | SyntaxGuard (AST), TypeGuard (mypy) |
| **Semantic Guards** | Validate behavioral correctness | TestGuard, DynamicTestGuard |
| **Architectural Guards** | Validate design constraints | ArchitectureBoundaryGuard, DIContainerGuard |
| **Safety Guards** | Validate operational safety | SandboxGuard, TimeoutGuard, PathGuard |
| **Human Guards** | Require human approval | HumanGuard |
| **Composite Guards** | Combine multiple guards | CompositeGuard, ParallelGuard |

### Guard Notation

| Notation | Meaning |
|----------|---------|
| Gᵢ | Guard with identifier i |
| G† | Guard requiring human oversight (dagger notation) |
| G(a, C) → (v, φ) | Guard function signature |

---

## Context Hierarchy

```
C_total = ⟨E, C_local, H_feedback⟩
         │     │         │
         │     │         └── Transient: Rejections for current node
         │     │
         │     └── Local: ⟨Ψ, aₖ⟩ (specification + current artifact)
         │
         └── Ambient: ⟨R, Ω⟩ (repository + global constraints)
```

### Scope Visibility

| Scope | Visible To | Mutable |
|-------|-----------|---------|
| Ambient (E) | All steps | No (append-only R) |
| Local (C_local) | Current step | Yes (aₖ updates) |
| Feedback (H) | Current node retries | Yes (accumulates) |

---

## See Also

- [domain_notation.md](domain_notation.md) — Symbol reference table
- [domain_definitions.md](domain_definitions.md) — Foundational and architectural definitions
- [agent_function.md](agent_function.md) — Agent percepts and actions
- [agent_program.md](agent_program.md) — Implementation details
- [guard_definitions.md](../paper_guards/guard_definitions.md) — Complete guard catalog
