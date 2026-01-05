# Domain Definitions

This document formalizes the relationship between deterministic workflows and stochastic generators, distinguishing foundational concepts in agency from the specific architectural interpretations employed in this framework.

> **See also**: [domain_notation.md](domain_notation.md) (symbol reference), [domain_ubiquitous_language.md](domain_ubiquitous_language.md) (DDD terms)

---

## 2.1 Foundational Definitions

### 2.1.1 Agency

- **Agent Function (Russell & Norvig, 2020):** A specification mapping percept history to action selection. Defines *what* the agent decides.

- **Agent Program (Russell & Norvig, 2020):** An implementation of the agent function on a specific architecture, mapping current percept to action. Defines *how* the agent decides.

- **Agent Taxonomy:** The progression from simple reflex → model-based reflex → goal-based → utility-based agents, distinguished by internal state maintenance and reasoning sophistication.

- **Weak Agency (Wooldridge & Jennings, 1995):** A software system exhibiting autonomy, reactivity, and pro-activeness without implying consciousness or mental states.

### 2.1.2 Actor Functional Architecture (Ghallab, Nau & Traverso, 2025)

- **Planning:** Determining *what to do*. Open-loop search over predicted states using a predictive model. Synthesizes an organized set of actions leading to a goal. The designer/orchestrator holds the goal representation and performs goal-based search offline.

- **Acting:** Determining *how to do* chosen actions. Closed-loop process with feedback from observed effects. Progressive refinement of abstract actions into concrete commands given current context.

- **Learning:** Improving performance with greater autonomy and versatility. Two modes:
  - *End-to-end:* Reactive black-box function; effective but difficult to verify.
  - *Model-based:* Explicit predictive models; supports analysis and explanation.

- **Note:** The Agent Function/Agent Program distinction parallels Ghallab et al.'s Descriptive/Operational model distinction. Derivation rules formalize this transformation.

### 2.1.3 Rationality Constraints

- **Bounded Rationality (Simon, 1955):** Rational agents under computational constraints do not optimize globally; they *satisfice*, selecting the first solution meeting the aspiration level within the available search budget.

- **Control Boundary (Sutton & Barto, 1998):** The agent comprises only components modifiable by the control policy. Components outside this boundary constitute the environment.

### 2.1.4 Cooperation Models

- **Promise Theory (Burgess, 2015):** A model of voluntary cooperation where autonomous agents issue promises regarding intended behavior. The consumer bears responsibility for verifying promise fulfillment, replacing command-and-control assumptions.

---

## 2.2 Planning-Execution Separation

### 2.2.1 Offline Planning (Goal-Based Search)

- **Workflow as Pre-Computed Plan:** The workflow structure is the output of goal-based search performed at design time. The designer reasons about hypothetical action sequences to construct a state machine satisfying goal predicates.

- **Goal Representation:** The designer/orchestrator holds explicit goal representations. Runtime agents inherit the goal structure implicitly through preconditions and guard predicates.

- **Outputs:** Workflow state machine, guard specifications, action preconditions, postcondition assertions.

### 2.2.2 Online Execution (Precondition-Gated Reflex)

- **State Sensing:** The agent observes current environment state through predicate evaluation. Multiple facets may be sensed simultaneously (e.g., specification alignment, test coverage, code correctness).

- **Action Applicability:** A function φ: S × A → {applicable, blocked} determines which actions are available. Guards implement applicability predicates at runtime.

- **Guard-Verified Transitions:** State commits only after guard validation. Invalid generations are rejected without polluting workflow state. The agent does not search—it reacts to verified observations.

- **No Runtime Goal Search:** The executing agent follows pre-computed structure. Goals are implicit in precondition ordering, not explicit representations the agent reasons about.

### 2.2.3 Execution Trace (Directed Acyclic Graph)

- **Structure:** A DAG capturing the complete execution history where:
  - *Nodes:* Generation events, guard evaluations, state snapshots, artifact versions
  - *Edges:* State transitions, retry branches, artifact dependencies, causal links

- **Properties:**
  - *Append-only:* History is never modified, only extended
  - *Immutable:* Past nodes and edges cannot be altered
  - *Enables Replay:* Any execution path can be reconstructed from the trace

- **Retry Branching:** Multiple generation attempts at the same workflow state produce sibling nodes. Only the branch reaching guard satisfaction (⊤) advances the workflow.

- **Artifact Dependencies:** Edges encode that output of action A feeds input of action B, capturing data flow orthogonal to control flow.

- **Relation to S_env:** The execution trace *is* the information state. S_env is the append-only versioned repository; the DAG is its structure.

- **Bridge to Learning:** The trace serves as:
  - Substrate for in-context learning (§2.3.1)—successful paths inform subsequent generations
  - Training data for model adaptation (§2.3.2)—execution history enables fine-tuning

---

## 2.3 Learning Modes

### 2.3.1 Intra-Episode (In-Context Learning)

- S_env accumulates generation history, guard feedback, and artifact provenance within a single execution episode.

- The LLM conditions on prior attempts, guard failure reasons, and successful patterns without weight modification.

- Satisficing applies: learning continues until guard returns ⊤ or retry budget exhausted (⊥_fatal).

### 2.3.2 Inter-Episode (Model Adaptation)

- Execution traces from completed episodes provide training signal for:
  - LoRA / adapter parameter updates
  - Distillation from successful execution paths
  - Reinforcement from guard feedback (⊤ as reward signal)

- Requires sufficient compute; operates outside the control boundary during intra-episode execution.

---

## 2.4 Architectural Definitions

### 2.4.1 Control Boundary (Generative Application)

Applying Sutton & Barto's definition to LLM-based systems, the agent boundary is defined by modifiability relative to time horizon:

- **Intra-Episode:** Agent controls context composition (C) and workflow state transitions (S_workflow). The LLM is part of the environment.

- **Inter-Episode:** With sufficient compute, the agent may control adapter parameters (LoRA) or distilled weights.

- **Base Model:** Pre-trained weights remain permanently in the environment, providing a stochastic generation oracle.

### 2.4.2 Dual-State Architecture

The system state space S separates into two distinct spaces:

- **S_workflow (Control State):** A deterministic finite state machine tracking goal progress, guard satisfaction, and transition history. Commits only on guard success.

- **S_env (Information State):** The execution trace DAG (§2.2.3). Append-only, versioned, accumulates all generations (successful and failed), guard feedback, and artifacts. Enables in-context learning without polluting control flow.

### 2.4.3 Atomic Action Pair

The generator-guard coupling ensuring deterministic control over stochastic generation:

- **Generator:** Produces candidate output conditioned on context (prompt, S_env history).

- **Guard:** Deterministic predicate evaluating generation validity.

- **Tri-State Semantics:**
  - ⊤ (success): Generation satisfies validity criteria; commit to S_workflow
  - ⊥_retry (recoverable): Generation invalid but retry permitted; append to S_env, re-invoke generator
  - ⊥_fatal (unrecoverable): Retry budget exhausted or unrecoverable failure; escalate

- **Satisficing Interpretation:**
  - ⊤ = aspiration level met (Simon)
  - Retry budget = available search constraint
  - First generation achieving ⊤ is accepted; global optimum not sought

---

## References

- Burgess, M. (2015). *Promise Theory: Principles and Applications*
- Ghallab, M., Nau, D., & Traverso, P. (2025). *Acting, Planning, and Learning*. Cambridge University Press
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.)
- Simon, H. (1955). A Behavioral Model of Rational Choice
- Sutton, R., & Barto, A. (1998). *Reinforcement Learning: An Introduction*
- Wooldridge, M., & Jennings, N. (1995). Intelligent Agents: Theory and Practice
