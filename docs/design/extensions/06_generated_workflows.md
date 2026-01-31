# Generated Workflows

This document formally extends the Dual-State Framework to treat workflows as generated artifacts, enabling agents to determine their own execution path through goal-directed workflow generation.

> **Depends on**: [01_versioned_environment.md](01_versioned_environment.md) (W_ref, Definitions 10-16), [02_artifact_extraction.md](02_artifact_extraction.md) (extraction, Definitions 17-18)
>
> **See also**: [00_notation_extensions.md](00_notation_extensions.md) (symbols), [../agent_design_process/domain_notation.md](../agent_design_process/domain_notation.md) (base notation)

---

## Relationship to Paper

This extension builds on Definitions 1-9 from the base paper and Definitions 10-24 from Extensions 01-05. The paper defines workflows as static structures authored at design time (Definition 9: Planning Problem). This extension introduces **workflow artifacts** that are generated at runtime by a specialized "Planner ActionPair", enabling dynamic workflow composition while preserving the core Dual-State dynamics.

**Key insight**: The planner is "just another ActionPair" — it follows the same generate → guard → retry pattern as any other step. The difference is its output (a workflow definition) becomes the input to the next execution phase.

---

## 1. Motivation

The base framework separates planning from execution:

> "**Workflow as Pre-Computed Plan**: The workflow structure is the output of goal-based search performed at design time." (Section 2.2.1, domain_definitions.md)

This separation is intentional but limiting:

1. **Adaptability**: Static workflows cannot adapt to discovered requirements
2. **Reusability**: Similar goals require separate workflow definitions
3. **Learning**: Workflow structure is not subject to the training loop
4. **Autonomy**: Agents execute paths, but don't choose them

The key insight: **if workflows are artifacts, they can be generated, validated, stored, and learned from** — using the same mechanisms as any other artifact.

### Remark (Planning vs Acting)

This extension does NOT collapse the planning/execution distinction. Instead, it introduces a **two-level hierarchy**:

- **Meta-level**: Planning workflow (generates object workflows)
- **Object-level**: Executing the generated workflow

Both levels follow Definition 7 (System Dynamics). The meta-level is itself a workflow — the simplest possible workflow with a single ActionPair.

---

## 2. Workflow Artifacts

### Definition 25: Workflow Artifact

A **workflow artifact** is an artifact whose content encodes a complete workflow definition:

```
a_W = ⟨W, M_components, M_params⟩ ∈ A_W ⊂ A
```

Where:

- **W = ⟨{AP₁, ..., APₙ}, deps, goal⟩**: Workflow structure
  - `{AP₁, ..., APₙ}`: Set of ActionPair specifications
  - `deps: AP → P(AP)`: Dependency graph
  - `goal`: Terminal condition (all guards satisfied)
- **M_components**: Mapping from action_pair_id to component reference
- **M_params**: Configuration parameters (rmax, model, constraints)

### Schema (DS-PDDL JSON Encoding)

Workflow artifacts use the DS-PDDL format defined in the paper (Appendix: Workflow Specification Format). This PDDL-inspired format enables LLM-generatable workflow specifications:

```json
{
  "version": "1.0",
  "workflows": {
    "generated-workflow-001": {
      "name": "Authentication API",
      "specification": "Build a REST API for user authentication",
      "action_pairs": {
        "g_test": {
          "prompt": "Generate pytest tests for {specification}",
          "guard": "SyntaxGuard",
          "requires": []
        },
        "g_impl": {
          "prompt": "Implement code that passes {g_test}",
          "guard": "TestGuard",
          "requires": ["g_test"]
        }
      }
    }
  }
}
```

### Remark (DS-PDDL Alignment)

The workflow artifact schema matches the existing DS-PDDL format:

| DS-PDDL Field | Workflow Artifact Mapping |
|---------------|---------------------------|
| `specification` | Ψ_goal passed to planner |
| `action_pairs` | W.action_pairs |
| `requires` | deps function |
| `guard` | Component reference (resolved via C) |
| `prompt` | Generator template (supports {placeholders}) |

This alignment means:

1. Existing workflow.json files are valid workflow artifacts
2. Planner generates the same format humans author
3. Human amendments can modify generated workflows directly

### Axiom (Workflow Artifact Validity)

A workflow artifact is valid iff all components are resolvable and dependencies are well-formed:

```
Valid(a_W) ⟺ ∀ ap ∈ a_W.action_pairs:
  (1) ap.generator ∈ C_registry ∨ ∃r ∈ ℛ: r.type = "generator" ∧ r.id = ap.generator
  (2) ap.guard ∈ C_registry ∨ ∃r ∈ ℛ: r.type = "guard" ∧ r.id = ap.guard
  (3) ∀ req ∈ ap.requires: req ∈ dom(a_W.action_pairs)
```

---

## 3. Component Registry

### Definition 26: Component Registry

A **component registry** provides resolution from names to implementations:

```
C = C_static ∪ C_dynamic
```

Where:

- **C_static: N → Component**: Immutable, design-time registered components
- **C_dynamic: N → ℛ_ref**: Components discovered from repository at runtime

### Resolution Function

```
resolve: N × ℛ → Component ∪ {⊥}

resolve(n, ℛ) =
  C_static(n)                           if n ∈ dom(C_static)
  instantiate(E(ℛ, Φ_component(n)))     if n ∈ dom(C_dynamic)
  ⊥                                     otherwise
```

### Remark (Static Base, Dynamic Extensions)

The registry follows a layered model:

1. **Core components** (SyntaxGuard, ImportGuard, TestGuard) are statically registered
2. **Custom components** defined in previous workflows are discoverable via extraction
3. **Resolution priority**: static before dynamic (performance, determinism)

This enables extensibility without sacrificing startup reliability.

---

## 4. Planner ActionPair

### Definition 27: Planner ActionPair

A **Planner ActionPair** is a specialized ActionPair that generates workflow artifacts:

```
A_plan = ⟨ρ_plan, a_gen_plan, G_plan⟩
```

Where:

- **ρ_plan**: Precondition (typically ⊤ — planner runs first)
- **a_gen_plan: C → A_W**: Generator producing workflow artifacts
- **G_plan: A_W × C → (v, φ)**: Guard validating workflow well-formedness

### Remark (Planner Context)

The planner generator has access to:

| Context Component | What It Provides |
|-------------------|------------------|
| **Ψ_goal** | Goal specification (what to achieve) |
| **Ω** | Global constraints (what restrictions apply) |
| **ℛ** | Repository (what patterns/components are available) |
| **C_registry** | Available generators and guards |

The planner can query ℛ for:

- Successful workflows addressing similar goals (via E(ℛ, Φ_similar(Ψ_goal)))
- Available custom components (via E(ℛ, Φ_component_type("guard")))
- Patterns that succeeded previously (via E(ℛ, Φ_accepted ∧ Φ_action_pair))

---

## 5. Workflow Guard

### Definition 28: Workflow Guard

A **Workflow Guard** validates workflow artifact well-formedness:

```
G_W: A_W × C → (v, φ)

G_W(a_W, C) =
  (⊥, φ_structure)    if ¬WellFormed(a_W)
  (⊥, φ_deps)         if ¬DepsResolvable(a_W, C)
  (⊥, φ_components)   if ¬ComponentsAvailable(a_W, C)
  (⊤, ε)              otherwise
```

### Well-formedness Predicates

```
WellFormed(a_W) =
  (1) a_W.action_pairs ≠ ∅                      // Non-empty
  (2) DAG(a_W.requires)                         // Acyclic dependencies
  (3) ∀ ap: ap.requires ⊆ dom(a_W.action_pairs) // Closed dependencies

DepsResolvable(a_W, C) =
  ∀ ap ∈ a_W.action_pairs, req ∈ ap.requires:
    req ∈ E(C.ℛ, Φ_accepted(req)) ∨ req ∈ dom(a_W.action_pairs)

ComponentsAvailable(a_W, C) =
  ∀ ap ∈ a_W.action_pairs:
    resolve(ap.generator, C.ℛ) ≠ ⊥ ∧
    resolve(ap.guard, C.ℛ) ≠ ⊥
```

### Remark (Workflow Guard Composition)

G_W is naturally a CompositeGuard:

```python
G_W = CompositeGuard([
    DAGGuard(),           # WellFormed check
    DepsGuard(),          # DepsResolvable check
    ComponentGuard(C),    # ComponentsAvailable check
])
```

Each sub-guard provides specific feedback (φ) enabling targeted regeneration.

---

## 6. Meta-Workflow

### Definition 29: Meta-Workflow

The **Meta-Workflow** is the minimal, fixed workflow for generating workflows:

```
W_meta = ⟨{A_plan}, Φ_goal_plan, rmax_plan⟩
```

Where:

- **{A_plan}**: Single planner ActionPair
- **Φ_goal_plan**: Goal predicate (workflow artifact accepted by G_W)
- **rmax_plan**: Retry budget for planning

### Remark (The Only Static Workflow)

W_meta is the **only statically-defined workflow** in a fully dynamic system. All object-level workflows are generated. This creates a clean separation:

| Level | Workflow | Source | Purpose |
|-------|----------|--------|---------|
| Meta | W_meta | Static (code) | Generate object workflows |
| Object | W | Generated (artifact) | Accomplish goal |

### Remark (Bootstrap Problem)

W_meta solves the bootstrap problem: "how does the first workflow get generated?"

The answer: W_meta is hardcoded, containing exactly one ActionPair (the planner). This is the minimal structure needed to begin dynamic generation.

---

## 7. Two-Level Execution Model

### Definition 30: Two-Level Execution

The **Two-Level Execution Model** composes meta-level planning with object-level execution:

```
Execute₂L: (W_meta, Ψ_goal) → Result

Execute₂L(W_meta, Ψ_goal) =
  (1) r_W ← Execute(W_meta, Ψ_goal)           // Meta-level: generate workflow
  (2) if r_W.status ≠ ACCEPTED: return Fail(r_W)
  (3) W ← deserialize(r_W.a)                  // Extract workflow definition
  (4) store(r_W, ℛ) with W_ref = hash(W)      // Store with content address
  (5) return Execute(W, Ψ_goal)               // Object-level: execute workflow
```

### Remark (Specification Reuse)

The same Ψ_goal is passed to both levels:

- **Meta-level**: Planner uses Ψ_goal to decide what workflow to generate
- **Object-level**: Executors use Ψ_goal to generate artifacts

This ensures the generated workflow is appropriate for the goal.

---

## 8. Failure Mode Hierarchy

### Definition 31: Failure Mode Hierarchy

Failures occur at three distinct levels with different recovery semantics:

```
Failure = Fail_object | Fail_workflow | Fail_meta
```

| Level | Failure Type | Cause | Recovery |
|-------|--------------|-------|----------|
| Object | Fail_object(step, φ) | ActionPair exhausts rmax | Standard retry, then escalate |
| Workflow | Fail_workflow(W, φ) | Generated workflow inadequate | Policy-dependent |
| Meta | Fail_meta(φ) | Cannot generate valid workflow | Always escalate |

### Object-Level Failure

```
Fail_object(step, φ) → Standard retry within generated workflow
```

Handled identically to static workflows (Definition 8: Satisficing with Retry).

### Workflow-Level Failure

```
Fail_workflow(W, φ) → Policy-based: ESCALATE or REGENERATE
```

Occurs when:

- Multiple object-level failures suggest workflow structure is wrong
- A "workflow validation step" explicitly fails
- Human reviewer rejects the workflow

### Meta-Level Failure

```
Fail_meta(φ) → Always escalate to human
```

Occurs when:

- Planner cannot generate valid workflow after rmax_plan attempts
- Goal is likely unachievable with available components

---

## 9. Configurable Escalation Policy

### Definition 32: Configurable Escalation Policy

An **Escalation Policy** determines how workflow-level failures are handled:

```
Policy = ⟨mode, rmax_regen⟩

mode ∈ {ESCALATE, REGENERATE, HYBRID}
rmax_regen: ℕ   // Budget for workflow regeneration
```

### Policy Semantics

```
HandleFailure: (Failure, Policy, C) → Action

HandleFailure(Fail_object(step, φ), policy, C) =
  Retry(step, φ)  // Standard behavior

HandleFailure(Fail_workflow(W, φ), policy, C) =
  if policy.mode = ESCALATE:
    Escalate(φ, C)
  else if policy.mode = REGENERATE ∧ regen_budget > 0:
    Regenerate(W_meta, φ, C)
  else if policy.mode = HYBRID:
    if is_recoverable(φ): Regenerate(W_meta, φ, C)
    else: Escalate(φ, C)

HandleFailure(Fail_meta(φ), policy, C) =
  Escalate(φ, C)  // Always — cannot recover without human
```

### Remark (HYBRID Policy)

The HYBRID policy uses feedback to determine recoverability:

| Feedback Pattern | Interpretation | Action |
|------------------|----------------|--------|
| "Missing component X" | Structural issue | REGENERATE with different components |
| "Dependency cycle" | Structural issue | REGENERATE with different ordering |
| "Goal unachievable" | Fundamental issue | ESCALATE — human must revise goal |
| "Timeout exceeded" | Resource issue | ESCALATE — human must decide |

---

## 10. Theorems

### Theorem 11: Dynamics Preservation

**Statement**: Two-level execution preserves Definition 7 (System Dynamics) at both levels.

**Proof**:

1. **Meta-level**: W_meta is a standard workflow with A_plan as its single ActionPair.
   - Generation: a_W ~ a_gen_plan(C)
   - Sensing: ⟨v, φ⟩ = G_plan(a_W, C)
   - State Update: Based on v
   - This is Definition 7 applied to W_meta.

2. **Object-level**: The generated W is a standard workflow.
   - Each step executes Definition 7 independently.
   - No modifications to generation, sensing, or state update.

3. **Composition**: Execute₂L is sequential composition of Definition 7 executions.
   - No new state spaces, transitions, or dynamics are introduced.

Therefore, two-level execution is semantically equivalent to nested Dual-State dynamics.

∎

### Theorem 12: Workflow Artifact Integrity

**Statement**: A stored workflow artifact r_W with W_ref can always be resolved to its original workflow W.

**Proof**:

1. By Definition 11 (Workflow Reference), W_ref = hash(W).
2. By Definition 10 (Repository Item), r_W stores ⟨a_W, Ψ, Ω, W_ref, H, source, metadata⟩.
3. a_W encodes W (Definition 25).
4. By the Integrity Axiom (Definition 11): hash(resolve(W_ref)) = W_ref.
5. The deserialize function is deterministic: deserialize(serialize(W)) = W.
6. Therefore, for any r_W ∈ ℛ: deserialize(r_W.a) = W always.

∎

### Theorem 13: Component Resolution Completeness

**Statement**: If G_W(a_W, C) = ⊤, then all components in a_W are resolvable at execution time.

**Proof**:

1. G_W returns ⊤ only if ComponentsAvailable(a_W, C) holds (Definition 28).
2. ComponentsAvailable requires: ∀ ap ∈ a_W.action_pairs: resolve(ap.generator, ℛ) ≠ ⊥ ∧ resolve(ap.guard, ℛ) ≠ ⊥.
3. By Definition 26, resolution is deterministic given C and ℛ.
4. By Definition 2, ℛ is append-only — resolved components remain available.
5. Between G_W validation and Execute(W, Ψ_goal), no components are removed.
6. Therefore, Execute(W, Ψ_goal) can instantiate all ActionPairs.

∎

---

## 11. Relationship to HTN

Hierarchical Task Network (HTN) planning and Generated Workflows address related problems with different approaches:

| Aspect | HTN | Generated Workflows |
|--------|-----|---------------------|
| Decomposition | Runtime method selection | Generation-time workflow composition |
| Methods | Multiple alternatives per task | Single workflow per generation |
| Backtracking | Built-in search over decompositions | Policy-based (escalate/regenerate) |
| Primitives | Operators with preconditions | Registered ActionPairs with guards |
| Verification | Postconditions (may fail at runtime) | Guards (deterministic validation) |
| Learning | Limited (method heuristics) | Full training loop (Extension 04) |

### Remark (Compiled HTN Plans)

Generated Workflows can be understood as **compiled HTN plans**:

1. The Planner performs HTN-style goal decomposition
2. The output is a concrete workflow (no runtime method selection)
3. On failure, regeneration is equivalent to HTN backtracking

**Benefits over traditional HTN**:

- Workflow artifacts are verifiable before execution (G_W)
- Complete audit trail in ℛ
- Subject to the training loop (improve planner over time)
- Human can amend workflow (not just primitives)

### Remark (YAGNI Compliance)

Full HTN integration (runtime method selection, explicit backtracking) remains deferred per decisions.md. Generated Workflows provide the simpler case where goal-directed search happens **once, at planning time**, producing a concrete workflow.

---

## 12. Summary

1. **Workflow artifacts** encode complete workflow definitions (Definition 25)

2. **Component registry** provides static + dynamic component resolution (Definition 26)

3. **Planner ActionPair** generates workflow artifacts (Definition 27)

4. **Workflow Guard** validates well-formedness (Definition 28)

5. **Meta-Workflow** is the only static workflow — all others are generated (Definition 29)

6. **Two-Level Execution** composes meta and object levels (Definition 30)

7. **Failure hierarchy** distinguishes object, workflow, and meta failures (Definition 31)

8. **Escalation policy** configures failure recovery (Definition 32)

9. **Dynamics are preserved** — both levels follow Definition 7 (Theorem 11)

---

## 13. Future Work

- **Learned workflow templates**: Extract successful workflow patterns for reuse
- **Multi-step planning**: Generate workflow incrementally (act → observe → extend)
- **Workflow evolution**: Track W → W' refinement across generations
- **HTN integration**: Full runtime method selection when needed at scale

---

## See Also

- [01_versioned_environment.md](01_versioned_environment.md) — W_ref content addressing (Definition 11)
- [02_artifact_extraction.md](02_artifact_extraction.md) — Extraction function E(ℛ, Φ) (Definitions 17-18)
- [04_learning_loop.md](04_learning_loop.md) — Training planner on successful workflows
- [../agent_design_process/domain_definitions.md](../agent_design_process/domain_definitions.md) — Planning-Execution Separation (Section 2.2)
- [../decisions/decisions.md](../decisions/decisions.md) — HTN deferral rationale
