# Learning Loop

This document formally extends the Dual-State Framework to characterize training data extraction and policy updates, enabling continuous learning from workflow execution traces stored in ℛ.

> **Depends on**: [01_versioned_environment.md](01_versioned_environment.md) (repository items, Definitions 10-16), [02_artifact_extraction.md](02_artifact_extraction.md) (extraction, Definitions 17-18)
>
> **See also**: [00_notation_extensions.md](00_notation_extensions.md) (symbols), [05_learning_implementation.md](05_learning_implementation.md) (practical guide)

---

## Relationship to Paper

This extension formalizes Section 6.1 (Continuous Learning via The Optimization Loop) of the base paper. The paper proposes a four-tier optimization hierarchy:

1. **Tier 1 (Coach)**: Immediate semantic correction — *deferred to future work*
2. **Tier 2 (Sparse Reward)**: Guard verdicts as RL signal — *formalized here*
3. **Tier 3 (Dense Reward)**: Coach-provided shaping — *deferred to future work*
4. **Tier 4 (Policy Distillation)**: Fine-tuning from traces — *formalized here*

This extension provides the formal foundation for Tiers 2 and 4, using repository items (Definition 10) and extraction (Definition 17) as infrastructure.

---

## 1. Motivation

The standard execution model treats retries as computational waste. However, each refinement cycle produces valuable training signal:

- **Failed attempts** reveal what doesn't satisfy guards
- **Successful artifacts** demonstrate what does
- **Feedback history** captures the reasoning path from failure to success

The paper's key insight (Section 6.1.4):

> "The eventual success $a_{success}$ is treated as the target label, but the update is also conditioned on the Coach's feedback $\phi_{dense}$. This encourages the model not just to memorize the answer, but to internalize the *reasoning process* that led to it."

This extension formalizes how to extract training data from ℛ and update the generator policy π_θ.

---

## 2. Formal Definitions

### Definition 21: Refinement Predicate

A **Refinement Predicate** selects repository items that are part of retry→success chains:

```
Φ_refinement: r → {⊤, ⊥}

Φ_refinement(r) = ⊤ iff:
  r.status = ACCEPTED ∧
  ∃r' ∈ provenance(r) : r'.status = REJECTED
```

Where:

- **provenance(r)**: The chain of prior items leading to r (via provenance links in metadata)
- **ACCEPTED/REJECTED**: Guard verdict status stored in repository item

### Remark (First-Attempt Successes)

Items where `Φ_refinement(r) = ⊥` but `r.status = ACCEPTED` represent first-attempt successes. These may also be valuable for training (demonstrating ideal behavior) but lack the refinement signal. The choice to include them is a policy decision (see Definition 22).

---

### Definition 22: Training Trace

A **Training Trace** τ is the result of extraction with a training-specific filter:

```
τ = E(ℛ, Φ_training)
```

Where Φ_training is a policy choice. Common configurations:

| Policy | Filter | Use Case |
|--------|--------|----------|
| Refinements only | `Φ_refinement` | Learn from retry→success chains |
| All successes | `Φ_status(ACCEPTED)` | Include first-attempt successes |
| Generator only | `Φ_refinement ∧ Φ_source(GENERATOR)` | Exclude human artifacts |
| Human only | `Φ_refinement ∧ Φ_source(HUMAN)` | Preference alignment from human traces |
| High-value | `Φ_refinement ∧ Φ_retry_count(≥3)` | Focus on hard problems |
| Recent | `Φ_refinement ∧ Φ_timestamp(>cutoff)` | Avoid specification drift |

### Remark (Composability)

Filter predicates compose via conjunction (∧) and disjunction (∨), enabling fine-grained control over training data selection without modifying the extraction mechanism.

---

### Definition 23: Sparse Reward Signal

The **Sparse Reward Signal** derives directly from guard verdicts:

```
R_sparse: r → {-1, +1}

R_sparse(r) = +1 if r.status = ACCEPTED
              -1 if r.status = REJECTED
```

### Remark (Relationship to Paper)

This corresponds to Section 6.1.2 (Tier 2: Sparse Reward Signal):

```
R_sparse(s_env, G) = +1 if G(s_env) = ⊤
                     -1 if G(s_env) = ⊥
```

The repository item's status field stores this verdict, making R_sparse a simple lookup rather than re-evaluation.

### Remark (The Maze Isomorphism)

Per the paper: "The optimization loop effectively maps the high-dimensional, opaque manifold of the LLM onto a navigable, reward-driven maze." Guards are semantic walls; the sparse reward teaches the model to avoid them.

---

### Definition 24: Policy Update

The **Policy Update** fine-tunes the generator using training traces:

```
L(θ) = -E_τ[log π_θ(r.a | r.Ψ, r.H)]
```

Where:

- **π_θ**: Generator policy (model weights θ)
- **r.a**: Successful artifact (target)
- **r.Ψ**: Specification snapshot (prompt context)
- **r.H**: Feedback history (reasoning context)
- **E_τ**: Expectation over training traces

### Remark (Conditioning on Feedback)

The loss conditions on both specification (r.Ψ) and feedback history (r.H). This teaches the model not just *what* to produce, but *how to recover* from failures. Per Section 6.1.4:

> "This encourages the model not just to memorize the answer, but to internalize the *reasoning process* (the feedback) that led to it."

### Remark (Supervised Fine-Tuning)

Definition 24 is standard cross-entropy loss for supervised fine-tuning (SFT). For RL-based approaches using R_sparse directly, see Future Work.

---

## 3. Theorem 9: Training Trace Completeness

**Statement**: Repository items contain sufficient information for supervised fine-tuning without external state.

**Proof**:

1. **Prompt**: r.Ψ contains the specification snapshot at generation time (Definition 10)

2. **Completion**: r.a contains the artifact content (Definition 10)

3. **Reasoning context**: r.H contains feedback history (Definition 10)

4. **Refinement chain**: Provenance links in r.metadata connect failed attempts to eventual success

5. **No external state**: All fields are stored within the repository item itself. Context derivation (Definition 13) reconstructs full context from r alone.

Therefore, the tuple (r.Ψ, r.H, r.a) provides (prompt, reasoning, completion) — the standard format for instruction fine-tuning.

∎

**Corollary 9.1 (ℛ as Training Dataset)**: The repository ℛ is a self-contained training dataset. No external logging, telemetry, or state synchronization is required.

---

## 4. What the Model Learns

Training on `L(θ) = -E_τ[log π_θ(r.a | r.Ψ, r.H)]` teaches the joint distribution:

```
P(artifact | specification, feedback_history, codebase)
```

This manifests as learning:

| Learned Aspect | Source in Repository Item | Effect |
|----------------|---------------------------|--------|
| Specification style | r.Ψ patterns | Model learns "how we write specs here" |
| Codebase conventions | r.a patterns that achieve ACCEPTED | Model learns naming, structure, idioms |
| Guard preferences | Implicit in what achieves ACCEPTED | Model learns what satisfies validation |
| Refinement patterns | r.H → mapping from φ to corrections | Model learns to interpret and act on feedback |

### Remark (Specialization)

The trained model becomes a **specialist** for the codebase. It learns:

- The team's specification language
- The guards' acceptance criteria
- The codebase's conventions and patterns

This is a feature for dedicated deployment (the model "knows how we do things here") but reduces generalization to other codebases.

### Remark (RLHF Without the H)

This approach is effectively **RLHF where guards replace human feedback**:

- Guards provide deterministic, consistent preference signals
- No human labeling required (guards are the oracle)
- Preferences are auditable (guard implementations are code)

---

## 5. Potential Issues and Mitigations

| Issue | Severity | Description | Mitigation |
|-------|----------|-------------|------------|
| **Overfitting to codebase** | Medium | Model becomes less useful for other projects | Accept for dedicated deployment; or train on multiple codebases with codebase_id conditioning |
| **Specification drift** | Low | Ψ conventions change over time, old traces become stale | Weight recent traces higher via Φ_timestamp; periodic retraining |
| **Guard bias amplification** | Low | If guards have blind spots, model learns to exploit them | Guards are deterministic and auditable — fix the guard, retrain |
| **Human preference lock-in** | Low | If Φ_source(HUMAN) dominates, model mimics human style rigidly | Balance source distribution in Φ_training; use Φ_any_source |
| **Catastrophic forgetting** | Medium | Fine-tuning degrades general capabilities | Use LoRA/adapter approaches; keep base model frozen; see [05_learning_implementation.md](05_learning_implementation.md) |

### Remark (Guard Bias is Auditable)

Unlike human preference biases (which are opaque), guard biases are **inspectable code**. If the model learns to exploit a guard blind spot, the fix is clear: update the guard implementation and retrain on new traces.

---

## 6. Source Field as Policy Choice

### Remark (Human Generator Compatibility)

Definition 16 (Human Generator) stores `source = HUMAN` in repository items. This field enables training policy choices without structural changes:

```
Φ_generator_only(r) = r.source = GENERATOR
Φ_human_only(r) = r.source = HUMAN
Φ_any_source(r) = ⊤
```

### Remark (Human Traces as Preference Signal)

Human-provided artifacts may carry stronger signal for preference alignment:

- Humans apply judgment beyond guard criteria
- Human style preferences are captured implicitly
- Human corrections may be higher quality than generator refinements

The choice between Φ_generator_only, Φ_human_only, or Φ_any_source is a **policy decision**, not a structural constraint. The framework supports all options.

---

## 7. System Dynamics Preservation

### Theorem 10: Learning Loop Preserves System Dynamics

**Statement**: Training trace extraction and policy updates do not modify Definition 7 (System Dynamics).

**Proof**:

1. **Extraction is read-only**: By Theorem 3 (Extraction Invariance), E(ℛ, Φ_training) does not modify ℛ, S_workflow, or S_env.

2. **Policy update is external**: The loss computation L(θ) and gradient update occur outside the workflow execution loop. The updated π_θ' becomes the new generator for *future* workflows.

3. **No new state transitions**: Definition 7 dynamics (generate → sense → update) remain unchanged. The learning loop is orthogonal to execution.

∎

### Remark (Online vs Offline Learning)

- **Offline**: Extract τ from ℛ after workflow completion, train separately
- **Online**: Update π_θ during workflow execution (requires careful handling to avoid feedback loops)

This extension formalizes offline learning. Online learning is deferred to future work.

---

## 8. Summary

1. **Refinement Predicate (Definition 21)**: Selects retry→success chains from ℛ

2. **Training Trace (Definition 22)**: Extraction with configurable filter policy

3. **Sparse Reward (Definition 23)**: Guard verdict as reward signal (+1/-1)

4. **Policy Update (Definition 24)**: Cross-entropy loss conditioned on spec and feedback

5. **Theorem 9**: Repository items are self-contained training data

6. **Theorem 10**: Learning loop preserves system dynamics

---

## 9. Future Work

### Coach Formalization (Definitions 25+)

The paper's Tier 1 (Coach) and Tier 3 (Dense Reward) require formalizing:

```
a_coach: S_env × C × φ_guard → (ΔC, R_dense)
```

Where R_dense ∈ [0, 1] provides gradient signal even for failed attempts ("almost correct" vs "completely wrong").

### Multi-Codebase Training

Conditional model that adapts to codebase context:

```
π_θ(a | Ψ, H, codebase_id)
```

Enables a single model to serve multiple projects while maintaining specialization.

### Online Learning

Update π_θ during workflow execution:

```
After each ACCEPTED: θ' = θ - α∇L(θ)
```

Requires handling distribution shift and potential feedback loops.

### Reinforcement Learning Integration

Use R_sparse directly with policy gradient methods:

```
∇J(θ) = E_τ[R_sparse(r) · ∇log π_θ(r.a | r.Ψ)]
```

---

## See Also

- [01_versioned_environment.md](01_versioned_environment.md) — Repository items (Definitions 10-16)
- [02_artifact_extraction.md](02_artifact_extraction.md) — Extraction function (Definitions 17-18)
- [05_learning_implementation.md](05_learning_implementation.md) — Practical implementation guide with Unsloth
- [00_notation_extensions.md](00_notation_extensions.md) — Symbol reference
- Paper Section 6.1 — Continuous Learning via The Optimization Loop
