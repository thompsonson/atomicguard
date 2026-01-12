# ADR: Generator Optimization Strategies

## Status

Proposed

## Context

AtomicGuard's learning loop (Extensions 04-05) extracts training traces from the repository and uses them to improve generator performance. This ADR documents the landscape of optimization strategies and recommends an implementation approach.

The goal is to reduce E[retries] → 0, meaning generators learn to produce guard-accepted artifacts on the first attempt.

---

## Optimization Strategies

### Layer 1: Supervised Fine-Tuning (SFT)

Train on successful (specification, artifact) pairs extracted from the repository.

| Aspect | Details |
|--------|---------|
| **What** | Imitation learning on accepted artifacts |
| **AtomicGuard mapping** | Extension 05: τ = E(ℛ, Φ_training) |
| **Reward signal** | Implicit (only train on ACCEPTED artifacts) |
| **Pros** | Simple, stable, well-understood |
| **Cons** | No learning from failures, offline only |

**Papers**: Standard SFT literature

**Infrastructure**:

- [Unsloth](https://github.com/unslothai/unsloth) - 2x faster, 4-bit quantization, LoRA
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - Config-driven fine-tuning
- [torchtune](https://github.com/pytorch/torchtune) - PyTorch native

---

### Layer 2: Reinforcement Learning (GRPO/mmGRPO)

Policy gradient optimization using guard verdicts as explicit reward signals.

| Aspect | Details |
|--------|---------|
| **What** | Online RL with guard-based rewards |
| **AtomicGuard mapping** | Guard.validate() → R_sparse (Definition 23) |
| **Reward signal** | Explicit: +1 (ACCEPTED), -1 (REJECTED) |
| **Pros** | Learns from failures, online exploration |
| **Cons** | More complex, requires many rollouts |

**Key insight**: mmGRPO extends GRPO to multi-module programs by grouping LM calls by module (ActionPair in AtomicGuard).

**Papers**:

- [Multi-module GRPO: Composing Policy Gradients and Prompt Optimization](https://arxiv.org/abs/2508.04660) - Ziems et al., 2025

**Infrastructure**:

- [Arbor](https://github.com/ziems/arbor) - RL framework for DSPy programs (reference mmGRPO implementation)
- [TRL](https://github.com/huggingface/trl) - Hugging Face RL library

---

### Layer 3: Prompt Optimization (MIPROv2)

Search for better prompt templates and instructions without changing model weights.

| Aspect | Details |
|--------|---------|
| **What** | Gradient-free search over prompt space |
| **AtomicGuard mapping** | Generator.template optimization |
| **Reward signal** | Guard pass rate on evaluation set |
| **Pros** | No training required, interpretable changes |
| **Cons** | Limited by prompt expressiveness |

**Key insight**: MIPROv2 bootstraps demonstrations and searches for optimal instructions using LLM-generated candidates.

**Papers**:

- [DSPy: Compiling Declarative Language Model Calls](https://arxiv.org/abs/2310.03714) - Khattab et al., 2023
- [MIPROv2 Documentation](https://dspy.ai/learn/optimization/optimizers/)

**Infrastructure**:

- [DSPy](https://github.com/stanfordnlp/dspy) - Framework for programming LMs

---

### Layer 4: Reflective Evolution (GEPA)

Use LLM reflection on execution traces to evolve prompts, maintaining a Pareto frontier of candidates.

| Aspect | Details |
|--------|---------|
| **What** | Evolutionary search with natural language reflection |
| **AtomicGuard mapping** | Guard.feedback → reflection input |
| **Reward signal** | Natural language analysis of failures |
| **Pros** | 35x fewer rollouts than GRPO, uses rich feedback |
| **Cons** | Requires LLM calls for reflection |

**Key insight**: GEPA outperforms GRPO by 10-20% while using 35x fewer rollouts because natural language reflection provides richer learning signal than scalar rewards.

**Papers**:

- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) - Agrawal et al., 2025

**Infrastructure**:

- [DSPy GEPA](https://dspy.ai/api/optimizers/GEPA/overview/) - Integrated into DSPy
- [gepa-ai](https://github.com/gepa-ai/gepa) - Standalone package

---

### Composition: BetterTogether

Alternate between weight optimization (SFT/GRPO) and prompt optimization (MIPROv2).

| Aspect | Details |
|--------|---------|
| **What** | Iterative weight + prompt optimization |
| **Key insight** | "Get the same LM to teach itself" |
| **Results** | +60% over fine-tuning alone, +6% over prompts alone |

**Algorithm**:

```
for round in range(num_rounds):
    # Phase A: Optimize weights
    traces = extract(repository, Φ_training)
    model = finetune(model, traces)

    # Phase B: Optimize prompts for the new model
    for action_pair in workflow:
        action_pair.template = optimize_prompt(
            action_pair.template,
            model,
            metric=guard_pass_rate
        )
```

**Papers**:

- [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930) - Soylu et al., EMNLP 2024
- [ACL Anthology](https://aclanthology.org/2024.emnlp-main.597/)

---

### Inference-Time: Self-Critique with Repository Context

Pre-submission optimization that catches errors before guard evaluation, using repository knowledge to inform refinement.

| Aspect | Details |
|--------|---------|
| **What** | Generator critiques its own artifact before submission |
| **AtomicGuard mapping** | Insert between Generator.generate() and Guard.validate() |
| **Reward signal** | None (inference-time, no training) |
| **Pros** | Reduces E[retries], no training required, immediate benefit |
| **Cons** | Adds latency per generation, extra LLM call |

**Key insight**: Self-critique becomes significantly more effective when augmented with repository context:

| Context Source | What It Provides | Self-Critique Use |
|----------------|------------------|-------------------|
| **Similar accepted artifacts** | Patterns that pass guards | "Does my artifact follow these patterns?" |
| **Previous rejection feedback** | Common failure modes | "Am I making a known mistake?" |
| **Guard feedback history** | What guards check for | "Will this pass the syntax/test/arch guard?" |
| **Retry chains** | Progression from failure → success | "What corrections were needed before?" |

**Algorithm**:

```
def generate_with_critique(spec, context, repository):
    # Step 1: Generate initial artifact
    artifact_draft = generator.generate(spec, context)

    # Step 2: Retrieve relevant repository context
    similar_accepted = repository.query(
        Φ_similar(spec) ∧ Φ_accepted
    )
    recent_rejections = repository.query(
        Φ_same_guard ∧ Φ_rejected, limit=5
    )

    # Step 3: Self-critique with context
    critique_prompt = f"""
    Review this artifact for potential issues:

    ARTIFACT: {artifact_draft}

    SIMILAR ACCEPTED EXAMPLES:
    {similar_accepted}

    COMMON REJECTION REASONS:
    {[r.feedback for r in recent_rejections]}

    Identify issues and provide corrected version.
    """

    refined_artifact = generator.critique(critique_prompt)

    # Step 4: Submit to guard
    return guard.validate(refined_artifact)
```

**Relationship to offline optimization**:

| Offline Strategy | Self-Critique Role |
|------------------|-------------------|
| SFT | Critique catches errors SFT didn't prevent |
| MIPROv2 | Critique provides runtime flexibility beyond fixed prompts |
| GRPO/GEPA | Critique is like a single-step RL episode at inference |

**Papers**:

- [Enhancing LLM Planning Capabilities through Intrinsic Self-Critique](https://arxiv.org/abs/2512.24103) - Google DeepMind, 2024

**Implementation priority**: Can be implemented independently of Phases 1-4. Low infrastructure cost, immediate benefit. Consider implementing alongside or before Phase 1 SFT.

---

### Inference-Time: LLM-as-a-Judge (Coach)

Use a separate LLM to evaluate artifacts and provide coaching feedback before guard submission. Unlike self-critique (same model reviewing itself), LLM-as-a-Judge uses an external evaluator that can provide more objective assessment.

| Aspect | Details |
|--------|---------|
| **What** | External LLM evaluates artifact quality and provides feedback |
| **AtomicGuard mapping** | Soft pre-guard or coach for self-critique loop |
| **Reward signal** | Structured evaluation scores + natural language feedback |
| **Pros** | More objective than self-critique, richer feedback than guards |
| **Cons** | Known biases (position, verbosity, self-enhancement), extra LLM call |

**Key insight**: LLM-as-a-Judge fills the gap between deterministic guards and self-critique:

| Evaluator | Type | Feedback Quality | Objectivity |
|-----------|------|------------------|-------------|
| **Guard** | Deterministic | Binary + specific | High (but narrow) |
| **Self-Critique** | Same model | Rich but biased | Low (self-serving) |
| **LLM-as-a-Judge** | External model | Rich and structured | Medium (known biases) |

**AtomicGuard integration patterns**:

```
# Pattern A: Coach for Self-Critique
artifact_draft = generator.generate(spec)
judge_feedback = judge.evaluate(artifact_draft, spec)
refined = generator.refine(artifact_draft, judge_feedback)
result = guard.validate(refined)

# Pattern B: Soft Pre-Guard Filter
artifact = generator.generate(spec)
judge_score = judge.evaluate(artifact, spec)
if judge_score.confidence > threshold:
    result = guard.validate(artifact)  # Skip expensive guard if low confidence
else:
    # Regenerate or refine first
    ...

# Pattern C: Training Signal Generation (DPO)
# Judge compares artifact pairs to generate preference data
preferred, rejected = judge.compare(artifact_a, artifact_b, spec)
training_data.append((spec, preferred, rejected))  # For DPO training
```

**Known biases to manage** (from LLM-as-a-Judge literature):

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Position bias** | Prefers first/last option in comparisons | Randomize order, average across permutations |
| **Verbosity bias** | Prefers longer responses | Normalize for length, penalize padding |
| **Self-enhancement** | Prefers outputs similar to judge's own style | Use different model family for judge |
| **Sycophancy** | Agrees with user preferences | Blind evaluation without user context |

**Open challenges** (per [Thompson & Theerthala, 2025](https://matt.thompson.gr/2025/10/03/discussing-the-state-of-llmasajudge.html)):

1. **Domain-specific evaluation**: Need judge calibrated for code/artifact quality, not general text
2. **Comparison with alternatives**: How does LLM-as-a-Judge compare to guard-based evaluation?
3. **Reasoning-based approaches**: Chain-of-thought judging and majority voting underexplored

**Papers**:

- [A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594) - Comprehensive methodology, Nov 2024
- [From Generation to Judgment](https://arxiv.org/abs/2411.16594) - What/how/benchmark taxonomy, Nov 2024
- [Agent-as-a-Judge](https://arxiv.org/abs/2508.02994) - Agent evaluating agent chains, 2025
- [Judging LLM-as-a-Judge with MT-Bench](https://arxiv.org/abs/2306.05685) - Foundational benchmarks, 2023

**Implementation priority**: Medium. More valuable after Self-Critique is implemented. Consider for:

- Coaching feedback in retry loops
- Pre-filtering before expensive guards (TestGuard, ArchGuard)
- Generating preference pairs for future DPO training

---

### Inference-Time: Self-Consistency (Majority Voting)

Generate multiple candidate artifacts and select the most consistent answer through majority voting. Rather than relying on a single generation, this approach leverages the insight that correct solutions tend to converge across multiple reasoning paths.

| Aspect | Details |
|--------|---------|
| **What** | Generate N candidates, select most frequent/consistent answer |
| **AtomicGuard mapping** | Multiple Generator.generate() calls → vote → Guard.validate() |
| **Reward signal** | None (inference-time ensemble) |
| **Pros** | +10-20% accuracy on complex reasoning, no training required |
| **Cons** | N× inference cost, requires answer equivalence detection |

**Key insight**: "A complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer." When generators produce the same artifact through different reasoning paths, confidence in correctness increases.

**AtomicGuard integration patterns**:

```
# Pattern A: Simple Majority Voting
def generate_with_consistency(spec, context, n=5):
    candidates = [generator.generate(spec, context) for _ in range(n)]

    # Group by semantic equivalence (not string equality)
    groups = cluster_by_equivalence(candidates)

    # Select most frequent
    best_group = max(groups, key=len)
    return best_group[0]  # Representative from largest cluster

# Pattern B: Weighted Voting with Guard Pre-check
def generate_with_weighted_vote(spec, context, n=5):
    candidates = []
    for _ in range(n):
        artifact = generator.generate(spec, context)
        # Quick syntax check as weight
        syntax_ok = syntax_guard.validate(artifact).passed
        candidates.append((artifact, 1.0 if syntax_ok else 0.5))

    # Weighted vote
    return weighted_majority(candidates)

# Pattern C: MAKER-style Microagent Decomposition
def generate_with_decomposition(spec, context):
    subtasks = decompose(spec)  # Break into micro-tasks
    results = {}

    for subtask in subtasks:
        # Multiple microagents vote on each subtask
        votes = [microagent.solve(subtask) for _ in range(3)]
        results[subtask.id] = majority_vote(votes)

    return compose(results)  # Reconstruct full artifact
```

**Equivalence detection challenges**:

| Artifact Type | Equivalence Method |
|---------------|-------------------|
| Code | AST comparison (ignoring whitespace/comments) |
| Tests | Semantic equivalence of assertions |
| Config | Normalized key-value comparison |
| Text | Embedding similarity threshold |

**Relationship to other inference-time techniques**:

| Technique | When to Use | Combine With |
|-----------|-------------|--------------|
| **Self-Consistency/MAKER** | High-stakes, complex specs | Can vote on self-critiqued outputs |
| **Self-Critique** | Single generation refinement | Critique before voting |
| **LLM-as-a-Judge** | Quality evaluation | Judge can break ties |

**Voting algorithms**: Self-Consistency and MAKER share the same flow (generate N → vote → validate). The difference is the voting algorithm:

| Algorithm | Selection Method |
|-----------|------------------|
| **Majority Vote** | Most frequent answer |
| **Weighted Vote** | Frequency × confidence score |
| **MAKER Vote** | First to K votes ahead of next closest |
| **Judge-Assisted** | LLM-as-a-Judge breaks ties |

**AtomicGuard voting pattern** (works with any algorithm):

```
def generate_with_voting(action_pair, spec, context, n=3, vote_fn=majority_vote):
    # Step 1: Generate N candidates
    candidates = [
        action_pair.generator.generate(spec, context)
        for _ in range(n)
    ]

    # Step 2: Vote (algorithm is pluggable)
    best = vote_fn(candidates)

    # Step 3: Guard validates consensus
    return action_pair.guard.validate(best)
```

**MAKER's key insight for AtomicGuard**: The "first to K votes ahead" algorithm enables early termination—once a candidate leads by K votes, generation stops without exhausting all N samples. Since AtomicGuard workflows are already decomposed into ActionPairs, this voting happens per-step. Errors are caught and corrected locally rather than propagating through the entire workflow. This is particularly valuable for long workflows where a single bad generation early on would cascade, and the early-termination property reduces inference cost when consensus is reached quickly.

**Papers**:

- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al., 2022 (original SC-CoT)
- [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/abs/2511.09030) - Task decomposition + voting, 2025
- [Building a Self-Consistency Agent](https://matt.thompson.gr/2025/06/26/ia-series-n-building-a.html) - Practical implementation guide

**Implementation priority**: Medium-Low. Higher cost but valuable for:

- High-stakes artifact generation (security-critical code)
- Complex specifications with multiple valid solutions
- When single-generation pass rate is low but correct solutions exist

---

## Training Infrastructure Summary

| Tool | Purpose | AtomicGuard Use Case |
|------|---------|----------------------|
| [Unsloth](https://github.com/unslothai/unsloth) | Efficient LoRA/QLoRA fine-tuning | Extension 05 SFT |
| [Arbor](https://github.com/ziems/arbor) | RL for LLM programs (mmGRPO) | Future GRPO integration |
| [TRL](https://github.com/huggingface/trl) | RL training library | Alternative to Arbor |
| [DSPy](https://github.com/stanfordnlp/dspy) | Prompt optimization framework | MIPROv2, GEPA |

---

## Mapping to AtomicGuard Components

| AtomicGuard Component | Optimization Role |
|-----------------------|-------------------|
| Repository (ℛ) | Training data source - stores all artifacts |
| Guard.validate() | Reward signal - provides R_sparse (Definition 23) |
| Guard.feedback | Reflection input - rich signal for GEPA |
| Extraction E(ℛ, Φ) | Trace selection - filters training data |
| Generator.template | Prompt optimization target (MIPROv2) |
| Generator weights | Weight optimization target (SFT/GRPO) |
| ActionPair | Module boundary for mmGRPO grouping |

---

## Conceptual Framework: Situational Leadership

The phased approach maps naturally to Hersey & Blanchard's Situational Leadership model, where optimization intensity matches generator "development level":

| Development Level | Generator State | Leadership Style | Optimization Approach |
|-------------------|-----------------|------------------|----------------------|
| **D1** (Low competence, high commitment) | Fresh/untrained | **Directing** (S1) | SFT - explicit demonstration of correct behavior |
| **D2** (Some competence, low commitment) | Partially trained, inconsistent | **Coaching** (S2) | MIPROv2 - structured guidance via prompt search |
| **D3** (High competence, variable commitment) | Good but plateaued | **Supporting** (S3) | BetterTogether - iterative refinement |
| **D4** (High competence, high commitment) | Mature, reliable | **Delegating** (S4) | GEPA/GRPO - autonomous exploration |

**Key insight**: Just as leaders adapt their style to follower readiness, AtomicGuard adapts optimization strategy to generator capability. Early-stage generators need directive approaches (SFT shows exactly what to produce), while mature generators benefit from exploratory approaches (GRPO/GEPA discover novel solutions).

This framing supports the phased implementation: start with high-structure approaches (SFT), gradually reduce structure as generators demonstrate competence (prompt optimization), and eventually enable autonomous exploration (RL) for generators that have plateaued.

---

## Success Metrics

| Phase | Primary Metric | Target | Secondary Metrics |
|-------|---------------|--------|-------------------|
| **Phase 1: SFT** | Guard pass rate | ≥80% first attempt | Training loss convergence, Eval loss stability |
| **Phase 2: Prompt Opt** | Guard pass rate delta | +5-10% over SFT baseline | Prompt length, Inference latency |
| **Phase 3: BetterTogether** | Cumulative improvement | +15-20% over SFT alone | Rounds to convergence, Compute cost per round |
| **Phase 4: GRPO/GEPA** | Exploration success rate | Novel solutions found | Rollouts per improvement, Reflection quality |

**Universal metrics** (all phases):

- **E[retries]**: Expected number of guard rejections before acceptance (goal: → 0)
- **Time-to-acceptance**: Wall-clock time from specification to accepted artifact
- **Generalization**: Performance on held-out specifications not seen during optimization

---

## Cost/Benefit Trade-offs

| Strategy | Compute Cost | Data Requirements | Implementation Complexity | Expected Lift |
|----------|--------------|-------------------|---------------------------|---------------|
| **Self-Critique** | Low | Repository access | Low | +5-15% (pre-guard) |
| **LLM-as-a-Judge** | Low-Medium | None (inference-time) | Low-Medium | +5-10% (coaching) |
| **Self-Consistency** | Medium-High | None (N× generations) | Medium | +10-20% (complex specs) |
| **SFT** | Low-Medium | 100-1000 traces | Low | Baseline |
| **MIPROv2** | Medium | 50-200 eval examples | Low | +5-10% |
| **BetterTogether** | Medium-High | Same as SFT + MIPROv2 | Medium | +15-20% |
| **GRPO** | High | Online rollouts | High | +10-25% |
| **GEPA** | Medium | 35x fewer than GRPO | Medium | +10-20% |

**Cost drivers**:

- Self-Critique: One additional LLM call per generation + repository query latency
- LLM-as-a-Judge: One judge LLM call per evaluation (can use smaller/cheaper model)
- Self-Consistency: N× generation cost (typically N=5-10) + equivalence computation
- SFT: GPU hours for fine-tuning, LoRA reduces this significantly
- MIPROv2: LLM API calls for candidate generation and evaluation
- BetterTogether: Multiple SFT rounds × prompt optimization rounds
- GRPO: Many rollouts (1000s) through guard evaluation
- GEPA: LLM calls for reflection, but 35x fewer rollouts than GRPO

**Break-even analysis**: Given AtomicGuard's guard infrastructure already exists, the marginal cost of optimization is primarily compute. If a 10% improvement in E[retries] saves 100 guard invocations per day, prompt optimization pays for itself within weeks.

---

## Phase Transition Criteria

### Phase 1 → Phase 2 (SFT → Prompt Optimization)

**Trigger**: Move to prompt optimization when:

- [ ] SFT training has converged (validation loss stable for 3+ epochs)
- [ ] Guard pass rate ≥ 70% on held-out test set
- [ ] At least 500 training traces in repository
- [ ] Baseline metrics established for comparison

### Phase 2 → Phase 3 (Prompt Opt → BetterTogether)

**Trigger**: Move to BetterTogether when:

- [ ] Prompt optimization has converged (no improvement in 10+ iterations)
- [ ] Combined SFT + prompt optimization yields ≥ 80% pass rate
- [ ] Sufficient compute budget for iterative rounds
- [ ] Clear evidence that prompts optimized for base model underperform on fine-tuned model

### Phase 3 → Phase 4 (BetterTogether → GRPO/GEPA)

**Trigger**: Consider RL/reflection when:

- [ ] BetterTogether has plateaued (< 1% improvement over 2 rounds)
- [ ] Analysis shows failures are exploration problems (not fitting problems)
- [ ] Guard.feedback provides actionable natural language (favors GEPA)
- [ ] Budget for extensive rollouts exists (favors GRPO if yes, GEPA if no)

### Regression Criteria

**Fall back to simpler approach if**:

- Performance degrades on held-out test set
- Optimization introduces instability (high variance in pass rate)
- Compute costs exceed allocated budget
- Time-to-acceptance increases despite higher pass rate

---

## Decision

### Phase 1: SFT (Extension 05) — CURRENT PLAN

Implement supervised fine-tuning on extracted traces using Unsloth.

**Rationale**:

- Simplest and most stable approach
- Provides foundation for all other strategies
- Already specified in Extension 05
- LoRA prevents catastrophic forgetting

### Phase 2: Prompt Optimization — FUTURE

Add MIPROv2-style search over generator templates.

**Rationale**:

- Guards provide natural evaluation metric (pass rate)
- No additional training infrastructure required
- Complements SFT (optimizes prompts for fine-tuned model)

**Prerequisites**: Extension 05 complete

### Phase 3: BetterTogether — FUTURE

Alternate SFT and prompt optimization in iterative rounds.

**Rationale**:

- Proven to outperform either approach alone
- Guards serve as consistent metric across rounds
- Builds on Phases 1 and 2

**Prerequisites**: Phases 1 and 2 complete

### Phase 4: GRPO/GEPA — FUTURE (OPTIONAL)

Consider if SFT + prompt optimization plateau.

**Options**:

- **GRPO**: Online RL using guard rewards - more exploration
- **GEPA**: Reflective evolution using guard feedback - fewer rollouts

**Rationale**:

- Add complexity only if simpler approaches insufficient
- GEPA particularly promising due to guard.feedback alignment

**Prerequisites**: Phases 1-3 evaluated

---

## Consequences

### Positive

- Clear implementation roadmap with increasing complexity
- Each phase builds on previous work
- Guards serve as universal reward signal across all strategies
- No formal framework changes required (pure implementation)
- Repository provides training data for all approaches

### Negative

- Full BetterTogether requires multiple optimization rounds (compute cost)
- GRPO/Arbor adds significant infrastructure complexity
- Prompt optimization search can be expensive (many LLM calls)
- Evaluation requires held-out test set to avoid overfitting

---

## References

### Papers

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| DSPy: Compiling Declarative Language Model Calls | Khattab et al. | 2023 | [arXiv](https://arxiv.org/abs/2310.03714) |
| BetterTogether: Fine-Tuning and Prompt Optimization | Soylu et al. | 2024 | [arXiv](https://arxiv.org/abs/2407.10930), [ACL](https://aclanthology.org/2024.emnlp-main.597/) |
| mmGRPO: Multi-module Policy Gradients | Ziems et al. | 2025 | [arXiv](https://arxiv.org/abs/2508.04660) |
| GEPA: Reflective Prompt Evolution | Agrawal et al. | 2025 | [arXiv](https://arxiv.org/abs/2507.19457) |
| Intrinsic Self-Critique for LLM Planning | Google DeepMind | 2024 | [arXiv](https://arxiv.org/abs/2512.24103) |
| A Survey on LLM-as-a-Judge | Various | 2024 | [arXiv](https://arxiv.org/abs/2411.15594) |
| From Generation to Judgment | Various | 2024 | [arXiv](https://arxiv.org/abs/2411.16594) |
| Agent-as-a-Judge | Various | 2025 | [arXiv](https://arxiv.org/abs/2508.02994) |
| Judging LLM-as-a-Judge with MT-Bench | Zheng et al. | 2023 | [arXiv](https://arxiv.org/abs/2306.05685) |
| Self-Consistency Improves Chain of Thought | Wang et al. | 2022 | [arXiv](https://arxiv.org/abs/2203.11171) |
| MAKER: Multi-Agent Voting for Reliable LLM Tasks | Various | 2025 | [arXiv](https://arxiv.org/abs/2511.09030) |

### Code Repositories

| Repository | Purpose | Link |
|------------|---------|------|
| DSPy | LLM programming framework | <https://github.com/stanfordnlp/dspy> |
| Arbor | RL for DSPy programs | <https://github.com/ziems/arbor> |
| GEPA | Reflective prompt evolution | <https://github.com/gepa-ai/gepa> |
| Unsloth | Efficient fine-tuning | <https://github.com/unslothai/unsloth> |
| TRL | RL training library | <https://github.com/huggingface/trl> |

### Documentation

- DSPy Optimizers: <https://dspy.ai/learn/optimization/optimizers/>
- GEPA API: <https://dspy.ai/api/optimizers/GEPA/overview/>
- Unsloth Documentation: <https://docs.unsloth.ai/>
