# Paper vs Implementation Scope

This document clarifies the relationship between the design documentation and the research paper (`docs/paper/main.tex`). The design docs represent the **current implementation scope**, while the paper includes the **full theoretical framework** and future work.

---

## Scope Summary

| Aspect | Design Docs | Paper |
|--------|-------------|-------|
| Guards | G₁–G₂₂ (22 guards) | G₁–G₂₉ (29+ guards) |
| SDLC Phases | 8 phases | 9 phases |
| Theoretical Results | Referenced | Formally proven |
| Learning Components | Not included | Coach, Critic, Policy Distillation |

---

## Deferred Guards (G₂₃–G₂₉)

The following guards are defined in the paper but **not yet implemented**:

### Phase 8: Version Control Safety (Paper Only)

| Guard | Transition | Purpose |
|-------|------------|---------|
| G₂₃ | STAGED → COMMIT_READY | Pre-commit hook validation |

### Bootstrap/Legacy System Guards (Paper Appendix F)

| Guard | Transition | Purpose |
|-------|------------|---------|
| G₂₄ | LEGACY → ANALYZED | Static analysis of legacy codebase |
| G₂₅ | ANALYZED → DEPENDENCY_MAPPED | Dependency graph extraction |
| G₂₆ | DEPENDENCY_MAPPED → CHARACTERIZED | Characterization test generation |
| G₂₇ | CHARACTERIZED → COVERAGE_SUFFICIENT | Coverage threshold validation |
| G₂₈ | COVERAGE_SUFFICIENT → GAPS_IDENTIFIED | Test gap analysis |
| G₂₉ | CHARACTERIZED → REFACTOR_SAFE | Safe refactoring validation |

**Rationale**: Bootstrap guards are designed for brownfield/legacy system integration. The current implementation focuses on greenfield workflows.

---

## Deferred Theoretical Content

### Formal Results (Paper Sections 5–6)

The paper contains formal propositions not reproduced in design docs:

- **Proposition 1**: Workflow State Projection — Guards define deterministic projection Γ: S_env → S_workflow
- **Proposition 4**: Asymptotic Soundness — P(fail) ≤ (1-ε)^R_max
- **Corollary 1**: Reliability Bound — R_max ≥ ln(1 - δ^(1/K)) / ln(1 - ε)
- **Corollary 2**: Complexity Bound — O(|S_reach| × R_max × |G|)
- **Algorithm 1**: EXECUTE-PLAN pseudocode

These results provide formal guarantees. The implementation follows them but doesn't reproduce the proofs.

### Human Oversight Notation

The paper uses `†` (dagger) notation to mark guards requiring human approval (G₁†, G₄†, G₆†). This notation is implied but not explicit in the design docs.

---

## Future Work Components (Paper Section 8)

The paper describes a continuous learning architecture not yet implemented:

| Component | Paper Section | Description |
|-----------|---------------|-------------|
| Coach (a_coach) | 8.1.1 | LLM-as-judge for semantic feedback |
| Critic | 8.1.2 | Sparse reward signals from guard validation |
| Reward Shaping | 8.1.3 | Dense reward for intermediate progress |
| Policy Distillation | 8.1.4 | Fine-tuning from successful traces |
| Meta-Policy Optimization | 8.2 | Dynamic guard selection |
| GuardGym | 8.3 | Control-oriented benchmark suite |
| Multi-Agent Shared Truth | 8.4 | Deterministic consensus via guards |

**Rationale**: These represent research directions. The current implementation provides the foundational architecture they build upon.

---

## Experimental Results

The paper includes experimental validation (Section 7) not in design docs:

- 13 LLMs tested (1.3B–15B parameters)
- 3 diagnostic probes (LRU Cache, Template Engine, Password Validator)
- Reliability improvements up to +66 percentage points
- TDD workflow benchmark results

---

## Reference

For the complete theoretical treatment, see:

- `docs/paper/main.tex` — Full paper with proofs and experiments
- Appendix C — Complete guard catalog
- Appendix F — Bootstrap/legacy system workflow
