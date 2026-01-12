# Dual-State Agent: Decision Log

## Deferred (YAGNI)

| Decision | Rationale |
|----------|-----------|
| BYRD Box mapping | Semantics implicit in `DualStateAgent.execute()` try/succeed/retry/fail loop |
| Builder pattern | Current constructor with list is explicit and testable; add if workflow complexity grows |
| DS-PDDL implementation | Paper contribution (Appendix); Python implementation proves semantics; DSL is syntax concern |
| HTN integration | Needed for parallelism/hierarchy but overkill for PoC sequential workflows |

## Implemented

| Decision | Choice | Alternative Considered |
|----------|--------|------------------------|
| Fatal escalation | `GuardResult.fatal` + `EscalationRequired` exception | Simple retry-or-fail binary |
| Workflow outcome | `WorkflowStatus` enum (SUCCESS, FAILED, ESCALATION) | Boolean `success` field |
| Generator opacity | `agen` may be semantic agent (ReAct, CoT); workflow sees only final artifact | Expose internal reasoning to workflow |
| ActionPair structure | ⟨agen, G⟩ pure executor | Included ρ (precondition) |
| WorkflowState ownership | Workflow owns, Agent stateless | Agent owns state |
| Precondition handling | Inferred from `requires` | Explicit lambda |
| Guard inputs | Scoped dependencies (Option A) | Full context access (Option B) |
| Artifact storage | Interface + Implementation | Direct GitPython usage |
| Workflow orchestration | `add_step()` with requires | Constructor with list |
| Failure semantics | Return provenance, no rollback | Rollback on failure |
| Context naming | `Cambient` (supports cousins) | `Cparent` (implies lineage) |
| Guard signature | `G(artifact, **dependencies)` | `G(artifact)` or `G(artifact, context)` |
| Prompt management | `PromptTemplate` dataclass | String concatenation |
| Evaluation order | Sequential O(n) | Tree/HTN O(log n) |
| ADD placement | Example (`examples/checkpoint/04_sdlc/generators/add.py`) | Core package |
| PydanticAI for structured output | Add as dependency | Manual JSON parsing |
| PydanticAI retries | Disabled (`retries=0`) | Merge with AtomicGuard rmax |
| pytestarch API validation | Whitelist-only (no blocklist) | Blocklist of known bad patterns |

## Paper Changes Required

| Section | Change | Status (5 Jan 2025) |
|---------|--------|---------------------|
| Definition 3 | Rename Cparent → Cambient | ✅ Done (paper uses `𝓔` Ambient Environment) |
| Definition 6 | Extend G : A × C → {⊥, ⊤} × Σ* | ✅ Done (line 410: tri-state + feedback) |
| Definition 7.2 | Update ⟨v, φ⟩ = G(a', C) | ✅ Done (line 446) |
| Lemma 1 | Extend determinism to (a, C) | ✅ Done (line 551: `G(a, C)`) |
| New Remark | Guard Input Scoping (minimal required inputs) | ✅ Done (lines 414-416) |

## Future Considerations

| Topic | Notes |
|-------|-------|
| DS-PDDL | Language-agnostic, non-coder authoring, LLM-generatable specs |
| Builder pattern | Fluent API if workflows become complex |
| HTN/Tree | O(log n) precondition eval; needed at scale (100s guards, multi-agent) |
| Guard library | DDD guards (ACL integrity, architecture fitness, DI container) |
| Parallel guard | ThreadPoolExecutor for independent checks |
| pytestarch error hints | Add common hallucination→fix mappings if needed - Currently whitelist-only; no hints |
| Generated Workflows | Extension 06: Planner ActionPair generates workflow artifacts; two-level execution (meta→object); configurable failure handling (ESCALATE/REGENERATE/HYBRID); requires Extensions 01-02 |
