# Composite Guards

This document formally extends the Dual-State Framework to support guard composition, enabling multiple validation checks to be combined into a single guard function while preserving system dynamics.

> **Depends on**: Base paper (Definition 4: Guard Function, Definition 7: System Dynamics)
>
> **See also**: [06_generated_workflows.md](06_generated_workflows.md) (CompositeGuard usage), [00_notation_extensions.md](00_notation_extensions.md) (symbols)

---

## Relationship to Paper

The paper defines guards as:

```
G: A × C → {⊤, ⊥_retry, ⊥_fatal} × Σ*
```

This extension introduces **composite guards** that combine multiple guard functions while preserving this interface. The composition is opaque to the workflow executor — composite guards are guards.

**Key insight**: Composite guards enable all validation feedback to route to the generator that can act on it, solving the "IdentityGenerator problem" where pass-through generators cannot respond to guard failures.

---

## 1. Motivation

Consider a workflow with sequential validation steps:

```
g_coder → g_quality → g_arch_validate → g_merge_ready
```

When `g_quality` fails:

1. Feedback goes to `g_quality`'s generator
2. If using IdentityGenerator (pass-through), the same artifact is resubmitted
3. Same errors, same feedback, retry exhausted
4. The generator that could fix the code (`CoderGenerator`) never sees the feedback

**Root cause**: Validation guards are attached to the wrong generators.

**Solution**: Compose multiple guards into a single ActionPair so all feedback routes to the generator that can act on it.

### Before (Broken)

```
g_coder ─────────────────────────────────────────────────────┐
  Generator: CoderGenerator                                  │
  Guard: AllTestsPassGuard (syntax only)                     │
                                                             ↓
g_quality ─────────────────────────────────────────────────→ fails × 4
  Generator: IdentityGenerator (pass-through)                │
  Guard: QualityGatesGuard (mypy/ruff)                       │
  Problem: Feedback cannot reach CoderGenerator              │
```

### After (Composite Guard)

```
g_coder ──────────────────────────────────────────────────→ success
  Generator: CoderGenerator
  Guard: CompositeGuard([
    AllTestsPassGuard(),      # Syntax/structure - fast
    QualityGatesGuard(),      # mypy/ruff - medium
    ArchValidationGuard(),    # pytest-arch - slow
  ], compose=SEQUENTIAL)

  Feedback from ANY sub-guard → CoderGenerator → retry with fixes
```

---

## 2. Composite Guard

### Definition 38: Composite Guard

A **Composite Guard** is a guard function composed of multiple sub-guards:

```
G_composite = ⟨{G₁, ..., Gₙ}, compose, policy⟩

where:
  {G₁, ..., Gₙ}: Set of guard functions (Definition 4)
  compose: Composition strategy (SEQUENTIAL | PARALLEL)
  policy: Aggregation policy for results
```

The composite guard satisfies the Guard interface:

```
G_composite: A × C → {⊤, ⊥_retry, ⊥_fatal} × Σ*
```

### Axiom (Composition Closure)

Composite guards are guards. Any guard can be a sub-guard of a composite, enabling arbitrary nesting depth:

```
∀ G_c = ⟨{G₁, ..., Gₙ}, compose, policy⟩:
  G_c ∈ Guard ⟺ ∀i: Gᵢ ∈ Guard
```

---

## 3. Sequential Guard

### Definition 39: Sequential Guard (Fail-Fast)

A **Sequential Guard** executes sub-guards in order, stopping on first failure:

```
G_seq = ⟨[G₁, ..., Gₙ], SEQUENTIAL, FIRST_FAIL⟩

Execute(G_seq, a, C) =
  for i = 1 to n:
    (vᵢ, φᵢ) = Gᵢ(a, C)
    if vᵢ ≠ ⊤:
      return (vᵢ, φᵢ)  # Fail fast with this guard's feedback
  return (⊤, ε)  # All passed
```

### Properties

1. **Short-circuit evaluation**: Expensive guards skipped on early failure
2. **Feedback attribution**: Failure feedback comes from specific sub-guard
3. **Order matters**: Place cheap/fast guards first

### Remark (Cost Optimization)

If guards have costs c₁ < c₂ < ... < cₙ, sequential ordering minimizes expected validation cost on failure:

```
E[cost] = c₁ + P(G₁=⊤)·c₂ + P(G₁=⊤)·P(G₂=⊤)·c₃ + ...
```

---

## 4. Parallel Guard

### Definition 40: Parallel Guard (Concurrent)

A **Parallel Guard** executes sub-guards concurrently with configurable aggregation:

```
G_par = ⟨{G₁, ..., Gₙ}, PARALLEL, policy⟩

Execute(G_par, a, C) =
  results = parallel_map(λGᵢ. Gᵢ(a, C), {G₁, ..., Gₙ})
  return aggregate(results, policy)
```

### Properties

1. **All guards execute**: No short-circuit (useful for comprehensive feedback)
2. **Concurrent execution**: Wall-clock time = max(individual times)
3. **Complete feedback**: All failures reported, not just first

---

## 5. Aggregation Policy

### Definition 41: Aggregation Policy

An **Aggregation Policy** determines composite pass/fail from sub-guard results:

```
AggregationPolicy ∈ {ALL_PASS, ANY_PASS, MAJORITY_PASS}

aggregate: [{(v, φ)}] × Policy → (v, φ)
```

### ALL_PASS

All guards must pass for the composite to pass:

```
aggregate(results, ALL_PASS) =
  if ∀(vᵢ, φᵢ) ∈ results: vᵢ = ⊤
    then (⊤, ε)
    else (⊥_retry, concat({φᵢ | vᵢ ≠ ⊤}))
```

### ANY_PASS

At least one guard must pass:

```
aggregate(results, ANY_PASS) =
  if ∃(vᵢ, φᵢ) ∈ results: vᵢ = ⊤
    then (⊤, ε)
    else (⊥_retry, concat({φᵢ}))
```

### MAJORITY_PASS

More than half of guards must pass:

```
aggregate(results, MAJORITY_PASS) =
  let passed = |{(vᵢ, φᵢ) | vᵢ = ⊤}|
  if passed > n/2
    then (⊤, ε)
    else (⊥_retry, concat({φᵢ | vᵢ ≠ ⊤}))
```

---

## 6. Nested Composition

### Definition 42: Nested Composition

Guards can be **nested** arbitrarily using composition closure:

```
G_nested = Sequential([
    G_syntax,           # Fast, fail first
    Parallel([          # Independent, run concurrently
        G_mypy,
        G_ruff,
    ], ALL_PASS),
    G_architecture,     # Expensive, run last
])
```

### Remark (Practical Nesting Patterns)

| Pattern | Structure | Use Case |
|---------|-----------|----------|
| Fast-first | Sequential([cheap, expensive]) | Cost optimization |
| Parallel linters | Parallel([mypy, ruff, pylint]) | Independent checks |
| Layered | Sequential([syntax, Parallel([...]), tests]) | Combined approach |

---

## 7. Theorem: Dynamics Preservation

### Theorem 14: Composite Guard Preserves System Dynamics

**Statement**: Composite guards preserve Definition 7 (System Dynamics).

**Proof**:

1. By Definition 38, G_composite satisfies the Guard interface:

   ```
   G_composite: A × C → {⊤, ⊥_retry, ⊥_fatal} × Σ*
   ```

2. The composite guard's output (v, φ) is used identically to a simple guard in Definition 7:
   - If v = ⊤: Workflow advances (s_{t+1} = ⟨T(s_w, v), ...⟩)
   - If v = ⊥: Context refines (s_{t+1} = ⟨s_w, ⟨a', ⟨Ψ, H_t ∪ {(a', φ)}⟩⟩⟩)

3. The internal composition (sequential/parallel) is opaque to the workflow executor.

4. No new state spaces, transitions, or dynamics are introduced.

∎

### Corollary: Nested Composition Preserves Dynamics

By induction on nesting depth and Theorem 14, arbitrarily nested composite guards preserve system dynamics.

---

## 8. Sub-Guard Result

### Definition 43: Sub-Guard Result

A **Sub-Guard Result** captures the outcome of a single sub-guard within a composite:

```
SubGuardResult = ⟨guard_name, passed, feedback, execution_time⟩

where:
  guard_name: String      # Identifier for attribution
  passed: Boolean         # vᵢ = ⊤
  feedback: String        # φᵢ
  execution_time: Float   # Milliseconds
```

### Remark (Observability)

Sub-guard results enable:

1. **Performance profiling**: Identify slow guards
2. **Failure attribution**: Know which guard failed
3. **Debugging**: Full trace of composite execution

---

## 9. Schema Support

### Workflow Schema Extension

The workflow schema supports composite guards via `guard_config`:

```json
{
  "g_coder": {
    "generator": "CoderGenerator",
    "guard": "composite",
    "guard_config": {
      "compose": "sequential",
      "policy": "all_pass",
      "guards": [
        {"type": "syntax"},
        {"type": "quality_gates", "config": {"run_mypy": true}},
        {"type": "arch_validation"}
      ]
    }
  }
}
```

### Backwards Compatibility

Simple guard arrays remain supported:

```json
{
  "guard": "composite",
  "guards": ["syntax", "import", "tests"]
}
```

This is equivalent to `Sequential([...], ALL_PASS)`.

---

## 10. Implementation Patterns

### Python Implementation

```python
class SequentialGuard(GuardInterface):
    """Execute guards in sequence, fail-fast on first failure."""

    def __init__(self, guards: Sequence[GuardInterface],
                 policy: AggregationPolicy = AggregationPolicy.ALL_PASS):
        self._guards = guards
        self._policy = policy

    def validate(self, artifact: Artifact, **deps) -> GuardResult:
        results = []
        for guard in self._guards:
            result = guard.validate(artifact, **deps)
            results.append(SubGuardResult(
                guard_name=guard.__class__.__name__,
                passed=result.passed,
                feedback=result.feedback,
            ))
            if not result.passed:
                break  # Fail-fast
        return self._aggregate(results)
```

### Factory Pattern

```python
def create_composite_guard(spec: dict, registry: dict) -> GuardInterface:
    """Create composite guard from specification."""
    compose = spec.get("compose", "sequential")
    policy = AggregationPolicy(spec.get("policy", "all_pass"))

    guards = []
    for guard_spec in spec["guards"]:
        if isinstance(guard_spec, str):
            guards.append(registry[guard_spec]())
        else:
            guard_type = guard_spec["type"]
            config = guard_spec.get("config", {})
            guards.append(registry[guard_type](**config))

    if compose == "sequential":
        return SequentialGuard(guards, policy)
    else:
        return ParallelGuard(guards, policy)
```

---

## 11. Summary

| Definition | Name | Purpose |
|------------|------|---------|
| 38 | Composite Guard | Guard composed of multiple sub-guards |
| 39 | Sequential Guard | Fail-fast ordered execution |
| 40 | Parallel Guard | Concurrent execution with aggregation |
| 41 | Aggregation Policy | ALL_PASS, ANY_PASS, MAJORITY_PASS |
| 42 | Nested Composition | Arbitrary guard tree structures |
| 43 | Sub-Guard Result | Individual sub-guard outcome for attribution |

**Theorem 14** establishes that composite guards preserve Definition 7 (System Dynamics).

**Key benefit**: All validation feedback routes to the generator that can act on it, eliminating the useless IdentityGenerator retry loop.

---

## 12. Future Work

- **Weighted policies**: Guards with different voting weights
- **Conditional composition**: Skip guards based on artifact properties
- **Async parallel guards**: Better support for I/O-bound validation
- **Guard caching**: Skip re-execution of passed guards on retry

---

## See Also

- [06_generated_workflows.md](06_generated_workflows.md) — Uses CompositeGuard for workflow validation (Definition 28)
- [04_learning_loop.md](04_learning_loop.md) — Training from composite guard feedback
- [../agent_design_process/domain_definitions.md](../agent_design_process/domain_definitions.md) — Guard function (Definition 4)
