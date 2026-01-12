# 07 — Incremental Execution

**Definitions 33-37.** Formalizes configuration-based change detection for selective re-execution of action pairs. Like incremental builds in CI/CD systems, only action pairs with changed inputs are re-executed—unchanged steps reuse previously accepted artifacts.

Uses content-addressable configuration references (Ψ_ref) with Merkle-style dependency propagation to detect which action pairs require re-execution.

> **Depends on**: [01_versioned_environment.md](01_versioned_environment.md) (W_ref, Definitions 10-16), [03_multiagent.md](03_multiagent.md) (Shared repository, Definitions 19-20)
>
> **See also**: [00_notation_extensions.md](00_notation_extensions.md) (symbols)

---

## Relationship to Paper

This extension builds on the Versioned Repository (Definition 4) and Artifact model. The paper stores all generation attempts for provenance; this extension adds configuration-based change detection to enable incremental execution.

**Key insight**: The repository already stores artifacts with full context. By adding a configuration fingerprint (Ψ_ref), we can detect which action pairs have unchanged inputs and skip re-execution—analogous to how build systems skip unchanged targets.

**Execution modes:**

- **Full Execution**: Execute all action pairs regardless of prior state (clean build)
- **Incremental Execution**: Execute only action pairs with changed inputs (incremental build)

---

## 1. Configuration Reference

**Definition 33 (Configuration Reference).** A configuration reference Ψ_ref is a content-addressable identifier for the complete configuration state of an action pair:

```
Ψ_ref : AP → H

where:
  AP = set of action pairs in workflow
  H = set of cryptographic hashes (SHA-256)
```

The reference is computed as:

```
Ψ_ref(ap) = hash(
  prompt(ap),
  model(ap),
  guard_config(ap),
  {Ψ_ref(dep) : dep ∈ requires(ap)},
  {content_hash(artifact(dep)) : dep ∈ requires(ap)}
)
```

**Components:**

- `prompt(ap)`: Prompt configuration (role, constraints, task, feedback_wrapper)
- `model(ap)`: Model identifier and sampling parameters
- `guard_config(ap)`: Guard-specific configuration
- `requires(ap)`: Set of upstream action pair dependencies
- `artifact(dep)`: Accepted output artifact from dependency

**Remark (Root Action Pairs).** Action pairs with no dependencies have `requires(ap) = ∅`, so their Ψ_ref is computed purely from their own configuration:

```
Ψ_ref(ap_root) = hash(prompt(ap_root), model(ap_root), guard_config(ap_root), {}, {})
```

**Remark (W_ref vs Ψ_ref).** Both are content-addressable hashes but differ in scope:

| Hash | Scope | Purpose | Changes When |
|------|-------|---------|--------------|
| W_ref | Entire workflow structure | Integrity verification on resume | Workflow steps/deps change |
| Ψ_ref | Single action pair + upstream | Cache invalidation | Prompt, model, guard config, or upstream artifact changes |

W_ref changing implies all Ψ_ref values change (workflow structure is input to action pair config). However, Ψ_ref can change independently—e.g., editing a prompt changes Ψ_ref without changing W_ref if workflow structure is unchanged.

---

## 2. Dependency Graph

**Definition 34 (Action Pair Dependency Graph).** The dependency graph D = (AP, E) is a directed acyclic graph where:

```
AP = {ap₁, ap₂, ..., apₙ}  (action pairs)
E ⊆ AP × AP                 (dependency edges)

(apᵢ, apⱼ) ∈ E  ⟺  apᵢ ∈ requires(apⱼ)
```

**Dependency direction:** Edges point from prerequisite to dependent. If `g_test → g_impl`, then `g_test` must complete before `g_impl` can execute.

```
g_test → g_impl → g_coder
  ↑         ↑
  │         └── requires(g_impl) = {g_test}
  │
  └── requires(g_test) = {}
```

**Topological ordering:** Action pairs execute in order respecting E. For any (apᵢ, apⱼ) ∈ E, apᵢ completes before apⱼ begins.

---

## 3. Merkle Propagation

**Definition 35 (Merkle Propagation).** Configuration references propagate through the dependency graph such that any upstream change invalidates downstream references.

**Propagation property:**

```
∀ apᵢ, apⱼ ∈ AP :
  (apᵢ, apⱼ) ∈ E ∧ Ψ_ref(apᵢ) changes
  ⟹ Ψ_ref(apⱼ) changes
```

**Proof sketch:** Ψ_ref(apⱼ) includes Ψ_ref(apᵢ) as hash input. Change in input → change in hash output (collision resistance).

**Transitive closure:** For path apᵢ → ... → apₖ in D:

```
Ψ_ref(apᵢ) changes ⟹ Ψ_ref(apₖ) changes
```

**Why include upstream artifact content?**

If `g_test` produces different test content (even with same config), then `g_impl` should re-run because it depends on those tests. The artifact hash captures "what did my dependency actually produce?"

**Example:**

1. Run 1: `g_test` produces `test_add.py` with content hash `abc123`
2. `g_impl` runs, its Ψ_ref includes `abc123`
3. Run 2: `g_test` config unchanged, but human amended the test to `test_add_v2.py` with hash `def456`
4. `g_impl` Ψ_ref now different (includes `def456`), change detected → re-execute

---

## 4. Change Detection

**Definition 36 (Unchanged Action Pair).** An action pair is unchanged (can be skipped) iff a matching accepted artifact exists in the repository:

```
unchanged(ap) ⟺ ∃a ∈ ℛ :
  a.action_pair_id = ap ∧
  a.config_ref = Ψ_ref_current(ap) ∧
  a.status = ACCEPTED
```

Where:

- `Ψ_ref_current(ap)`: Configuration reference computed from current workflow.json, prompts.json, and upstream artifacts
- `ACCEPTED`: Guard passed (per `ArtifactStatus` enum)

**Execution decision:**

```
execute(ap) ⟺ ¬unchanged(ap)   # Execute if changed or never run
skip(ap)    ⟺ unchanged(ap)    # Skip if unchanged
```

**Remark (REJECTED artifacts).** Artifacts with `status = REJECTED` are stored for provenance but do not satisfy the unchanged predicate. A rejected artifact indicates the configuration produced invalid output; re-execution may succeed with different LLM sampling.

---

## 5. Artifact Model Extension

The existing `Artifact` model (Definition 4) is extended with a configuration reference field:

```python
@dataclass(frozen=True)
class Artifact:
    # ... existing fields ...

    # Extension 07: Incremental Execution (Definition 33)
    config_ref: str | None = None  # Ψ_ref: Configuration fingerprint for change detection
```

This mirrors the existing `workflow_ref: str | None` field from Extension 01.

**Lookup predicate:**

```
lookup(ap, Ψ_ref) = {a ∈ ℛ : a.action_pair_id = ap ∧ a.config_ref = Ψ_ref}
```

---

## 6. Multi-Agent Change Detection Coherence

**Theorem 7 (Change Detection Coherence).** In a multi-agent workflow sharing repository ℛ, agents observe consistent change detection results without explicit synchronization.

**Proof:**

1. Ψ_ref is deterministically computed from shared inputs (workflow.json, prompts.json, upstream artifacts in ℛ)
2. All agents compute identical Ψ_ref for same action pair given same inputs
3. Repository lookup `∃a ∈ ℛ : a.config_ref = Ψ_ref_current(ap)` yields same result for all agents
4. First agent to execute stores artifact with config_ref; subsequent agents detect unchanged state

**Corollary:** No message passing required for coordination. Shared repository provides consistency via Definition 19 (Multi-Agent System).

---

## 7. Invalidation Cascade

**Algorithm 1: Invalidation Detection**

```python
def invalidated_action_pairs(changed_aps: set[str], D: Graph) -> set[str]:
    """
    Given set of action pairs with changed configuration,
    return all action pairs requiring re-execution.
    """
    invalidated = set(changed_aps)

    for ap in topological_order(D):
        if any(dep in invalidated for dep in requires(ap)):
            invalidated.add(ap)

    return invalidated
```

**Example:**

```
D: g_config → g_add → g_coder
           → g_bdd ↗

Case 1: changed_aps = {g_config}
        invalidated = {g_config, g_add, g_bdd, g_coder}

Case 2: changed_aps = {g_add}
        invalidated = {g_add, g_coder}
        # g_config, g_bdd unchanged
```

---

## 8. Concrete Example

**Initial state:** Empty repository, workflow with three action pairs.

```
Workflow: g_test → g_impl → g_review
```

**Run 1: Full execution** (no prior artifacts)

| Step | Ψ_ref | Changed? | Action |
|------|-------|----------|--------|
| g_test | `a1b2c3...` | yes (new) | Execute, store artifact |
| g_impl | `d4e5f6...` | yes (new) | Execute, store artifact |
| g_review | `g7h8i9...` | yes (new) | Execute, store artifact |

**Run 2: Incremental execution** (no config changes)

| Step | Ψ_ref | Changed? | Action |
|------|-------|----------|--------|
| g_test | `a1b2c3...` | **no** | Skip |
| g_impl | `d4e5f6...` | **no** | Skip |
| g_review | `g7h8i9...` | **no** | Skip |

**Run 3: Incremental execution** (prompt change to g_impl)

| Step | Ψ_ref | Changed? | Action |
|------|-------|----------|--------|
| g_test | `a1b2c3...` | **no** | Skip |
| g_impl | `x1y2z3...` | yes (config) | Execute, store artifact |
| g_review | `j0k1l2...` | yes (cascade) | Execute, store artifact |

**Run 4: Incremental execution** (human amends g_test artifact)

| Step | Ψ_ref | Changed? | Action |
|------|-------|----------|--------|
| g_test | `a1b2c3...` | **no** | Skip (config unchanged) |
| g_impl | `m3n4o5...` | yes (upstream artifact) | Execute |
| g_review | `p6q7r8...` | yes (cascade) | Execute |

---

## 9. Implementation Notes

**Hash function requirements:**

- Collision resistant (SHA-256)
- Deterministic serialization of inputs (canonical JSON, sorted keys)

**Computing Ψ_ref:**

```python
def compute_config_ref(
    action_pair_id: str,
    workflow_config: dict,
    prompt_config: dict,
    upstream_artifacts: dict[str, Artifact],
) -> str:
    """Compute Ψ_ref for an action pair."""
    ap_config = workflow_config["action_pairs"][action_pair_id]
    prompt = prompt_config.get(action_pair_id, {})

    # Collect upstream refs and artifact hashes
    upstream_refs = {}
    artifact_hashes = {}
    for dep_id in ap_config.get("requires", []):
        dep_artifact = upstream_artifacts[dep_id]
        upstream_refs[dep_id] = dep_artifact.config_ref
        artifact_hashes[dep_id] = hashlib.sha256(
            dep_artifact.content.encode()
        ).hexdigest()

    # Build canonical input
    hash_input = {
        "prompt": prompt,
        "model": ap_config.get("model", workflow_config.get("model")),
        "guard_config": ap_config.get("guard_config", {}),
        "upstream_refs": upstream_refs,
        "artifact_hashes": artifact_hashes,
    }

    # Compute hash
    canonical = json.dumps(hash_input, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
```

---

## 10. Relationship to Base Framework

| Base Concept | Extension |
|--------------|-----------|
| Artifact model | Extended with `config_ref` field |
| ArtifactStatus.ACCEPTED | Unchanged detection criterion |
| Repository ℛ | Source of truth for change detection |
| Multi-agent coordination | Coherent change detection via shared repository |

The base system dynamics (Definition 7) are preserved. Ψ_ref provides an efficient mechanism for change detection without altering execution semantics.

---

## 11. Summary

| Definition | Name | Purpose |
|------------|------|---------|
| 33 | Configuration Reference | Content-addressable fingerprint of action pair config |
| 34 | Dependency Graph | DAG structure for change propagation |
| 35 | Merkle Propagation | Upstream changes cascade to downstream |
| 36 | Unchanged Action Pair | Criteria for skipping re-execution |
| 37 | Artifact Extension | `config_ref` field on Artifact model |

**Theorem 7** establishes that multi-agent systems achieve coherent change detection through the shared repository without explicit synchronization.

**Execution modes:**

- **Full Execution**: `execute(ap) = true` for all action pairs (ignore change detection)
- **Incremental Execution**: `execute(ap) = ¬unchanged(ap)` (skip unchanged action pairs)

---

## Dependency Graph

```
┌─────────────────────────────────────────┐
│  Versioned Environment (Def 10-16)      │
│  - W_ref content addressing             │
│  - Repository items                     │
└─────────────────────────────────────────┘
              │
              ├──────────────────────────────┐
              ▼                              ▼
┌──────────────────────────┐    ┌────────────────────────────┐
│  Artifact Extraction     │    │  Multi-Agent Workflows     │
│  (Def 17-18)             │    │  (Def 19-20)               │
│  - Query predicates      │    │  - Shared repository       │
└──────────────────────────┘    └────────────────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
              ┌────────────────────────────┐
              │  Incremental Execution     │
              │  (Def 33-37)               │
              │  - Ψ_ref change detection  │
              │  - Selective re-execution  │
              └────────────────────────────┘
```

---

## See Also

- [01_versioned_environment.md](01_versioned_environment.md) — W_ref content addressing (Definition 11)
- [02_artifact_extraction.md](02_artifact_extraction.md) — Query predicates for artifact lookup
- [03_multiagent.md](03_multiagent.md) — Shared repository (Definition 19)
- [06_generated_workflows.md](06_generated_workflows.md) — Dynamic workflow generation
