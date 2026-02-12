# Architecture Cleanup Issues

Anti-patterns identified during architectural review (2026-02-10). Each issue is self-contained — tests pass before and after. Ordered by dependency and risk.

## Status Summary

| Issue | Status | Commit |
|-------|--------|--------|
| 0: Architecture test infrastructure | DONE | `2c3f98d` |
| 1: Move StagnationInfo/FeedbackSummarizer to domain/ | DONE | `fd50863` |
| 2: Remove Extension 10 (WorkflowEventStore/Emitter) | DONE | `e27d20a` |
| 3: Fix dependency inversion in application/workflow.py | DONE | `7e7ccea` |
| 4: Fix silent exception swallowing in domain/ | DONE | `e44e170` |
| 5: Remove CheckpointDAG (DAG is the checkpoint) | DONE | `969330b` |
| 6: Deduplicate experiment runner workflow construction | DONE | `12a75d7` |
| 7: Fix generator-workflow coupling via hardcoded dep_id matching | PLANNED | 15 occurrences across 9 files |

---

## Issue 7: Fix generator-workflow coupling via hardcoded dep_id matching

**Status**: PLANNED

**Depends on**: None (standalone)

### Problem

Generators in `examples/swe_bench_ablation/generators/` contain hardcoded string matching on `dep_id` to find their dependencies. This creates tight coupling — if a workflow uses different action pair IDs than what the generator expects, the generator silently fails to find its context.

**Root cause example**: The decomposed workflow passed `ap_fix_approach` as a dependency to patch generator, but patch generator looks for `"analysis"` in `dep_id`. Since `"analysis"` is not in `"ap_fix_approach"`, the generator received no analysis context and produced empty patches.

```python
# Current pattern in generators (patch.py:317)
for dep_id, artifact_id in context.dependency_artifacts:
    if "analysis" in dep_id.lower():  # Hardcoded string match
        artifact = context.ambient.repository.get_artifact(artifact_id)
        # ...
```

This is an **implicit contract** between workflow step names and generator implementation. Changing a workflow's action pair IDs breaks generators without compile-time or runtime errors.

### All occurrences (15 total across 9 files)

| File | Line | Hardcoded String | Purpose |
|------|------|------------------|---------|
| `patch.py` | 317 | `"analysis"` | Find bug analysis |
| `patch.py` | 330 | `"localize"` | Find localization |
| `patch.py` | 343 | `"test"` | Find test code |
| `diff_review.py` | 84 | `"analysis"` | Find bug analysis |
| `diff_review.py` | 97 | `"test"` | Find test code |
| `diff_review.py` | 106 | `"patch"` | Find patch |
| `test_gen.py` | 80 | `"analysis"` | Find bug analysis |
| `root_cause.py` | 70 | `"classify"` | Find classification |
| `fix_approach.py` | 90 | `"root_cause"` | Find root cause |
| `fix_approach.py` | 103 | `"context_read"/"context"` | Find context |
| `fix_approach.py` | 116 | `"localise_issue"/"localize"` | Find localization |
| `workflow_gen.py` | 112 | `"classify"/"classification"` | Find classification |
| `context_read.py` | 89 | `"localise_issue"/"localize"` | Find localization |
| `test_localization.py` | 80 | `"localise_issue"/"localize"` | Find localization |
| `test_localization.py` | 93 | `"structure"` | Find structure |
| `impact_analysis.py` | 78 | `"fix_approach"` | Find fix approach |

### Files to modify

1. `examples/swe_bench_ablation/generators/patch.py`
2. `examples/swe_bench_ablation/generators/diff_review.py`
3. `examples/swe_bench_ablation/generators/test_gen.py`
4. `examples/swe_bench_ablation/generators/root_cause.py`
5. `examples/swe_bench_ablation/generators/fix_approach.py`
6. `examples/swe_bench_ablation/generators/workflow_gen.py`
7. `examples/swe_bench_ablation/generators/context_read.py`
8. `examples/swe_bench_ablation/generators/test_localization.py`
9. `examples/swe_bench_ablation/generators/impact_analysis.py`

### Suggested resolutions

#### Option A: Type-tagged artifacts (Recommended)

Add a `type` field to artifacts that indicates their semantic type (e.g., `"analysis"`, `"localization"`, `"test_code"`). Generators match on type instead of dep_id.

```python
# Instead of:
for dep_id, artifact_id in context.dependency_artifacts:
    if "analysis" in dep_id.lower():
        artifact = context.ambient.repository.get_artifact(artifact_id)

# Use:
for artifact in context.get_artifacts_by_type("analysis"):
    # artifact is already resolved
```

**Pros**: Clean separation, workflow-agnostic, explicit typing
**Cons**: Requires schema changes to Artifact model

#### Option B: Resolved content in context

Workflow resolves artifact UUIDs before calling generator and passes actual content directly in context. Generators don't need to fetch or match.

```python
# Workflow does:
context.dependency_content = {
    "analysis": resolved_analysis_object,
    "localization": resolved_localization_object,
}

# Generator does:
analysis = context.dependency_content.get("analysis")
```

**Pros**: Simplest for generators, no fetching logic
**Cons**: Still relies on naming convention, just moves coupling to workflow

#### Option C: Schema-based dispatch

Generators auto-detect artifact type by validating against known Pydantic schemas.

```python
def _detect_artifact_type(self, content: str) -> str:
    data = json.loads(content)
    for schema_name, schema_class in [("analysis", Analysis), ("localization", Localization)]:
        try:
            schema_class.model_validate(data)
            return schema_name
        except ValidationError:
            continue
    return "unknown"
```

**Pros**: No naming convention needed, self-describing
**Cons**: Expensive validation, ambiguous if schemas overlap

### Risk

Medium. Changes affect how all generators resolve their dependencies. Requires coordinated update across 9 generator files and potentially the Artifact model or Context class.

### Verification

After fix, run decomposed workflow and verify:

1. Patch generator receives fix_approach content
2. All 12 `ap_gen_patch` attempts produce non-empty edits
3. No "No patch or edits found in output" errors from missing context

### Acceptance criteria

- No hardcoded string matching on `dep_id` in any generator
- Generators receive correct dependency content regardless of workflow step naming
- Decomposed workflow produces non-empty patches
- All existing tests pass

---

## Background: Formal Framework Alignment

These issues were identified by reviewing the implementation against the formal framework in `docs/design/agent_design_process/domain_definitions.md`:

- **S_env** (Information State): The append-only artifact DAG. The execution trace IS the repository (SS2.2.3).
- **S_workflow** (Control State): Deterministic FSM tracking guard satisfaction and transition history (SS2.4.2).
- **Separation of concerns**: Domain is pure models and business rules. Application orchestrates. Infrastructure adapts.

The anti-patterns arose primarily from Extension 09/10 implementation that introduced parallel stores and observer-pattern coupling instead of working within the existing architectural boundaries.
