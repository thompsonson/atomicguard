# Appendix C: Guard Function Catalog

This appendix enumerates the deterministic guard functions (𝒢) that enforce correctness constraints across the multi-agent SDLC workflow. Each guard validates a specific state transition and is defined by a conjunction of verifiable predicates.

> **Note**: This document covers guards G₁–G₂₂ (implementation scope). The paper defines additional guards (G₂₃–G₂₉) for version control and legacy system bootstrapping. See [paper_scope_differences.md](paper_scope_differences.md) for details on deferred features.

## Notation

- **Gᵢ**: Boolean-valued sensing action
- **∧**: Strict conjunction (all must hold)
- **∨**: Valid alternatives
- **→**: State transition guarded

---

## Phase 1: Architecture Definition (ADD Agent)

### G₁: INTENT_RECEIVED → DOMAIN_MODEL_DEFINED

```
entities_identified ∧
value_objects_identified ∧
invariants_documented ∧
ubiquitous_language_defined
```

### G₂: DOMAIN_MODEL_DEFINED → PROJECT_STRUCTURE_DEFINED

```
layer_boundaries_specified ∧
directory_tree_valid ∧
package_structure_documented
```

### G₃: PROJECT_STRUCTURE_DEFINED → SKELETON_CREATED

```
all_documented_dirs_exist ∧
init_files_present ∧
structure_matches_specification
```

### G₄: SKELETON_CREATED → ARCHITECTURE_TESTS_GENERATED

```
pytestarch_syntax_valid ∧
all_gates_have_tests ∧
test_file_imports_resolve
```

---

## Phase 2: Test Definition (TDD/BDD Agents)

### G₅: DOMAIN_MODEL_DEFINED → UNIT_TESTS_GENERATED

```
entity_lifecycle_tests_exist ∧
value_object_immutability_tested ∧
business_rules_validated ∧
fixtures_defined
```

### G₆: INTENT_RECEIVED → BDD_SCENARIOS_DEFINED

```
gherkin_syntax_valid ∧
actors_identified ∧
outcomes_testable ∧
step_definitions_scaffolded
```

---

## Phase 3: Implementation (Coder Agent)

### G₇: FILE_CREATION_REQUESTED → FILE_VALIDATED

```
(path_in_documented_structure ∨
  (architectural_rules_satisfied ∧ documentation_updated)) ∧
layer_boundaries_enforced ∧
parent_directories_exist
```

### G₈: CODE_GENERATED → SYNTAX_VALIDATED

```
ast_parse_succeeds ∧
imports_resolve ∧
no_syntax_errors
```

**Implementation**: `SyntaxGuard`

### G₉: SYNTAX_VALIDATED → TYPE_VALIDATED

```
mypy_check_passes ∧
type_annotations_present ∧
no_type_mismatches
```

**Implementation**: `TypeGuard`

### G₁₀: TYPE_VALIDATED → FUNCTIONALLY_CORRECT

```
unit_tests_pass ∧
integration_tests_pass ∧
test_coverage ≥ threshold
```

**Implementation**: `TestGuard`, `DynamicTestGuard`

---

## Phase 4: Architectural Compliance (Quality Gates)

### G₁₁: IMPLEMENTATION_COMPLETE → ARCHITECTURE_VALIDATED

```
domain_never_imports_infrastructure ∧
application_never_imports_infrastructure ∧
infrastructure_only_imports_interfaces ∧
no_circular_dependencies
```

### G₁₂: ARCHITECTURE_VALIDATED → DI_VALIDATED

```
container_registers_interfaces_only ∧
no_concrete_classes_in_registry ∧
factory_interfaces_in_domain ∧
factory_implementations_in_infrastructure
```

### G₁₃: DI_VALIDATED → FACTORY_VALIDATED

```
factory_names_match_purpose ∧
factories_return_interfaces ∧
no_hardcoded_instantiation ∧
all_dependencies_injectable
```

---

## Phase 5: Behavioral Validation (Tester Agent)

### G₁₄: FUNCTIONALLY_CORRECT → BDD_VALIDATED

```
acceptance_tests_pass ∧
workflows_complete_successfully ∧
error_paths_handled ∧
cli_commands_functional
```

### G₁₅: BDD_VALIDATED → QUALITY_GATES_PASSING

```
code_formatted ∧
linter_score ≥ threshold ∧
security_scan_clean ∧
no_hardcoded_secrets
```

---

## Phase 6: Operational Safety

### G₁₆: EXECUTION_REQUESTED → EXECUTION_SAFE

```
timeout_enforced ∧
sandbox_boundaries_respected ∧
no_wildcard_operations ∧
rollback_available
```

**Implementation**: `TimeoutGuard`, `SandboxGuard`

### G₁₇: FILE_OPERATION_REQUESTED → FILE_OPERATION_SAFE

```
path_within_workspace ∧
no_sensitive_paths_accessed ∧
(backup_exists ∨ operation_idempotent)
```

**Implementation**: `PathGuard`

---

## Phase 7: Structure Audit (Project Structure Agent)

### G₁₈: IMPLEMENTATION_COMPLETE → STRUCTURE_AUDITED

```
no_misplaced_files ∧
all_files_documented ∧
layer_violations_zero ∧
documentation_synchronized
```

### G₁₉: STRUCTURE_AUDITED → GAP_ANALYSIS_READY

```
violations_categorized ∧
responsible_agents_identified ∧
remediation_paths_defined
```

---

## Phase 8: Human Oversight (Human-in-the-Loop)

### G₂₀: CANDIDATE_READY → HUMAN_APPROVED

Pauses workflow to poll an external oracle (human) for approval.

```
review_session_completed ∧
approval_signature_present
```

**Implementation**: `HumanGuard`

---

## Composite Guards

### G₂₁: CompositeGuard

Sequential evaluation with fail-fast semantics:

```
G_composite(a) = G₁(a) ∧ G₂(a) ∧ ... ∧ Gₙ(a)
```

Returns first failure feedback for context refinement.

### G₂₂: ALL_STREAMS_VALIDATED → PRODUCTION_READY

```
G₁₁ ∧ G₁₂ ∧ G₁₃ ∧    /* Architecture stream */
G₁₀ ∧ G₁₄ ∧          /* Functional stream */
G₁₅ ∧ G₁₈ ∧          /* Quality/Structure streams */
G₂₀                   /* Human approval */
```

---

## Implementation Reference

| Guard | Class | Predicates |
|-------|-------|------------|
| G₈ | `SyntaxGuard` | `ast.parse()` succeeds |
| G₉ | `TypeGuard` | `mypy` returns no errors |
| G₁₀ | `TestGuard` | All test assertions pass |
| G₁₀ | `DynamicTestGuard` | Generated tests pass |
| G₁₆ | `TimeoutGuard` | Execution < timeout |
| G₂₀ | `HumanGuard` | Human approves via input |
| G₂₁ | `CompositeGuard` | All sub-guards pass |

---

## Guard Interface

All guards implement:

```python
class GuardInterface(ABC):
    @abstractmethod
    def validate(self, artifact: Artifact, **deps) -> GuardResult:
        """Returns GuardResult with validation outcome."""

@dataclass
class GuardResult:
    passed: bool          # ⊤ (accept) or ⊥ (reject)
    feedback: str = ""    # φ ∈ Σ* - feedback for next attempt
    fatal: bool = False   # ⊥_fatal - skip retry, escalate to human
```

The `feedback` field provides context for the next generation attempt when `passed=False`.

---

## Fatal Guard Semantics

Guards may return `fatal=True` to indicate **non-recoverable failures** that should not be retried. This implements the guard fatal state (`⊥_fatal`) from Definition 6.

### When to Use Fatal

| Scenario | Example | Rationale |
|----------|---------|-----------|
| Security violation | Code attempts file system access outside sandbox | Cannot be fixed by regeneration |
| Impossible specification | Tests require conflicting behaviors | Specification error, not generation error |
| Human-approved artifact fails | Human approved tests that have syntax errors | Human must review their approval |
| Resource exhaustion | Generated code exceeds memory limits | Architectural constraint violation |

### Behavior on Fatal

1. **No retry**: Agent raises `EscalationRequired` immediately
2. **Artifact preserved**: Failed artifact stored in DAG for review
3. **Workflow halts**: Returns `WorkflowStatus.ESCALATION`
4. **Feedback surfaced**: `escalation_feedback` contains guard's message

### Example: Security Guard with Fatal

```python
class SandboxGuard(GuardInterface):
    """G₁₆: Validates execution safety with fatal on violations."""

    FORBIDDEN_PATTERNS = ['os.system', 'subprocess.run', 'eval(', 'exec(']

    def validate(self, artifact: Artifact, **deps) -> GuardResult:
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in artifact.content:
                return GuardResult(
                    passed=False,
                    feedback=f"Security violation: {pattern} not allowed",
                    fatal=True  # No retry - escalate immediately
                )
        return GuardResult(passed=True)
```

### Distinction from Retryable Failures

| Failure Type | `fatal` | Action | Example |
|--------------|---------|--------|---------|
| Retryable | `False` | Refine context, retry | Syntax error, test failure |
| Fatal | `True` | Escalate to human | Security violation, impossible spec |
