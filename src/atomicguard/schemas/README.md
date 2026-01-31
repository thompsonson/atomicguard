# AtomicGuard Configuration Schemas

This directory contains JSON Schema definitions for AtomicGuard configuration files.
These schemas align with the formal framework defined in the paper.

## Schemas

| Schema | File | Purpose |
|--------|------|---------|
| Workflow | `workflow.schema.json` | Defines workflow structure and action pairs |
| Prompts | `prompts.schema.json` | Defines prompt templates for generators |
| Artifact | `artifact.schema.json` | Defines artifact storage format in DAG |

## Formal Framework Alignment

The schemas map directly to the paper's mathematical definitions:

### Key Symbols

| Symbol | Name | Schema Location |
|--------|------|-----------------|
| Ψ | Specification | `workflow.specification` |
| Ω | Constraints | `workflow.constraints` + `action_pairs[].guard_config` |
| A | Action Pair | `workflow.action_pairs[]` |
| ρ | Precondition | `action_pairs[].requires` |
| G | Guard | `action_pairs[].guard` |
| G_θ | Parameterized Guard | `action_pairs[].guard` + `guard_config` |
| α | Artifact | `artifact.schema.json` root object |
| H | Feedback History | `artifact.context.feedback_history` |
| v | Guard Result | `artifact.guard_result.passed` + `.fatal` |
| φ | Feedback | `artifact.guard_result.feedback` |
| W_ref | Workflow Reference | `artifact.workflow_ref` (Extension 01) |
| Ψ_ref | Config Reference | `artifact.config_ref` (Extension 07) |

### Guard Parameters (θ)

Guard-specific thresholds are placed in `guard_config`:

```json
{
  "guard": "gates_extracted",
  "guard_config": {
    "min_gates": 3
  }
}
```

These parameters define acceptance thresholds: G_θ(α, C) = ⊤ iff threshold is met.

The guard function is parameterized:

```
G_θ: 𝒜 × 𝒞 → {⊤, ⊥_retry, ⊥_fatal} × Σ*

where θ = guard-specific parameters (e.g., min_gates=3)
```

### Composite Guards (Extension 08)

Guards can be composed using `CompositeGuardSpec`:

```json
{
  "guard": "composite",
  "guards": {
    "compose": "sequential",
    "policy": "all_pass",
    "guards": [
      "syntax",
      {"type": "quality_gates", "config": {"run_mypy": true}},
      {"compose": "parallel", "guards": ["ruff_check", "pylint_check"]}
    ]
  }
}
```

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `compose` | `sequential` \| `parallel` | `sequential` | Composition strategy (Definitions 39-40) |
| `policy` | `all_pass` \| `any_pass` \| `majority_pass` | `all_pass` | Aggregation policy (Definition 41) |
| `guards` | `GuardSpec[]` | - | Sub-guards to compose (supports nesting) |

Simple array syntax remains supported for backwards compatibility:

```json
{
  "guard": "composite",
  "guards": ["syntax", "import", "tests"]
}
```

This is equivalent to `Sequential([...], ALL_PASS)`.

### Specification Sources (Ψ)

Specification is decoupled from workflow and can be:

```json
// Inline text
"specification": "Implement a Stack with push/pop"

// File reference
"specification": { "type": "file", "ref": "specs/requirements.md" }

// Folder with glob
"specification": { "type": "folder", "ref": "docs/", "glob": "**/*.md" }

// External service
"specification": { "type": "service", "ref": "jira://PROJECT-123" }
```

### Action Pair Definition (A = ⟨ρ, a_gen, G⟩)

```json
{
  "g_test": {
    "guard": "composite",
    "guards": ["syntax", "human"],
    "requires": ["g_spec"]
  }
}
```

- **ρ (Precondition)**: `requires` lists action_pair_ids that must be ⊤
- **a_gen (Generator)**: Defined in `prompts.json` keyed by action_pair_id
- **G (Guard)**: `guard` type + optional `guard_config` for θ

### Context Hierarchy (C = ⟨ℰ, C_local, H⟩)

Stored in `artifact.context`:

```json
{
  "specification": "...",
  "constraints": "...",
  "feedback_history": [
    { "artifact_id": "...", "feedback": "..." }
  ],
  "dependency_artifacts": {
    "g_test": "artifact-uuid"
  }
}
```

Where:

- `specification` = Ψ (from C_local)
- `constraints` = Ω (from ℰ)
- `feedback_history` = H (accumulated)
- `dependency_artifacts` = Resolved ρ dependencies

### Extension Fields

The artifact schema includes fields from framework extensions:

| Field | Extension | Purpose |
|-------|-----------|---------|
| `source` | Core | Origin of content: `generated`, `human`, `imported` |
| `workflow_ref` | 01 (Versioned Environment) | W_ref - Content-addressed workflow hash for integrity verification |
| `config_ref` | 07 (Incremental Execution) | Ψ_ref - Configuration fingerprint for change detection |
| `metadata` | Core | Extensible metadata dictionary |

These enable:

- **Workflow integrity**: Verify workflow hasn't changed since checkpoint (W_ref)
- **Incremental execution**: Skip unchanged action pairs based on config fingerprint (Ψ_ref)
- **Provenance tracking**: Distinguish LLM-generated vs human-provided artifacts

## Validation

Use `jsonschema` to validate configuration files:

```python
from atomicguard.schemas import validate_workflow, validate_prompts

import json

# Validate workflow
with open("workflow.json") as f:
    data = json.load(f)
validate_workflow(data)  # Raises ValidationError if invalid

# Validate prompts
with open("prompts.json") as f:
    data = json.load(f)
validate_prompts(data)
```

Or use the lower-level API:

```python
import json
import jsonschema
from atomicguard.schemas import get_workflow_schema

schema = get_workflow_schema()
with open("workflow.json") as f:
    data = json.load(f)

# Get detailed error information
validator = jsonschema.Draft202012Validator(schema)
errors = list(validator.iter_errors(data))
for error in errors:
    print(f"{error.json_path}: {error.message}")
```

## IDE Support

Add schema references to your JSON files for autocomplete:

```json
{
  "$schema": "./src/atomicguard/schemas/workflow.schema.json",
  "name": "My Workflow",
  "action_pairs": { ... }
}
```

Or configure VS Code `settings.json`:

```json
{
  "json.schemas": [
    {
      "fileMatch": ["**/workflow.json"],
      "url": "./src/atomicguard/schemas/workflow.schema.json"
    },
    {
      "fileMatch": ["**/prompts.json"],
      "url": "./src/atomicguard/schemas/prompts.schema.json"
    }
  ]
}
```

## Example Workflow

Complete example showing all schema features:

```json
{
  "name": "TDD Stack with Import Guard",
  "description": "Test-driven development workflow with automated validation",
  "specification": {
    "type": "file",
    "ref": "specs/stack.md"
  },
  "constraints": "Use pytest, no external dependencies",
  "model": "qwen2.5-coder:14b",
  "rmax": 3,
  "action_pairs": {
    "g_test": {
      "guard": "composite",
      "guards": ["syntax", "import", "human"],
      "guard_config": {
        "human_prompt_title": "REVIEW GENERATED TESTS"
      }
    },
    "g_impl": {
      "guard": "dynamic_test",
      "requires": ["g_test"]
    }
  }
}
```

## Future: PDDL-like Syntax

These JSON schemas are designed to be transpilable to a more formal PDDL-like syntax:

```lisp
(define (workflow tdd-stack)
  (:specification (file "specs/stack.md"))
  (:constraints "Use pytest, no external dependencies")
  (:rmax 3)

  (:action-pair g_test
    :guard (composite syntax import human)
    :guard-config (:human-prompt-title "REVIEW TESTS"))

  (:action-pair g_impl
    :requires (g_test)
    :guard dynamic_test))
```

This can be achieved by writing a JSON → PDDL transpiler once the JSON Schema is stable.
