# Enhanced SDLC Workflow (sdlc_v2)

An SDLC workflow demonstrating AtomicGuard extensions for software development lifecycle automation.

## Overview

This workflow generates implementation code from architecture documentation and requirements through a series of guarded action pairs. Each step produces an artifact that is validated before proceeding.

## Workflow Diagram

```
                              DESIGN PHASE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    ┌─────────────────────────────────────────────────────┐  │
│                    │                     SPECIFICATION                   │  │
│                    │            (architecture.md + requirements.md)      │  │
│                    └─────────────────────────────────────────────────────┘  │
│                                            │                                │
│                                            ▼                                │
│                    ┌─────────────────────────────────────────────────────┐  │
│                    │                    g_config                         │  │
│                    │       ConfigExtractorGenerator → ConfigGuard        │  │
│                    │  Output: ProjectConfig {source_root, package_name}  │  │
│                    └─────────────────────────────────────────────────────┘  │
│                                            │                                │
│                         ┌──────────────────┼──────────────────┐             │
│                         │                  │                  │             │
│                         ▼                  │                  ▼             │
│    ┌────────────────────────────────┐     │     ┌────────────────────────┐ │
│    │            g_add               │     │     │         g_bdd          │ │
│    │  ADDGenerator → ArchTestsGuard │     │     │ BDDGenerator → BDDGuard│ │
│    │  Output: TestSuite (pytest-arch│     │     │ Output: BDDScenarios   │ │
│    └────────────────────────────────┘     │     └────────────────────────┘ │
│                         │                  │                  │             │
│                         ▼                  │                  │             │
│    ┌────────────────────────────────┐     │                  │             │
│    │           g_rules              │     │                  │             │
│    │ RulesExtractor → RulesGuard    │     │                  │             │
│    │ (DETERMINISTIC - no LLM)       │     │                  │             │
│    │ Output: ArchitectureRules      │     │                  │             │
│    └────────────────────────────────┘     │                  │             │
│                         │                  │                  │             │
└─────────────────────────┼──────────────────┼──────────────────┼─────────────┘
                          │                  │                  │
                          └──────────────────┼──────────────────┘
                                             │
                          IMPLEMENTATION PHASE
┌────────────────────────────────────────────┼────────────────────────────────┐
│                                            ▼                                │
│                    ┌─────────────────────────────────────────────────────┐  │
│                    │                    g_coder                          │  │
│                    │          CoderGenerator → AllTestsPassGuard         │  │
│                    │  Input: g_rules + g_bdd + Specification             │  │
│                    │  Output: ImplementationResult {files[], summary}    │  │
│                    └─────────────────────────────────────────────────────┘  │
│                                            │                                │
└────────────────────────────────────────────┼────────────────────────────────┘
                                             │
                            VALIDATION PHASE
┌────────────────────────────────────────────┼────────────────────────────────┐
│                                            ▼                                │
│                    ┌─────────────────────────────────────────────────────┐  │
│                    │                   g_quality                         │  │
│                    │        IdentityGenerator → QualityGatesGuard        │  │
│                    │  Runs: mypy + ruff in temp directory                │  │
│                    └─────────────────────────────────────────────────────┘  │
│                                            │                                │
│                                            ▼                                │
│                    ┌─────────────────────────────────────────────────────┐  │
│                    │                g_arch_validate                      │  │
│                    │        IdentityGenerator → ArchValidationGuard      │  │
│                    │  Runs: pytest-arch tests from g_add                 │  │
│                    └─────────────────────────────────────────────────────┘  │
│                                            │                                │
│                                            ▼                                │
│                    ┌─────────────────────────────────────────────────────┐  │
│                    │                 g_merge_ready                       │  │
│                    │         IdentityGenerator → MergeReadyGuard         │  │
│                    │  Checks: All prior gates passed                     │  │
│                    └─────────────────────────────────────────────────────┘  │
│                                            │                                │
└────────────────────────────────────────────┼────────────────────────────────┘
                                             │
                                             ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                 FILE EXTRACTION                         │
                    │   FileExtractor service writes files to filesystem      │
                    │   (Only after ALL validation gates pass)                │
                    └─────────────────────────────────────────────────────────┘
```

## Dependency Graph

```
g_config ─┬─→ g_add ─────→ g_rules ─────────────┐
          │                    │                │
          └─→ g_bdd ───────────┼────────────────┼─→ g_coder ─→ g_quality ─→ g_arch_validate ─→ g_merge_ready
                               │                │                              │
                               └────────────────┴──────────────────────────────┘
```

## Action Pairs

### 1. g_config - Project Configuration Extraction

| Component | Description |
|-----------|-------------|
| **Generator** | `ConfigExtractorGenerator` |
| **Guard** | `ConfigGuard` |
| **LLM Required** | Yes |
| **Dependencies** | None (root step) |

**Purpose**: Extract project metadata from architecture documentation.

**Input**: Raw specification text (architecture + requirements docs)

**Output** (`ProjectConfig`):

```json
{
  "source_root": "src/taskmanager",
  "package_name": "taskmanager"
}
```

**Guard Validation**:

- Valid JSON structure
- `source_root` is non-empty
- No error field in response

---

### 2. g_add - Architecture Decision Documentation

| Component | Description |
|-----------|-------------|
| **Generator** | `ADDGenerator` |
| **Guard** | `ArchitectureTestsGuard` |
| **LLM Required** | Yes |
| **Dependencies** | `g_config` |

**Purpose**: Generate pytest-arch tests that enforce architectural constraints from the documentation.

**Input**: Specification + project config

**Output** (`TestSuite`):

```json
{
  "module_docstring": "Architecture tests for task manager",
  "imports": ["from pytestarch import Rule", "import pytest"],
  "fixtures": ["@pytest.fixture..."],
  "tests": [
    {
      "gate_id": "Gate1",
      "test_name": "test_domain_no_infrastructure_imports",
      "test_code": "def test_domain_no_infrastructure_imports(evaluable): ...",
      "documentation_reference": "Gate 1: Domain Purity"
    }
  ]
}
```

**Guard Validation**:

- Valid JSON matching TestSuite schema
- Minimum test count met
- Test names start with `test_`
- No duplicate test names
- Generated code is syntactically valid Python
- Only whitelisted pytestarch API methods used

---

### 3. g_bdd - Behavior-Driven Development Scenarios

| Component | Description |
|-----------|-------------|
| **Generator** | `BDDGenerator` |
| **Guard** | `BDDGuard` |
| **LLM Required** | Yes |
| **Dependencies** | `g_config` |

**Purpose**: Generate Gherkin scenarios that capture acceptance criteria from requirements.

**Input**: Specification + project config

**Output** (`BDDScenariosResult`):

```json
{
  "feature_name": "Task Management",
  "background": "Given a task repository is initialized",
  "scenarios": [
    {
      "name": "Create a new task",
      "feature": "Task Management",
      "gherkin": "Scenario: Create a new task\n  Given I have a task management system\n  When I create a task with title 'Test Task'..."
    }
  ]
}
```

**Guard Validation**:

- Valid JSON matching BDDScenariosResult schema
- Minimum scenario count met
- Each scenario has name, feature, and gherkin text

---

### 4. g_rules - Architecture Rules Extraction

| Component | Description |
|-----------|-------------|
| **Generator** | `RulesExtractorGenerator` |
| **Guard** | `RulesGuard` |
| **LLM Required** | **No** (deterministic) |
| **Dependencies** | `g_add`, `g_config` |

**Purpose**: Extract structured, actionable architecture rules from g_add tests. This step transforms the test artifacts into explicit constraints for the coder.

**Input**: g_add artifact (TestSuite) + g_config artifact (ProjectConfig)

**Output** (`ArchitectureRules`):

```json
{
  "import_rules": [
    {
      "source_layer": "domain",
      "forbidden_targets": ["infrastructure", "application"],
      "rationale": "Domain must be pure with no external dependencies"
    },
    {
      "source_layer": "application",
      "forbidden_targets": ["infrastructure"],
      "rationale": "Application depends on abstractions, not implementations"
    }
  ],
  "folder_structure": [
    {
      "layer": "domain",
      "path": "src/taskmanager/domain/",
      "allowed_modules": ["entities.py", "value_objects.py", "exceptions.py"],
      "purpose": "Pure business logic with no external dependencies"
    }
  ],
  "dependency_direction": "infrastructure → application → domain",
  "layer_descriptions": {
    "domain": "Pure business logic",
    "application": "Use cases and orchestration",
    "infrastructure": "External integrations"
  },
  "package_name": "taskmanager",
  "source_root": "src/taskmanager"
}
```

**Extraction Logic**:

- Parses test names like `test_domain_no_infrastructure_imports` → `domain` cannot import `infrastructure`
- Extracts folder structure from Package Structure section in specification
- Merges rules for same source layer

**Guard Validation**:

- Valid JSON matching ArchitectureRules schema
- At least one import rule extracted
- At least one folder structure entry
- All rules have forbidden targets
- All folders have paths

---

### 5. g_coder - Implementation Generation

| Component | Description |
|-----------|-------------|
| **Generator** | `CoderGenerator` |
| **Guard** | `AllTestsPassGuard` |
| **LLM Required** | Yes |
| **Dependencies** | `g_rules`, `g_bdd` |

**Purpose**: Generate Python implementation that satisfies all architecture rules and BDD scenarios.

**Input**: g_rules (structured rules) + g_bdd (scenarios) + specification

The coder receives explicit constraints formatted from g_rules:

```
## ARCHITECTURE RULES (MUST FOLLOW)

### Import Constraints (CRITICAL)
- **domain/** MUST NOT import from: infrastructure/, application/
  - Reason: Domain must be pure with no external dependencies
- **application/** MUST NOT import from: infrastructure/
  - Reason: Application depends on abstractions, not implementations

### Dependency Direction
infrastructure → application → domain
(Outer layers can import inner, never the reverse)

### Folder Structure (CREATE FILES HERE)
**domain**: `src/taskmanager/domain/`
  - Purpose: Pure business logic with no external dependencies
  - Expected files: `entities.py`, `value_objects.py`, `exceptions.py`
...
```

**Output** (`ImplementationResult`):

```json
{
  "files": [
    {
      "path": "src/taskmanager/domain/entities.py",
      "content": "from dataclasses import dataclass\n\n@dataclass\nclass Task:\n    id: str\n    title: str"
    }
  ],
  "summary": "Implemented Task entity with value objects"
}
```

**Guard Validation**:

- Generated code has valid Python syntax
- Architecture tests from g_add pass
- All layer boundaries respected

---

### 6. g_quality - Code Quality Gates

| Component | Description |
|-----------|-------------|
| **Generator** | `IdentityGenerator` (pass-through) |
| **Guard** | `QualityGatesGuard` |
| **LLM Required** | No (deterministic) |
| **Dependencies** | `g_coder` |

**Purpose**: Run code quality tools (mypy, ruff) on the implementation in an isolated temp environment.

**Input**: g_coder artifact (ImplementationResult)

**Output**: Same as input (pass-through)

**Guard Validation**:

- Writes implementation to temp directory
- Runs `mypy src/ --ignore-missing-imports`
- Runs `ruff check src/`
- Both tools must pass for guard to pass
- Temp directory cleaned up after validation

---

### 7. g_arch_validate - Architecture Validation

| Component | Description |
|-----------|-------------|
| **Generator** | `IdentityGenerator` (pass-through) |
| **Guard** | `ArchValidationGuard` |
| **LLM Required** | No (deterministic) |
| **Dependencies** | `g_quality`, `g_add` |

**Purpose**: Run the actual pytest-arch tests from g_add against the implementation.

**Input**: g_coder artifact + g_add artifact (TestSuite)

**Output**: Same as input (pass-through)

**Guard Validation**:

- Writes implementation to temp directory
- Assembles architecture tests from g_add TestSuite
- Runs `pytest tests/test_architecture.py -v`
- All architecture tests must pass
- Temp directory cleaned up after validation

**Key Difference from g_add**:

- g_add *generates* the architecture tests
- g_arch_validate *executes* those tests against real code

---

### 8. g_merge_ready - Final Gate

| Component | Description |
|-----------|-------------|
| **Generator** | `IdentityGenerator` (pass-through) |
| **Guard** | `MergeReadyGuard` |
| **LLM Required** | No |
| **Dependencies** | `g_arch_validate` |

**Purpose**: Composite check that verifies all gates passed, produces final summary.

**Input**: All prior artifacts via dependencies

**Output**: Same as g_coder (final implementation)

**Guard Validation**:

- Checks all required gates have ACCEPTED artifacts
- Required gates: g_config, g_add, g_bdd, g_rules, g_coder, g_quality, g_arch_validate
- Returns summary of all gate statuses

---

## Extensions Used

| Extension | Description |
|-----------|-------------|
| **01 - Versioned Environment** | W_ref and config_ref for workflow integrity |
| **02 - Artifact Extraction** | Predicate-based artifact queries |
| **07 - Incremental Execution** | Skip unchanged steps based on config_ref |

## Usage

```bash
# Run workflow (with incremental execution)
uv run python -m examples.advanced.sdlc_v2.demo run --host http://localhost:11434

# Run full workflow (ignore cache)
uv run python -m examples.advanced.sdlc_v2.demo run-full

# Query artifacts
uv run python -m examples.advanced.sdlc_v2.demo artifacts --status accepted

# Resume from checkpoint
uv run python -m examples.advanced.sdlc_v2.demo resume <checkpoint_id>

# Clean output directory
uv run python -m examples.advanced.sdlc_v2.demo clean
```

## File Structure

```
examples/advanced/sdlc_v2/
├── demo.py              # CLI entry point
├── workflow.json        # Workflow configuration
├── prompts.json         # LLM prompts for each action pair
├── models.py            # Pydantic models for artifacts
├── generators/
│   ├── __init__.py
│   ├── config.py        # ConfigExtractorGenerator
│   ├── add.py           # ADDGenerator (architecture tests)
│   ├── bdd.py           # BDDGenerator (Gherkin scenarios)
│   ├── rules.py         # RulesExtractorGenerator (deterministic)
│   ├── coder.py         # CoderGenerator (implementation)
│   └── identity.py      # IdentityGenerator (pass-through for validation)
├── guards/
│   ├── __init__.py
│   ├── base.py          # TempDirValidationMixin
│   ├── config_guard.py  # ConfigGuard
│   ├── architecture_guard.py  # ArchitectureTestsGuard
│   ├── bdd_guard.py     # BDDGuard
│   ├── rules_guard.py   # RulesGuard
│   ├── tests_guard.py   # AllTestsPassGuard
│   ├── quality_guard.py # QualityGatesGuard (mypy, ruff)
│   ├── arch_validation_guard.py  # ArchValidationGuard (pytest-arch)
│   └── merge_ready_guard.py      # MergeReadyGuard (composite)
├── services/
│   ├── __init__.py
│   ├── extraction.py    # ArtifactExtractionService (Extension 02)
│   ├── file_extractor.py # FileExtractor (post-workflow file writing)
│   └── incremental.py   # IncrementalExecutionService (Extension 07)
└── sample_input/
    ├── architecture.md  # Sample architecture documentation
    └── requirements.md  # Sample requirements documentation
```

## Key Design Decisions

### Why g_rules is Deterministic

The `g_rules` step uses no LLM - it deterministically parses test names and specification text:

1. **Fast**: No LLM latency or cost
2. **Reproducible**: Same input always produces same output
3. **Reliable**: No risk of LLM hallucination

Test name patterns like `test_domain_no_infrastructure_imports` encode the rule directly, making extraction straightforward.

### Why Separate g_add and g_rules

Following single-responsibility principle:

- `g_add` generates **tests** (executable pytest-arch code)
- `g_rules` extracts **rules** (structured constraints for the coder)

This separation allows the coder to receive clear, formatted instructions rather than raw test JSON.

### Gate Hierarchy

The workflow follows a design → implementation → validation pattern:

```
Design Phase:
  g_config → SPECIFICATION_COMPLETE
  g_add    → ARCHITECTURE_TESTS_DEFINED
  g_rules  → ARCHITECTURE_RULES_EXTRACTED
  g_bdd    → BDD_SCENARIOS_DEFINED

Implementation Phase:
  g_coder  → IMPLEMENTATION_COMPLETE

Validation Phase:
  g_quality       → QUALITY_GATES_PASSING
  g_arch_validate → INFRASTRUCTURE_VALIDATED
  g_merge_ready   → READY_TO_MERGE
```

Each gate ensures the previous artifact is valid before proceeding.

### Guards as Sensing-Only Actions

**Design Principle**: Guards should be sensing-only actions that validate without mutating state.

This workflow follows a clear separation of concerns:

| Component | Role | File I/O |
|-----------|------|----------|
| **Generators** | Create artifacts | None (in-memory) |
| **Guards** | Validate artifacts | Read-only |
| **FileExtractor** | Materialize to filesystem | Write (post-workflow) |

File extraction only happens AFTER all guards pass, via the `FileExtractor` service. This ensures:

1. **Pure Guards**: Guards validate but don't mutate state
2. **Reproducible**: Running validation again produces the same result
3. **Clean Workdir**: Files only appear after full workflow success
4. **Testable**: Guards can be unit tested without filesystem setup
