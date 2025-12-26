# Part 4: The Four-Stage Pipeline

Understand how ADD works internally.

## Paper Definitions

ADD implements the paper's **Hierarchical Context Composition**:

```
C_total = ⟨ℰ, C_local, H_feedback⟩
ℰ (Ambient Environment) = ⟨ℛ, Ω⟩
```

| Symbol | Name | Implementation | Description |
|--------|------|----------------|-------------|
| **Ψ** | Specification | `context.specification` | Input documentation (immutable) |
| **Ω** | Global Constraints | `context.ambient.constraints` | Project-wide config (source_root) |
| **ℛ** | Repository | `context.ambient.repository` | ArtifactDAG storing all generated artifacts |
| **H** | Feedback History | `context.feedback_history` | Accumulated guard rejections for retry |

## Overview

ADD is a **composite generator** that orchestrates four internal action pairs:

```
┌──────────────────────────────────────────────────────────────────┐
│                      ADDGenerator                                │
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │  Stage 0    │     │  Stage 1    │     │  Stage 2    │        │
│  │ ConfigExtr  │────▶│ DocParser   │────▶│ TestCodeGen │───┐    │
│  └─────────────┘     └─────────────┘     └─────────────┘   │    │
│        │                   │                   │            │    │
│        ▼                   ▼                   ▼            │    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │    │
│  │ ConfigGuard │     │ GatesGuard  │     │ Composite   │   │    │
│  │  (sets Ω)   │     │             │     │   Guard     │   │    │
│  └─────────────┘     └─────────────┘     └─────────────┘   │    │
│                                                            │    │
│  ┌─────────────────────────────────────────────────────────┘    │
│  │                                                               │
│  │  ┌─────────────┐                                              │
│  └─▶│  Stage 3    │                                              │
│     │ FileWriter  │                                              │
│     └─────────────┘                                              │
│           │                                                      │
│           ▼                                                      │
│     ┌─────────────┐                                              │
│     │ Structure   │                                              │
│     │   Guard     │                                              │
│     └─────────────┘                                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Stage 0: Config Extraction (Ω)

**File**: `generators.py` → `ConfigExtractorGenerator`

### Purpose

Extracts global constraints (Ω) from the documentation before other stages run.
This is the first action pair, ensuring Ω is available to ALL subsequent generators.

### Input

- Ψ (architecture documentation)

### Process

1. LLM reads the documentation
2. Extracts `ProjectConfig` (source_root, package_name) using PydanticAI
3. ADDGenerator creates NEW Context with Ω in `ambient.constraints`

### Output

```json
{
  "source_root": "src/myapp",
  "package_name": "myapp"
}
```

### Guard: ConfigGuard

**File**: `guards.py` → `ConfigGuard`

Validates:

- Valid JSON structure
- `source_root` is non-empty

### Why a New Context?

Context is frozen (immutable). After extracting Ω, ADDGenerator creates
a new Context with populated `ambient.constraints`:

```python
updated_ambient = AmbientEnvironment(
    repository=context.ambient.repository,
    constraints=config_result.model_dump_json(),  # Ω as JSON string
)
context = Context(ambient=updated_ambient, ...)
```

## Stage 1: Gate Extraction

**File**: `generators.py` → `DocParserGenerator`

### Input

- Architecture documentation (markdown text)

### Process

1. LLM reads the documentation
2. Extracts structured gates using PydanticAI
3. Returns `GatesExtractionResult` JSON

### Output

```json
{
  "gates": [
    {
      "gate_id": "Gate1",
      "layer": "domain",
      "constraint_type": "dependency",
      "description": "Domain must not import infrastructure",
      "source_reference": "Architecture Gates > Gate 1"
    }
  ],
  "ubiquitous_language": [...],
  "layer_boundaries": [...]
}
```

### Guard: GatesExtractedGuard

**File**: `guards.py` → `GatesExtractedGuard`

Validates:

- Minimum number of gates extracted
- No duplicate gate IDs
- Valid JSON schema

```python
guard = GatesExtractedGuard(min_gates=3)
```

## Stage 2: Test Generation

**File**: `generators.py` → `TestCodeGenerator`

### Input

- Extracted gates from Stage 1

### Process

1. LLM receives the gates
2. Generates pytestarch test code for each gate
3. Returns `TestSuite` JSON

### Output

```json
{
  "module_docstring": "Test suite for architecture gates",
  "imports": [
    "from pytestarch import get_evaluable_architecture, Rule",
    "import pytest"
  ],
  "fixtures": [
    "@pytest.fixture(scope=\"module\")\ndef evaluable():..."
  ],
  "tests": [
    {
      "gate_id": "Gate1",
      "test_name": "test_gate1_domain_independence",
      "test_code": "def test_gate1_domain_independence(evaluable):..."
    }
  ]
}
```

### Guard: CompositeGuard

**File**: `guards.py` + `add_generator.py`

Three guards run in sequence:

```python
guard = CompositeGuard(
    TestSyntaxGuard(),      # 1. Valid Python syntax
    TestNamingGuard(),      # 2. Test names start with test_
    PytestArchAPIGuard(),   # 3. Valid pytestarch API
)
```

#### TestSyntaxGuard

- Parses code with `ast.parse()`
- Catches syntax errors

#### TestNamingGuard

- Verifies all test names start with `test_`
- Checks for duplicate names

#### PytestArchAPIGuard

- **The key innovation**: Actually imports pytestarch and executes the code
- Catches hallucinated API methods like `modules_in()` (doesn't exist)
- Uses the real library as source of truth

```python
# This catches:
# AttributeError: 'Rule' object has no attribute 'modules_in'
```

## Stage 3: File Writing

**File**: `generators.py` → `FileWriterGenerator`

### Input

- `TestSuite` from Stage 2

### Process

1. Assembles test code into a Python file
2. Writes files to disk
3. Returns `ArtifactManifest` JSON

### Output

```json
{
  "files": [
    {"path": "tests/architecture/test_gates.py", "content": "..."},
    {"path": "tests/architecture/__init__.py", "content": ""}
  ],
  "test_count": 8,
  "gates_covered": ["Gate1", "Gate2", ...]
}
```

### Guard: ArtifactStructureGuard

**File**: `guards.py` → `ArtifactStructureGuard`

Validates:

- Minimum number of tests
- At least one file in manifest
- Non-empty content in test files

## Retry Logic

Each stage has its own retry loop:

```
┌─────────────────────────────────────────┐
│              Action Pair                 │
│                                          │
│   ┌──────────┐        ┌──────────┐      │
│   │Generator │───────▶│  Guard   │      │
│   └──────────┘        └──────────┘      │
│        ▲                   │            │
│        │                   ▼            │
│        │    ┌──────────────────┐        │
│        └────│ Failed? Retry    │        │
│             │ with feedback    │        │
│             └──────────────────┘        │
│                                          │
└─────────────────────────────────────────┘
```

Configuration:

- `rmax=3` means 4 attempts total (1 initial + 3 retries)
- Feedback from failed guard is passed back to generator
- If all retries exhausted, raises `EscalationRequired`

## Data Flow Example

```
1. Input: Ψ = "Architecture documentation with source_root: src/myapp"

2. ConfigExtractor extracts (Stage 0):
   Ω = {source_root: "src/myapp", package_name: "myapp"}
   → Stored in context.ambient.constraints

3. DocParser extracts (Stage 1):
   {gate_id: "Gate1", constraint_type: "dependency", ...}
   → Artifact stored in ℛ (ArtifactDAG)
   → Added to context.dependency_artifacts["gates"]

4. TestCodeGen generates (Stage 2):
   - Reads gates from context.dependency_artifacts["gates"]
   - Reads source_root from Ω for fixture
   "def test_gate1(evaluable):
       rule = Rule().modules_that()..."
   → Artifact stored in ℛ
   → Added to context.dependency_artifacts["test_suite"]

5. Guards validate:
   ✓ Syntax valid
   ✓ Name starts with test_
   ✓ pytestarch API correct

6. FileWriter creates (Stage 3):
   - Reads test_suite from context.dependency_artifacts["test_suite"]
   tests/architecture/test_gates.py
```

### Artifact Passing (Paper-Aligned)

Per the paper, generators access prior artifacts via `context.dependency_artifacts`:

```python
# In ADDGenerator.generate():
# After AP1 completes, add artifact ID to dependency_artifacts for AP2
context = Context(
    ambient=context.ambient,
    specification=context.specification,
    dependency_artifacts=(("gates", gates_artifact.artifact_id),),  # ID stored, artifact in ℛ
)

# In TestCodeGenerator.generate():
# Read gates from dependency_artifacts (retrieve from ℛ)
gates_id = context.get_dependency("gates")
if gates_id:
    gates_artifact = context.ambient.repository.get_artifact(gates_id)
    gates = GatesExtractionResult.model_validate_json(gates_artifact.content)
```

## Viewing the Artifacts

Each stage stores its artifacts:

```bash
ls examples/add/output/artifacts/objects/
```

Artifacts contain:

- Generated content
- Context snapshot
- Attempt number
- Previous attempt ID (for provenance)

## Exercise: Follow the Pipeline

Run ADD with verbose logging:

```bash
uv run python -m examples.add.run --host http://localhost:11434 -v 2>&1 | tee run_output.txt
```

Then answer:

1. How many attempts did Stage 2 need?
2. What feedback was given on failed attempts?
3. What was the final artifact ID?

---

**Previous**: [03 - Understanding Output](03-understanding-output.md) | **Next**: [05 - Customization](05-customization.md)
