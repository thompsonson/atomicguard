# Part 2: Understanding the Input

Learn how to write architecture documentation that ADD can parse.

## The Sample Documentation

Open the sample architecture documentation:

```bash
cat examples/add/sample_docs/architecture.md
```

This file has three main sections:

1. **Layer Structure** - Visual diagram and description of layers
2. **Architecture Gates** - The rules to enforce
3. **Ubiquitous Language** - Glossary of terms

## Anatomy of a Gate Definition

Each gate follows this structure:

```markdown
### Gate 1: Domain Independence

**Rule**: The domain layer MUST NOT import from infrastructure.

**Rationale**: Domain logic should be pure and framework-agnostic.
It should not depend on databases, APIs, or any external systems.

**Scope**: All modules in `domain/`

**Constraint Type**: dependency
```

Let's break this down:

| Field | Purpose | Example |
|-------|---------|---------|
| **Title** | Unique identifier | "Gate 1: Domain Independence" |
| **Rule** | What must be true | "MUST NOT import from infrastructure" |
| **Rationale** | Why this matters | "Domain logic should be pure..." |
| **Scope** | Where it applies | "All modules in `domain/`" |
| **Constraint Type** | Category | dependency, containment, naming |

## Gate Types

### 1. Dependency Gates

Control what modules can import what:

```markdown
### Gate 1: Domain Independence

**Rule**: The domain layer MUST NOT import from infrastructure.

**Constraint Type**: dependency
```

Generated test:

```python
def test_domain_independence(evaluable):
    rule = (
        Rule()
        .modules_that()
        .are_sub_modules_of("domain")
        .should_not()
        .import_modules_that()
        .are_sub_modules_of("infrastructure")
    )
    rule.assert_applies(evaluable)
```

### 2. Containment Gates

Control where certain classes/modules must live:

```markdown
### Gate 4: Entity Containment

**Rule**: All entity classes MUST be in the `domain.entities` package.

**Constraint Type**: containment
```

Generated test:

```python
def test_entity_containment(evaluable):
    rule = (
        Rule()
        .classes_that()
        .have_name_matching(".*Entity")
        .should_be_in_packages(["domain.entities"])
    )
    rule.assert_applies(evaluable)
```

### 3. Naming Gates

Enforce naming conventions:

```markdown
### Gate 7: Use Case Naming

**Rule**: Use case classes in `application/` MUST follow the pattern `*UseCase` or `*Handler`.

**Constraint Type**: naming
```

Generated test:

```python
def test_use_case_naming(evaluable):
    rule = (
        Rule()
        .classes_that()
        .have_name_matching(".*UseCase|.*Handler")
        .should_be_in_packages(["application"])
    )
    rule.assert_applies(evaluable)
```

## Layer Structure

ADD expects a Clean Architecture / Hexagonal Architecture with these layers:

```
┌─────────────────────────────────────────────┐
│              Infrastructure                 │
│  ┌───────────────────────────────────────┐  │
│  │            Application                │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │           Domain                │  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Dependencies flow inward only:**

- Infrastructure → Application → Domain ✓
- Domain → Infrastructure ✗

## Ubiquitous Language

Include a glossary to help the LLM understand your domain:

```markdown
## Ubiquitous Language

| Term | Definition |
|------|------------|
| Entity | Domain object with a unique identity that persists over time |
| Value Object | Immutable object defined by its attributes, not identity |
| Repository | Abstraction for collection-like access to aggregates |
| Use Case | Single application-level operation |
```

## Exercise: Modify a Gate

1. Open `examples/add/sample_docs/architecture.md`
2. Add a new Gate 9:

```markdown
### Gate 9: No External API Clients in Domain

**Rule**: The domain layer MUST NOT import HTTP clients or API libraries.

**Rationale**: Domain should be pure business logic without I/O concerns.

**Scope**: All modules in `domain/`

**Constraint Type**: dependency
```

1. Re-run ADD:

```bash
uv run python -m examples.add.run --host http://localhost:11434 -v
```

1. Check that 9 tests are now generated:

```bash
grep "def test_" examples/add/output/tests/architecture/test_gates.py | wc -l
```

## Tips for Writing Good Gates

1. **Be specific** - "domain" is better than "core"
2. **Use consistent layer names** - Match your actual package structure
3. **One rule per gate** - Don't combine multiple constraints
4. **Include rationale** - Helps the LLM understand intent
5. **Specify scope** - Tell ADD exactly where the rule applies

---

**Previous**: [01 - Quick Start](01-quickstart.md) | **Next**: [03 - Understanding Output](03-understanding-output.md)
