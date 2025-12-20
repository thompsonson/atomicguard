# Part 3: Understanding the Output

Learn how to read and use the generated pytestarch tests.

## The Generated Test File

Open the generated test file:

```bash
cat examples/add/output/tests/architecture/test_gates.py
```

## Anatomy of a Test

Each generated test has three parts:

### 1. The Fixture

```python
from pytestarch import get_evaluable_architecture, Rule
import pytest

@pytest.fixture(scope="module")
def evaluable():
    return get_evaluable_architecture("/project", "/project/src")
```

The `evaluable` fixture:

- Creates an architecture model of your codebase
- First argument: project root directory
- Second argument: source code directory
- `scope="module"` means it's created once per test file

### 2. The Rule

```python
def test_gate1_domain_no_infra_imports(evaluable):
    rule = (
        Rule()
        .modules_that()
        .are_sub_modules_of("domain")
        .should_not()
        .import_modules_that()
        .are_sub_modules_of("infrastructure")
    )
```

The rule uses a fluent API:

- `Rule()` - Start building a rule
- `.modules_that()` - Select modules
- `.are_sub_modules_of("domain")` - Filter to domain modules
- `.should_not()` - Negate the constraint
- `.import_modules_that()` - Look at imports
- `.are_sub_modules_of("infrastructure")` - Filter to infra modules

### 3. The Assertion

```python
    rule.assert_applies(evaluable)
```

This checks the rule against your actual codebase.

## Common Rule Patterns

### Pattern 1: Module Import Constraint

"Module A should not import Module B"

```python
rule = (
    Rule()
    .modules_that()
    .are_sub_modules_of("domain")
    .should_not()
    .import_modules_that()
    .are_sub_modules_of("infrastructure")
)
```

### Pattern 2: Named Module Constraint

"A specific module should not import another specific module"

```python
rule = (
    Rule()
    .modules_that()
    .are_named("myproject.core.entities")
    .should_not()
    .import_modules_that()
    .are_named("myproject.external")
)
```

### Pattern 3: Class Containment

"Classes matching a pattern should be in specific packages"

```python
rule = (
    Rule()
    .classes_that()
    .have_name_matching(".*Entity")
    .should_be_in_packages(["domain.entities"])
)
```

### Pattern 4: Class Naming

"Classes in a package should match a naming pattern"

```python
rule = (
    Rule()
    .classes_that()
    .have_name_matching(".*UseCase|.*Handler")
    .should_be_in_packages(["application"])
)
```

## Running the Tests

### Option 1: Against the Output Directory

```bash
cd examples/add/output
uv run pytest tests/architecture/test_gates.py -v
```

Note: Tests will fail if run from the output directory because there's no actual source code there. The fixture paths (`/project`, `/project/src`) are placeholders.

### Option 2: Integrate into Your Project

1. Copy the generated tests to your project:

```bash
cp examples/add/output/tests/architecture/test_gates.py \
   your-project/tests/architecture/
```

1. Update the fixture paths to match your project:

```python
@pytest.fixture(scope="module")
def evaluable():
    return get_evaluable_architecture(
        "/path/to/your/project",
        "/path/to/your/project/src"
    )
```

1. Run the tests:

```bash
cd your-project
pytest tests/architecture/test_gates.py -v
```

## Understanding Test Failures

When a test fails, pytestarch shows which modules violated the rule:

```
FAILED test_gate1_domain_no_infra_imports
AssertionError:
domain.services.order_service imports infrastructure.database.connection

The following modules violate the architecture rule:
  - domain.services.order_service
    imports: infrastructure.database.connection
```

This tells you:

1. Which module violated the rule (`domain.services.order_service`)
2. What it incorrectly imported (`infrastructure.database.connection`)
3. How to fix it (remove that import, or move the code)

## The Artifact Manifest

ADD also generates a JSON manifest:

```bash
cat examples/add/output/artifacts/objects/*/
```

Or access it programmatically:

```python
import json

with open("examples/add/output/artifacts/...") as f:
    manifest = json.load(f)

print(f"Tests: {manifest['test_count']}")
print(f"Gates covered: {manifest['gates_covered']}")
```

## Exercise: Trace a Rule

1. Open `examples/add/sample_docs/architecture.md`
2. Find "Gate 3: Dependency Direction"
3. Open `examples/add/output/tests/architecture/test_gates.py`
4. Find the corresponding test function
5. Map each line of the rule back to the gate definition

**Question**: Which gate does this test enforce?

```python
def test_gate8_database_access_in_infrastructure(evaluable):
    rule = (
        Rule()
        .modules_that()
        .import_modules_matching("sqlalchemy|pymongo")
        .should_be_sub_modules_of("infrastructure.persistence")
    )
    rule.assert_applies(evaluable)
```

---

**Previous**: [02 - Understanding Input](02-understanding-input.md) | **Next**: [04 - The Pipeline](04-pipeline.md)
