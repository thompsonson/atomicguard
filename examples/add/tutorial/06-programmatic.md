# Part 6: Programmatic Usage

Integrate ADD into your Python code.

## Basic Usage

### Minimal Example

```python
import json
from pathlib import Path

from atomicguard.domain.models import AmbientEnvironment, Context
from examples.add.add_generator import ADDGenerator

# Load your documentation
docs = Path("docs/architecture.md").read_text()

# Create generator
generator = ADDGenerator(
    model="ollama:qwen2.5-coder:14b",
    base_url="http://localhost:11434/v1",
    workdir=Path("tests/architecture"),
)

# Create context
ambient = AmbientEnvironment(repository=None, constraints="")
context = Context(ambient=ambient, specification=docs)

# Generate
artifact = generator.generate(context)

# Parse results
manifest = json.loads(artifact.content)
print(f"Generated {manifest['test_count']} tests")
```

## With Artifact Persistence

Store artifacts for debugging and auditing:

```python
from atomicguard.infrastructure.persistence import FilesystemArtifactDAG

# Create artifact storage
dag = FilesystemArtifactDAG("./artifacts")

# Create generator with persistence
generator = ADDGenerator(
    model="ollama:qwen2.5-coder:14b",
    base_url="http://localhost:11434/v1",
    workdir=Path("tests/architecture"),
    artifact_dag=dag,  # Enable persistence
)

# Generate
artifact = generator.generate(context)

# Artifacts are now stored in ./artifacts/objects/
```

## With Custom Prompts

Load prompts from file or define inline:

```python
from atomicguard.domain.prompts import PromptTemplate

# Define prompts
prompts = {
    "gates_extraction": PromptTemplate(
        role="You are an expert architecture analyzer.",
        constraints="Extract all gates. Be thorough.",
        task="Extract gates from the documentation.",
        feedback_wrapper="Error: {feedback}\n\nPlease fix.",
    ),
    "test_generation": PromptTemplate(
        role="You are a Python test generator.",
        constraints="Use pytestarch Rule API only.",
        task="Generate tests for each gate.",
        feedback_wrapper="Test failed: {feedback}\n\nFix it.",
    ),
}

# Create generator with custom prompts
generator = ADDGenerator(
    model="ollama:qwen2.5-coder:14b",
    base_url="http://localhost:11434/v1",
    workdir=Path("tests/architecture"),
    prompts=prompts,
)
```

## Handling Errors

```python
from atomicguard.domain.exceptions import EscalationRequired, RmaxExhausted

try:
    artifact = generator.generate(context)
    manifest = json.loads(artifact.content)
    print(f"Success! Generated {manifest['test_count']} tests")

except EscalationRequired as e:
    print(f"Fatal error: {e.feedback}")
    print(f"Last artifact: {e.artifact.artifact_id}")

except RmaxExhausted as e:
    print(f"Failed after {len(e.provenance)} attempts")
    for i, (artifact, feedback) in enumerate(e.provenance, 1):
        print(f"  Attempt {i}: {feedback}")
```

## Using with DualStateAgent

Wrap ADD in an agent for outer retry logic:

```python
from atomicguard.application.action_pair import ActionPair
from atomicguard.application.agent import DualStateAgent
from examples.add.guards import ArtifactStructureGuard

# Create action pair
action_pair = ActionPair(
    generator=ADDGenerator(
        model="ollama:qwen2.5-coder:14b",
        base_url="http://localhost:11434/v1",
        rmax=3,  # Inner retries
    ),
    guard=ArtifactStructureGuard(min_tests=5),
)

# Create agent
agent = DualStateAgent(
    action_pair=action_pair,
    artifact_dag=dag,
    rmax=2,  # Outer retries
    constraints="Generate architecture tests",
)

# Execute
artifact = agent.execute(docs)
```

## Accessing Internal Generators

Use the individual generators directly:

```python
from examples.add.generators import (
    ConfigExtractorGenerator,
    DocParserGenerator,
    TestCodeGenerator,
    FileWriterGenerator,
)
from examples.add.models import GatesExtractionResult, ProjectConfig
from atomicguard.domain.models import AmbientEnvironment, Context

# Stage 0: Extract config (Ω)
config_gen = ConfigExtractorGenerator(
    model="ollama:qwen2.5-coder:14b",
    base_url="http://localhost:11434/v1",
)
config_artifact = config_gen.generate(context)
config = ProjectConfig.model_validate_json(config_artifact.content)
print(f"Extracted Ω: source_root={config.source_root}")

# Update context with Ω
updated_ambient = AmbientEnvironment(
    repository=context.ambient.repository,
    constraints=config.model_dump_json(),
)
context = Context(ambient=updated_ambient, specification=context.specification)

# Stage 1: Extract gates
parser = DocParserGenerator(
    model="ollama:qwen2.5-coder:14b",
    base_url="http://localhost:11434/v1",
)
gates_artifact = parser.generate(context)
gates = GatesExtractionResult.model_validate_json(gates_artifact.content)
print(f"Extracted {len(gates.gates)} gates")

# Stage 2: Generate tests
# Per paper: Pass prior artifacts via context.dependencies
context_with_gates = Context(
    ambient=context.ambient,
    specification=context.specification,
    dependencies=(("gates", gates_artifact),),  # Paper-aligned!
)

test_gen = TestCodeGenerator(
    model="ollama:qwen2.5-coder:14b",
    base_url="http://localhost:11434/v1",
)
test_artifact = test_gen.generate(context_with_gates)
```

## Accessing Guards

Validate artifacts manually:

```python
from examples.add.guards import (
    ConfigGuard,
    GatesExtractedGuard,
    TestSyntaxGuard,
    TestNamingGuard,
    PytestArchAPIGuard,
)
from atomicguard.guards import CompositeGuard

# Config guard (Stage 0)
config_guard = ConfigGuard()
result = config_guard.validate(config_artifact)
if result.passed:
    print(f"Config valid: {result.feedback}")
else:
    print(f"Config invalid: {result.feedback}")

# Single guard
guard = GatesExtractedGuard(min_gates=5)
result = guard.validate(artifact)

if result.passed:
    print("Valid!")
else:
    print(f"Failed: {result.feedback}")

# Composite guard
guard = CompositeGuard(
    TestSyntaxGuard(),
    TestNamingGuard(),
    PytestArchAPIGuard(),
)
result = guard.validate(artifact)
```

## Integration Example: CI Pipeline

```python
#!/usr/bin/env python3
"""Generate architecture tests in CI."""

import json
import sys
from pathlib import Path

from examples.add.add_generator import ADDGenerator
from atomicguard.domain.models import AmbientEnvironment, Context
from atomicguard.domain.exceptions import EscalationRequired


def main():
    # Load docs from repo
    docs = Path("docs/architecture.md").read_text()

    # Configure for CI
    generator = ADDGenerator(
        model="ollama:qwen2.5-coder:14b",
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434/v1"),
        workdir=Path("tests/architecture"),
        min_gates=5,
        min_tests=5,
        rmax=5,
    )

    context = Context(
        ambient=AmbientEnvironment(repository=None, constraints=""),
        specification=docs,
    )

    try:
        artifact = generator.generate(context)
        manifest = json.loads(artifact.content)

        print(f"✓ Generated {manifest['test_count']} tests")
        print(f"✓ Covered gates: {manifest['gates_covered']}")

        # Run the generated tests
        import subprocess
        result = subprocess.run(
            ["pytest", "tests/architecture/", "-v"],
            capture_output=True,
        )

        if result.returncode != 0:
            print("✗ Architecture tests failed!")
            sys.exit(1)

        print("✓ All architecture tests passed!")

    except EscalationRequired as e:
        print(f"✗ Generation failed: {e.feedback}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## Exercise: Build a Wrapper Script

Create a script that:

1. Accepts a documentation path as argument
2. Generates tests to a specified output directory
3. Runs the generated tests with pytest
4. Reports success/failure

```bash
./generate_arch_tests.py docs/architecture.md tests/architecture/
```

---

**Previous**: [05 - Customization](05-customization.md) | **Next**: [07 - Troubleshooting](07-troubleshooting.md)
