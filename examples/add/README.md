# ADD (Architecture-Driven Development) Agent

This example demonstrates building a complex generator that internally
orchestrates multiple action pairs using PydanticAI for structured output.

## Tutorial

**New to ADD?** Start with the [step-by-step tutorial](tutorial/00-overview.md):

1. [Quick Start](tutorial/01-quickstart.md) - Run ADD in 5 minutes
2. [Understanding Input](tutorial/02-understanding-input.md) - Architecture documentation format
3. [Understanding Output](tutorial/03-understanding-output.md) - Generated test structure
4. [The Pipeline](tutorial/04-pipeline.md) - How ADD works internally
5. [Customization](tutorial/05-customization.md) - CLI options and configuration
6. [Programmatic Usage](tutorial/06-programmatic.md) - Python API integration
7. [Troubleshooting](tutorial/07-troubleshooting.md) - Common issues and fixes

## What ADD Does

The ADD Agent:

1. **Parses architecture documentation** - Extracts architecture gates,
   layer boundaries, and ubiquitous language from markdown documentation

2. **Generates pytest-arch tests** - Creates Python test code that enforces
   the extracted architecture constraints

3. **Writes test files** - Outputs the generated tests to the filesystem

## Architecture

```
ADDGenerator (GeneratorInterface)
├── ActionPair₁: DocParserGenerator + GatesExtractedGuard
├── ActionPair₂: TestCodeGenerator + TestSyntaxGuard + TestNamingGuard
└── ActionPair₃: FileWriterGenerator + ArtifactStructureGuard
```

From the parent workflow's perspective, `ADDGenerator` appears as a single
atomic generator. Internally, it orchestrates three action pairs with their
own retry loops.

## Requirements

- Python 3.12+
- Ollama running locally
- Model: `qwen2.5-coder:14b` (or configure in ADDGenerator)

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the model
ollama pull qwen2.5-coder:14b

# Start Ollama server
ollama serve
```

## Usage

### Running the Example

```bash
# From the project root
python -m examples.add.run
```

This will:

1. Load the sample architecture documentation from `sample_docs/architecture.md`
2. Run the ADD workflow to generate pytest-arch tests
3. Write output to `examples/add/output/`

### Using ADDGenerator in Your Code

```python
from pathlib import Path
from atomicguard.domain.models import AmbientEnvironment, Context
from atomicguard.infrastructure.persistence import InMemoryArtifactDAG
from examples.add.add_generator import ADDGenerator

# Load your architecture documentation
docs = Path("docs/architecture.md").read_text()

# Create generator
add_generator = ADDGenerator(
    model="ollama:qwen2.5-coder:14b",
    rmax=3,
    workdir=Path("tests/architecture"),
    min_gates=5,
    min_tests=5,
)

# Create context
dag = InMemoryArtifactDAG()
ambient = AmbientEnvironment(repository=dag, constraints="")
context = Context(ambient=ambient, specification=docs)

# Generate
artifact = add_generator.generate(context)

# The artifact contains a JSON manifest of generated files
import json
manifest = json.loads(artifact.content)
print(f"Generated {manifest['test_count']} tests")
```

### Integrating with a Workflow

```python
from atomicguard.application.workflow import Workflow
from atomicguard.application.action_pair import ActionPair
from examples.add.add_generator import ADDGenerator
from examples.add.guards import ArtifactStructureGuard

# ADD appears as a single step in the workflow
workflow = Workflow(rmax=2)

workflow.add_step(
    "g_architecture_tests",
    ActionPair(
        generator=ADDGenerator(min_gates=5),
        guard=ArtifactStructureGuard(min_tests=5),
    ),
)

result = workflow.execute(architecture_docs)
```

## File Structure

```
examples/add/
├── __init__.py           # Package init
├── add_generator.py      # Main ADDGenerator class
├── generators.py         # Internal generators (DocParser, TestCodeGen, FileWriter)
├── guards.py             # Domain-specific guards
├── models.py             # Pydantic models for structured output
├── run.py                # Example runner
├── README.md             # This file
├── prompts.json          # Prompt templates for each action pair
├── workflow.json         # Workflow configuration
├── tutorial/             # Step-by-step tutorial
│   ├── 00-overview.md
│   ├── 01-quickstart.md
│   ├── 02-understanding-input.md
│   ├── 03-understanding-output.md
│   ├── 04-pipeline.md
│   ├── 05-customization.md
│   ├── 06-programmatic.md
│   └── 07-troubleshooting.md
├── output/               # Generated test files (gitignored)
└── sample_docs/
    └── architecture.md   # Sample architecture documentation
```

## Design Decisions

### Why PydanticAI?

PydanticAI provides schema-validated structured output from LLMs. This ensures:

- Type-safe extraction of architecture gates
- Guaranteed schema compliance for test generation
- Clear error messages when output doesn't match schema

### Why Nested Action Pairs?

Each step in the ADD workflow has distinct validation needs:

- Gate extraction needs completeness checks
- Test generation needs syntax and naming validation
- File writing needs structure validation

By using action pairs internally, each step gets its own retry loop with
targeted feedback.

### Why GeneratorInterface?

ADDGenerator implements `GeneratorInterface` so it can be composed into
larger workflows. From the parent workflow's perspective, ADD is atomic -
the internal complexity is hidden.

## Extending ADD

### Custom Gate Types

Extend `ArchitectureGate` in `models.py`:

```python
class ArchitectureGate(BaseModel):
    # ... existing fields ...
    severity: Literal["error", "warning"] = "error"
    auto_fix_hint: str = ""
```

### Different Test Frameworks

Replace `TestCodeGenerator` with one targeting your preferred framework:

- ArchUnit (for JVM projects)
- Custom AST-based validators
- Import-linter rules

### Alternative LLM Backends

Change the model string in ADDGenerator:

```python
# OpenAI
add_generator = ADDGenerator(model="openai:gpt-4")

# Anthropic
add_generator = ADDGenerator(model="anthropic:claude-3-5-sonnet")

# Local Ollama with different model
add_generator = ADDGenerator(model="ollama:codellama:34b")
```
