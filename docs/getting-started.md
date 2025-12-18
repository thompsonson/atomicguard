# Getting Started with AtomicGuard

This guide will help you install AtomicGuard and run your first guard-validated generation.

## Prerequisites

- Python 3.12 or higher
- For real LLM usage: [Ollama](https://ollama.ai/) or an OpenAI-compatible API endpoint

## Installation

### From PyPI

```bash
pip install atomicguard
```

### From Source

```bash
git clone https://github.com/thompsonson/atomicguard.git
cd atomicguard
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,test]"
```

## Verify Installation

```python
import atomicguard
print(atomicguard.__version__)
```

## Your First Example

### Without an LLM (MockGenerator)

The quickest way to understand AtomicGuard is with the mock generator. See [examples/basic_mock.py](../examples/basic_mock.py) for a complete working example.

```python
from atomicguard import (
    MockGenerator,
    SyntaxGuard,
    ActionPair,
    DualStateAgent,
    InMemoryArtifactDAG,
)

# MockGenerator returns predefined responses
generator = MockGenerator(responses=["def add(a, b):\n    return a + b"])
guard = SyntaxGuard()
action_pair = ActionPair(generator=generator, guard=guard)
agent = DualStateAgent(action_pair, InMemoryArtifactDAG(), rmax=3)

artifact = agent.execute("Write a function that adds two numbers")
print(artifact.content)
```

### With Ollama (Real LLM)

For real LLM-powered generation, see [examples/basic_ollama.py](../examples/basic_ollama.py).

```python
from atomicguard import (
    OllamaGenerator,
    SyntaxGuard,
    TestGuard,
    CompositeGuard,
    ActionPair,
    DualStateAgent,
    InMemoryArtifactDAG,
)

# Requires: ollama pull qwen2.5-coder:7b
generator = OllamaGenerator(model="qwen2.5-coder:7b")
guard = CompositeGuard([SyntaxGuard(), TestGuard("assert add(2, 3) == 5")])
action_pair = ActionPair(generator=generator, guard=guard)
agent = DualStateAgent(action_pair, InMemoryArtifactDAG(), rmax=3)

artifact = agent.execute("Write a function that adds two numbers")
print(artifact.content)
```

## Key Concepts

- **Generator**: Produces artifacts (code) from prompts
- **Guard**: Validates artifacts against criteria (syntax, tests, etc.)
- **ActionPair**: Couples a generator with a guard (the atomic unit)
- **DualStateAgent**: Executes action pairs with retry logic up to `rmax` attempts
- **ArtifactDAG**: Stores artifacts and their relationships

## Next Steps

- Explore [examples/](../examples/) for more detailed usage patterns
- Read the [design documentation](design/) for architectural details
- Check [CONTRIBUTING.md](../CONTRIBUTING.md) to add custom guards

## Running Benchmarks

```bash
python -m benchmarks.workflow_benchmark --model qwen2.5-coder:7b --task tdd_stack --trials 10
```
