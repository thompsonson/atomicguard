# AtomicGuard

[![CI](https://github.com/thompsonson/atomicguard/actions/workflows/ci.yml/badge.svg)](https://github.com/thompsonson/atomicguard/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/thompsonson/atomicguard/branch/main/graph/badge.svg)](https://codecov.io/gh/thompsonson/atomicguard)
[![PyPI version](https://badge.fury.io/py/atomicguard.svg)](https://badge.fury.io/py/atomicguard)
[![Python versions](https://img.shields.io/pypi/pyversions/atomicguard.svg)](https://pypi.org/project/atomicguard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Dual-State Agent Framework for reliable LLM code generation.

> **New to AtomicGuard?** Start with the [Getting Started Guide](docs/getting-started.md).

**Paper:** *Managing the Stochastic: Foundations of Learning in Neuro-Symbolic Systems for Software Engineering* (Thompson, 2025)

## Overview

AtomicGuard implements guard-validated generation loops that dramatically improve LLM reliability. The core abstraction is the **Atomic Action Pair** ⟨agen, G⟩ — coupling each generation action with a validation guard.

Key results (Yi-Coder 9B, n=50):

| Task | Baseline | Guarded | Improvement |
|------|----------|---------|-------------|
| Template | 35% | 90% | +55pp |
| Password | 82% | 98% | +16pp |
| LRU Cache | 94% | 100% | +6pp |

## Installation

```bash
# From PyPI
pip install atomicguard

# From source
git clone https://github.com/thompsonson/atomicguard.git
cd atomicguard
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,test]"
```

## Quick Start

```python
from atomicguard import (
    OllamaGenerator, SyntaxGuard, TestGuard,
    CompositeGuard, ActionPair, DualStateAgent,
    InMemoryArtifactDAG
)

# Setup
generator = OllamaGenerator(model="qwen2.5-coder:7b")
guard = CompositeGuard([SyntaxGuard(), TestGuard("assert add(2, 3) == 5")])
action_pair = ActionPair(generator=generator, guard=guard)
agent = DualStateAgent(action_pair, InMemoryArtifactDAG(), rmax=3)

# Execute
artifact = agent.execute("Write a function that adds two numbers")
print(artifact.content)
```

See [examples/](examples/) for more detailed usage, including a [mock example](examples/basic_mock.py) that works without an LLM.

## Benchmarks

Run the simulation from the paper:

```bash
python -m benchmarks.simulation --model yi-coder:9b --trials 50 --task all --output results/results.db --format sqlite

# Generate report
python -m benchmarks.simulation --visualize --output results/results.db --format sqlite
```

## Project Structure

```
atomicguard/
├── src/atomicguard/     # Core library
├── benchmarks/          # Simulation code
├── docs/design/         # Design documents
├── examples/            # Usage examples
└── results/             # Generated reports & charts
```

## Citation

If you use this framework in your research, please cite the paper:

> Thompson, M. (2025). Managing the Stochastic: Foundations of Learning in Neuro-Symbolic Systems for Software Engineering. arXiv preprint arXiv:2512.20660.

```bibtex
@misc{thompson2025managing,
  title={Managing the Stochastic: Foundations of Learning in Neuro-Symbolic Systems for Software Engineering},
  author={Thompson, Matthew},
  year={2025},
  eprint={2512.20660},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2512.20660}
}
```

## License

MIT
