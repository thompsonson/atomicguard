# AtomicGuard

[![CI](https://github.com/thompsonson/atomicguard/actions/workflows/ci.yml/badge.svg)](https://github.com/thompsonson/atomicguard/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/thompsonson/atomicguard/branch/main/graph/badge.svg)](https://codecov.io/gh/thompsonson/atomicguard)
[![PyPI version](https://badge.fury.io/py/atomicguard.svg)](https://badge.fury.io/py/atomicguard)
[![Python versions](https://img.shields.io/pypi/pyversions/atomicguard.svg)](https://pypi.org/project/atomicguard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Dual-State Agent Framework for reliable LLM code generation.

**Paper:** *Managing the Stochastic: Foundations of Learning in Neuro-Symbolic Systems for Software Engineering* (Thompson, 2025) — [arXiv:2512.20660](https://arxiv.org/abs/2512.20660)

## Why AtomicGuard?

AI agents hallucinate. Worse, those hallucinations **compound** — each generation builds on the last, and errors propagate through the workflow.

AtomicGuard solves this by **decomposing goals** into small measurable tasks and enforcing **Bounded Indeterminacy**: the LLM generates content, but a deterministic state machine controls the logic. Every generation is validated before the workflow advances.

| Challenge | Solution |
|-----------|----------|
| **Safety** | Dual-State Architecture & Atomic Action Pairs |
| **State** | Versioned Repository Items & Configuration Snapshots |
| **Scale** | Multi-Agent Coordination via Shared DAG |
| **Improvement** | Continuous Learning from Guard Verdicts |

> [Learn more about the architecture](docs/design/architecture.md)

## Overview

The core abstraction is the **Atomic Action Pair** `A = ⟨a_gen, G⟩` — coupling each generation action with a validation guard. The workflow state never advances unless the guard passes.

| Layer | Controller | Nature |
|-------|------------|--------|
| **Content** | LLM (Generator) | Stochastic |
| **Logic** | State Machine (Workflow) | Deterministic |
| **Validation** | Guards | Deterministic |

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

See [examples/](examples/) for more detailed usage, including a [mock example](examples/basics/01_mock.py) that works without an LLM.

## Core Architecture

### Domain Layer

Immutable domain entities implementing the formal model (Definitions 1-9):

- **Artifact** — Immutable record of generated content with full provenance
- **Context** — Hierarchical composition: `C = ⟨Ψ, Ω, H, ℛ⟩` (specification, constraints, feedback, dependencies)
- **GuardResult** — Boolean pass/fail with feedback: `G(α, C) → {⊤/⊥, φ}`
- **PromptTemplate** — Content-agnostic prompt structure (ROLE, CONSTRAINTS, CONTEXT, HISTORY, TASK)

### Application Layer

- **ActionPair** — Couples generator + guard: `A = ⟨a_gen, G⟩`
- **DualStateAgent** — Executes guard-validated retry loop with rmax limit
- **Workflow** — Multi-step orchestration with DAG dependencies
- **CheckpointService / ResumeService** — Pause and resume workflows after human intervention

### Guards

| Type | Guards | Profile |
|------|--------|---------|
| **Static** | `SyntaxGuard`, `ImportGuard` | Pure AST analysis, O(n) |
| **Dynamic** | `TestGuard`, `DynamicTestGuard` | Subprocess execution |
| **Interactive** | `HumanReviewGuard` | Human-in-loop validation |
| **Composite** | `CompositeGuard`, `SequentialGuard`, `ParallelGuard` | Guard composition (Definitions 38-43) |

Aggregation policies: `ALL_PASS` (default), `ANY_PASS`, `MAJORITY_PASS`.

### Infrastructure

**LLM Adapters:**
- `OllamaGenerator` — OpenAI-compatible API (Ollama, vLLM, etc.)
- `HuggingFaceGenerator` — HuggingFace Inference API
- `MockGenerator` — Deterministic responses for testing

**Persistence:**
- `InMemoryArtifactDAG` — Non-persistent (dev/testing)
- `FilesystemArtifactDAG` — JSON files on disk with full provenance

## Extensions

Eight formal extensions (Definitions 10-43) build on the base framework:

| Extension | Definitions | Description |
|-----------|-------------|-------------|
| [01 — Versioned Environment](docs/design/extensions/01_versioned_environment.md) | 10-16 | Repository items with configuration snapshots (W_ref), checkpointing |
| [02 — Artifact Extraction](docs/design/extensions/02_artifact_extraction.md) | 17-18 | Read-only queries over the repository: `E: ℛ × Φ → 2^ℛ` |
| [03 — Multi-Agent Workflows](docs/design/extensions/03_multi_agent_workflows.md) | 19-20 | Coordination via shared DAG (Blackboard Pattern) |
| [04 — Learning Loop](docs/design/extensions/04_learning_loop.md) | 21-24 | Training traces from guard verdicts |
| [05 — Learning Implementation](docs/design/extensions/05_learning_implementation.md) | — | Practical guide (Unsloth/LoRA fine-tuning) |
| [06 — Generated Workflows](docs/design/extensions/06_generated_workflows.md) | 25-32 | Workflows as generated artifacts (Planner ActionPair) |
| [07 — Incremental Execution](docs/design/extensions/07_incremental_execution.md) | 33-37 | Skip unchanged steps via config fingerprints (Ψ_ref) |
| [08 — Composite Guards](docs/design/extensions/08_composite_guards.md) | 38-43 | Sequential/parallel guard composition |

> See [docs/design/extensions/README.md](docs/design/extensions/README.md) for the full reading order and dependency graph.

## G_plan Benchmark — Contingent Planning for LLMs

The `g_plan_benchmark` validates the G_plan taxonomy for plan validation and implements a **decomposed contingent planning pipeline** — an instance of Extension 06 (Generated Workflows).

### The G_plan Taxonomy

Three guard rigor levels for validating workflow plans:

| Rigor | Predicates | Complexity | Guard |
|-------|-----------|------------|-------|
| **Minimal** | `parseable ∧ is_dag ∧ guard_exists ∧ budget_defined` | O(V + E) | `MinimalPlanGuard` |
| **Medium** | Minimal + `reachable ∧ precond_satisfiable ∧ path_exists` | O(V × L) | `MediumPlanGuard` |
| **Expansive** | Medium + `∀π: terminates ∧ safe ∧ invariant_holds` | O(R^K) | `ExpansivePlanGuard` |

**Defect detection** (100 trials per defect): Minimal catches 3/8 (38%), Medium catches 8/8 (100%), Expansive 8/8 (100%) with exponential cost. **Medium is necessary and sufficient for production use.**

### Decomposed Planning Pipeline

Plan generation is decomposed into individually guarded steps, each compressing the search space for the next. The pipeline implements `g_analysis → g_recon → g_strategy → g_plan`:

```
g_analysis ──→ g_recon ──→ g_strategy ──→ g_plan_full
(classify)    (extract)    (select S1-S5)  (generate plan)
```

Each step is its own action pair with its own guard — if the problem classifier produces invalid output, it gets caught and retried before the planner runs. Context flows via `Context.amend(delta_constraints=...)`, enriching each subsequent prompt.

**Pipeline steps:**

| Step | Guard | Purpose | Output |
|------|-------|---------|--------|
| `g_analysis` | `AnalysisGuard` | Classify problem type, language, severity | Problem classification JSON |
| `g_recon` | `ReconGuard` | Extract files, stack traces, APIs, test references | Codebase reconnaissance JSON |
| `g_strategy` | `StrategyGuard` | Select resolution strategy (S1-S5) | Strategy selection JSON |
| `g_plan_full` | `MediumPlanGuard` | Generate plan conditioned on all prior steps | Workflow plan JSON |

**Strategy vocabulary:**

| ID | Name | Suited For |
|----|------|-----------|
| `S1_locate_and_fix` | Locate and Fix | Bug fixes |
| `S2_tdd_feature` | TDD Feature | New features |
| `S3_refactor_safely` | Refactor Safely | Refactoring |
| `S4_profile_and_optimize` | Profile and Optimize | Performance |
| `S5_investigate_first` | Investigate First | Unclear problems |

**Three pipeline modes** are available via `--pipeline`:

| Mode | Steps | LLM Calls (happy path) | Description |
|------|-------|------------------------|-------------|
| `single` | 1 | 1 | Single-shot plan generation |
| `classify-then-plan` | 2 | 2 | Classify problem, then generate conditioned plan |
| `full` | 4 | 4 | Full decomposition: analysis → recon → strategy → plan |

### Running the Benchmark

```bash
# Deterministic validation (no LLM required)
uv run python -m examples.advanced.g_plan_benchmark.demo validate
uv run python -m examples.advanced.g_plan_benchmark.demo validate --from-workflow

# Defect detection benchmark
uv run python -m examples.advanced.g_plan_benchmark.demo benchmark --trials 100

# Complexity cliff measurement
uv run python -m examples.advanced.g_plan_benchmark.demo complexity --trials 100

# LLM epsilon estimation (requires Ollama or HuggingFace)
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon --trials 20

# Full decomposed pipeline
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon \
    --trials 20 --pipeline full --model qwen2.5-coder:14b

# Classify-then-plan pipeline
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon \
    --trials 20 --pipeline classify-then-plan

# HuggingFace backend
uv run python -m examples.advanced.g_plan_benchmark.demo epsilon \
    --trials 20 --backend huggingface --model Qwen/Qwen2.5-Coder-32B-Instruct
```

Epsilon estimation measures:
- **epsilon-hat** per rigor level (Minimal / Medium / Expansive)
- **95% Wilson confidence intervals**
- **E[attempts]** = 1/epsilon (expected retries for a valid plan)
- **Per-step pass rates** (analysis, recon, strategy) in multi-step pipelines
- **Common failure modes** — frequency analysis of guard rejection reasons

> See [examples/advanced/g_plan_benchmark/README.md](examples/advanced/g_plan_benchmark/README.md) for full details.

## Examples

| Example | Directory | LLM Required | Demonstrates |
|---------|-----------|-------------|--------------|
| Mock agent | `examples/basics/01_mock.py` | No | Core concepts, retry loop, provenance |
| Ollama agent | `examples/basics/02_ollama.py` | Yes | Real LLM, CompositeGuard |
| Versioned env | `examples/basics/05_versioned_env.py` | No | Extension 01: W_ref |
| Extraction | `examples/basics/06_extraction.py` | No | Extension 02: repository queries |
| Multi-agent | `examples/basics/07_multiagent.py` | No | Extension 03: shared DAG coordination |
| Incremental | `examples/basics/08_incremental.py` | No | Extension 07: config fingerprints |
| TDD workflows | `examples/tdd/` | No | Guard composition, human review |
| Checkpoint/resume | `examples/checkpoint/` | Mixed | 5 complexity levels, human-in-loop |
| Full SDLC | `examples/advanced/sdlc_v2/` | Yes | Multi-agent SDLC pipeline |
| G_plan benchmark | `examples/advanced/g_plan_benchmark/` | Mixed | Plan validation, contingent planning |
| GUI tools | `examples/advanced/gui/` | No | Artifact viewer, workflow monitor |

## Project Structure

```
atomicguard/
├── src/atomicguard/              # Core library
│   ├── domain/                   #   Pure domain: models, interfaces, prompts
│   ├── application/              #   Orchestration: ActionPair, Agent, Workflow
│   ├── guards/                   #   Static, dynamic, interactive, composite
│   └── infrastructure/           #   LLM adapters, persistence, registry
├── examples/
│   ├── basics/                   #   Getting started (no LLM for most)
│   ├── tdd/                      #   TDD workflow patterns
│   ├── checkpoint/               #   Checkpoint/resume (5 levels)
│   └── advanced/
│       ├── g_plan_benchmark/     #   Plan validation & contingent planning
│       ├── sdlc_v2/              #   Full SDLC pipeline
│       └── gui/                  #   Artifact viewer, workflow monitor
├── tests/                        #   Comprehensive test suite (632+ tests)
├── benchmarks/                   #   Paper simulation benchmarks
└── docs/design/                  #   Architecture, extensions, ADRs
```

## LLM Backends

AtomicGuard supports multiple LLM backends. Each generator implements `GeneratorInterface` and can be swapped in with no other code changes.

### Ollama (local or cloud)

Uses the OpenAI-compatible API. Works with any Ollama-served model:

```python
from atomicguard.infrastructure.llm import OllamaGenerator

# Local instance (default: http://localhost:11434/v1)
generator = OllamaGenerator(model="qwen2.5-coder:7b")
```

### HuggingFace Inference API

Connects to HuggingFace Inference Providers via `huggingface_hub`. Supports any model available through the HF Inference API, including third-party providers like Together AI.

```bash
# Install the optional dependency
pip install huggingface_hub

# Set your API token
export HF_TOKEN="hf_your_token_here"
```

```python
from atomicguard.infrastructure.llm import HuggingFaceGenerator
from atomicguard.infrastructure.llm.huggingface import HuggingFaceGeneratorConfig

# Default: Qwen/Qwen2.5-Coder-32B-Instruct
generator = HuggingFaceGenerator()

# Custom model and provider
generator = HuggingFaceGenerator(HuggingFaceGeneratorConfig(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    provider="together",       # or "auto", "hf-inference"
    temperature=0.7,
    max_tokens=4096,
))
```

Drop-in replacement in any workflow:

```python
from atomicguard import (
    SyntaxGuard, TestGuard, CompositeGuard,
    ActionPair, DualStateAgent, InMemoryArtifactDAG
)
from atomicguard.infrastructure.llm import HuggingFaceGenerator

generator = HuggingFaceGenerator()
guard = CompositeGuard([SyntaxGuard(), TestGuard("assert add(2, 3) == 5")])
action_pair = ActionPair(generator=generator, guard=guard)
agent = DualStateAgent(action_pair, InMemoryArtifactDAG(), rmax=3)

artifact = agent.execute("Write a function that adds two numbers")
print(artifact.content)
```

## Benchmarks

Run the simulation from the paper:

```bash
python -m benchmarks.simulation --model yi-coder:9b --trials 50 --task all --output results/results.db --format sqlite

# Generate report
python -m benchmarks.simulation --visualize --output results/results.db --format sqlite
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
