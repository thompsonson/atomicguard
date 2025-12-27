# Multi-Agent SDLC Example

Demonstrates hierarchical composition (Remark 5 from the paper) by orchestrating multiple semantic agents in an end-to-end development pipeline.

## Pipeline Overview

```
                    ┌─────────────┐
                    │  g_config   │  ConfigExtractorGenerator
                    │ (extract)   │  → ProjectConfig
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       ┌─────────────┐           ┌─────────────┐
       │   g_add     │           │   g_bdd     │
       │(architecture)│           │ (scenarios) │
       │ ADDGenerator │           │ BDDGenerator │
       └──────┬──────┘           └──────┬──────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                    ┌─────────────┐
                    │  g_coder    │  CoderGenerator
                    │ (implement) │  → Implementation
                    └─────────────┘
```

## Generators

| Generator | Input | Output | Guard |
|-----------|-------|--------|-------|
| `ConfigExtractorGenerator` | Documentation | `ProjectConfig` | `config_extracted` |
| `ADDGenerator` | Documentation + Config | pytest-arch tests | `architecture_tests_valid` |
| `BDDGenerator` | Requirements + Config | Gherkin scenarios | `scenarios_valid` |
| `CoderGenerator` | Tests + Scenarios | Implementation files | `all_tests_pass` |

## Usage

```bash
# Run with default sample input
python -m examples.sdlc.run

# Run with custom documentation
python -m examples.sdlc.run --docs path/to/requirements.md

# Specify Ollama host and model
python -m examples.sdlc.run --host http://gpu:11434 --model qwen2.5-coder:14b

# Custom output directory
python -m examples.sdlc.run --workdir ./my_output

# Enable verbose logging
python -m examples.sdlc.run -v
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `http://localhost:11434` | Ollama API URL |
| `--model` | from workflow.json | Override model |
| `--docs` | sample_input/requirements.md | Requirements documentation |
| `--workflow` | workflow.json | Workflow configuration |
| `--prompts` | prompts.json | Prompt templates |
| `--output` | output/results.json | Results output path |
| `--artifact-dir` | output/artifacts | Artifact storage |
| `--workdir` | output/ | Working directory |
| `-v/--verbose` | false | Enable debug logging |

## Configuration Files

### workflow.json

Defines the multi-agent pipeline:

```json
{
  "name": "Multi-Agent SDLC",
  "model": "qwen2.5-coder:14b",
  "rmax": 3,
  "action_pairs": {
    "g_config": {
      "generator": "ConfigExtractorGenerator",
      "guard": "config_extracted"
    },
    "g_add": {
      "generator": "ADDGenerator",
      "guard": "architecture_tests_valid",
      "requires": ["g_config"]
    },
    "g_bdd": {
      "generator": "BDDGenerator",
      "guard": "scenarios_valid",
      "requires": ["g_config"]
    },
    "g_coder": {
      "generator": "CoderGenerator",
      "guard": "all_tests_pass",
      "requires": ["g_add", "g_bdd"]
    }
  }
}
```

### prompts.json

Prompt templates for each generator with role, constraints, and feedback wrappers.

## Guards

| Guard | Validation |
|-------|------------|
| `config_extracted` | ProjectConfig schema + non-empty fields |
| `architecture_tests_valid` | pytest-arch test structure + minimum test count |
| `scenarios_valid` | Gherkin syntax + Given/When/Then steps |
| `all_tests_pass` | Writes files and runs pytest |

## Output Structure

```
output/
├── results.json       # Workflow execution summary
├── run.log           # Execution log
└── artifacts/        # Artifact DAG storage
    ├── g_config/     # ProjectConfig artifacts
    ├── g_add/        # Architecture test artifacts
    ├── g_bdd/        # BDD scenario artifacts
    └── g_coder/      # Implementation artifacts
```

## Sample Input

The `sample_input/` directory contains example documentation:

- `requirements.md` - Feature requirements with user stories and acceptance criteria
- `architecture.md` - Architecture documentation with layer definitions and gates

## Key Concepts

### Hierarchical Composition (Remark 5)

Each generator (ADDGenerator, BDDGenerator, CoderGenerator) may be an autonomous Semantic Agent with its own internal retry loops and context management. From the orchestrator's perspective, these are atomic operations.

### Parallel Execution

Steps with the same preconditions can execute in parallel:

- `g_add` and `g_bdd` both depend only on `g_config`
- The orchestrator can run them concurrently

### Dependency Flow

```
g_config (∅) → g_add (g_config) ─┐
                                 ├→ g_coder (g_add, g_bdd)
g_config (∅) → g_bdd (g_config) ─┘
```

## Requirements

- Python 3.11+
- Ollama with qwen2.5-coder model
- pydantic-ai

## Installation

```bash
# From repository root
uv pip install -e ".[dev]"

# Pull required model
ollama pull qwen2.5-coder:14b
```
