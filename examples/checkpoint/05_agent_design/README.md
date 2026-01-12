# Agent Design Process Checkpoint Example

A 7-step workflow that implements the **Agent Design Process** from the Intelligent Agents series, producing both design documentation and runnable Dual-State Action Pair agent skeleton code.

## Overview

This example demonstrates:

- **PEAS Analysis** (Russell & Norvig framework)
- **Environment Classification** (6 dimensions)
- **Agent Function Specification** (percepts, actions, sequences)
- **Agent Type Selection** (with justification)
- **ATDD Acceptance Criteria** (Given-When-Then with 10 Principles)
- **Action Pair Design** (ρ, a_gen, G)
- **Implementation Generation** (workflow.json + Python skeletons)

## Workflow DAG

```
g_peas ─────┬──────────────────────────────────────────┐
            │                                          │
            ▼                                          │
      g_environment ───┬───────────────────────────────┤
            │          │                               │
            ▼          ▼                               │
      g_agent_function ────► g_agent_type              │
            │                    │                     │
            ▼                    │                     │
      g_atdd ◄───────────────────┘                     │
            │                                          │
            ▼                                          ▼
      g_action_pairs ──────────────────────► g_implementation
```

## Prerequisites

1. **Ollama** running locally:

   ```bash
   ollama serve
   ollama pull qwen2.5-coder:14b
   ```

2. **Dependencies** installed:

   ```bash
   uv sync
   ```

## Usage

### Run the Workflow

```bash
# Execute from problem statement
uv run python -m examples.checkpoint.05_agent_design.demo run

# With custom model
uv run python -m examples.checkpoint.05_agent_design.demo run --model qwen2.5-coder:7b
```

### Interactive Wizard

Create a custom problem statement interactively:

```bash
uv run python -m examples.checkpoint.05_agent_design.demo wizard
```

### Resume from Checkpoint

If a step fails after max retries, a checkpoint is created:

```bash
# List checkpoints
uv run python -m examples.checkpoint.05_agent_design.demo list

# Edit the artifact file (shown in output)
# Then resume
uv run python -m examples.checkpoint.05_agent_design.demo resume <checkpoint_id>
```

### Other Commands

```bash
# Clean output directory
uv run python -m examples.checkpoint.05_agent_design.demo clean

# Show workflow configuration
uv run python -m examples.checkpoint.05_agent_design.demo show-config
```

## Output

On success, the workflow produces:

### Design Documentation (JSON artifacts)

- `g_peas.json` - PEAS analysis
- `g_environment.json` - Environment properties
- `g_agent_function.json` - Agent function spec
- `g_agent_type.json` - Agent type selection
- `g_atdd.json` - Acceptance criteria
- `g_action_pairs.json` - Action pair design

### Generated Code (`output/generated/`)

- `workflow.json` - Workflow DAG configuration
- `models.py` - Pydantic schemas
- `generators/*.py` - Generator skeletons
- `guards/*.py` - Guard skeletons

## Key Concepts

### PEAS Framework

- **P**erformance: Metrics that define success
- **E**nvironment: Components of the task environment
- **A**ctuators: Actions the agent can perform
- **S**ensors: Percepts the agent can receive

### Environment Properties (6 Dimensions)

1. Observable: Fully vs Partially observable
2. Deterministic: Deterministic vs Stochastic
3. Static: Static vs Dynamic
4. Discrete: Discrete vs Continuous
5. Agents: Single vs Multi-agent
6. Known: Known vs Unknown

### Agent Types

- Simple Reflex
- Model-Based Reflex
- Goal-Based
- Utility-Based
- Learning

### Dual-State Action Pair: A = ⟨ρ, a_gen, G⟩

- **ρ (rho)**: Precondition - when can this action be taken?
- **a_gen**: Generator - what produces the output?
- **G**: Guard - how do we verify success?

### 10 Principles for Acceptance Criteria

1. Start with clear agent function specification
2. Test observable behaviors, not implementation
3. Avoid predetermined outcomes
4. Match criteria to agent type
5. Handle probabilistic outputs
6. Translate theory to testable behavior
7. Ignore implementation accidents
8. Cover full percept-action cycles
9. Define clear scope boundaries
10. Make tests stakeholder-meaningful

## References

- [IA Series: Intelligent Agents](https://matt.thompson.gr/2025/05/16/ia-series-n-intelligent-agents.html)
- [IA: Agent Design Process v2](https://matt.thompson.gr/2025/07/24/ia-agent-design-process-v.html)
- [IA: Building a Self-Consistency Agent](https://matt.thompson.gr/2025/06/26/ia-series-n-building-a.html)
- [IA: The Case of Claude](https://matt.thompson.gr/2025/12/07/ia-the-case-of-claude.html)
- Russell & Norvig, "Artificial Intelligence: A Modern Approach"
