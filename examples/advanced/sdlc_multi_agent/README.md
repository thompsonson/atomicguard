# Multi-Agent SDLC Workflow - Proof of Concept

**Version:** 1.0.0
**Status:** Proof of Concept
**Date:** January 2026

## Overview

This is a **Proof-of-Concept** implementation demonstrating AtomicGuard's multi-agent coordination capabilities. Three specialized agents collaborate through a shared DAG to transform user requirements into validated Python implementations.

### Three-Agent Workflow

```
┌────────────────────────────────────────────────────────────────┐
│                    Multi-Agent SDLC Orchestrator                │
│                     Total Retry Budget: 23                      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Phase 1: DDD Agent                                      │  │
│  │  Generator: DDDGenerator (LLM)                          │  │
│  │  Guard: DocumentationGuard (deterministic)              │  │
│  │  Budget: 5 attempts                                     │  │
│  │  Output: domain_model.md, infrastructure_requirements.md│  │
│  │          project_structure.md, ubiquitous_language.md   │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Phase 2: Coder Agent                                    │  │
│  │  Generator: CoderGenerator (LLM)                        │  │
│  │  Guard: AllTestsPassGuard (pytest)                      │  │
│  │  Budget: 11 attempts                                    │  │
│  │  Input: DDD docs (materialized from DAG)                │  │
│  │  Output: src/domain/*.py, tests/*.py                    │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Phase 3: Tester Agent                                   │  │
│  │  Generator: IdentityGenerator (pass-through)            │  │
│  │  Guard: AllTestsPassGuard (pytest)                      │  │
│  │  Budget: 7 attempts                                     │  │
│  │  Input: Implementation (materialized from DAG)          │  │
│  │  Output: Test results (validation only)                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Architectural Principle

**The DAG is the Shared Source of Truth**

- **Artifacts** are stored in the DAG with full provenance
- **Workspaces** are ephemeral filesystem views for agent execution
- **WorkspaceService** bridges the two representations bidirectionally

```
┌────────────────────────────────────────────────────────────────┐
│                      ArtifactDAG (Source of Truth)             │
│  • Stores structured artifacts as JSON                         │
│  • Tracks provenance (dependencies, versions)                  │
│  • Enables queries, rollback, incremental execution            │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          │ WorkspaceService
                          │ (Bidirectional Sync)
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│              Workspaces (Ephemeral Filesystem)                 │
│  • Created per phase per attempt                               │
│  • Claude SDK operates on real files                           │
│  • Persisted for debugging                                     │
│  • Cleaned up after validation                                 │
└────────────────────────────────────────────────────────────────┘
```

## Separation of Concerns

Each component has clearly defined responsibilities:

### Layer 1: CLI (demo.py)
- ✅ Parse commands
- ✅ Initialize services
- ✅ Display results
- ❌ NO business logic

### Layer 2: Orchestrator
- ✅ Phase sequencing (DDD → Coder → Tester)
- ✅ Retry budget management
- ✅ Coordinate WorkspaceService, Generators, Guards
- ✅ Store artifacts in DAG
- ❌ NO knowledge of LLM internals
- ❌ NO knowledge of validation logic

### Layer 3a: WorkspaceService
- ✅ Materialize artifacts to filesystem
- ✅ Capture filesystem changes to artifacts
- ✅ Manage workspace lifecycle
- ❌ NO validation logic
- ❌ NO LLM calls

### Layer 3b: Generators
- ✅ Call LLM with prompts
- ✅ Format input/output
- ❌ NO validation (Guard's job)
- ❌ NO retry logic (Orchestrator's job)

### Layer 3c: Guards
- ✅ Validate artifact content OR filesystem state
- ✅ Return GuardResult (passed/failed + feedback)
- ❌ NO LLM calls (deterministic only)
- ❌ NO retry logic (Orchestrator's job)

### Layer 4: ArtifactDAG
- ✅ Store artifacts with provenance
- ✅ Query artifacts
- ❌ NO workflow sequencing
- ❌ NO business logic

## Key Design Decisions

### Decision 1: Option C - Eager Materialization

**Chosen:** Pre-materialize ALL upstream artifacts before each phase starts

**Rationale:**
- Provides full context to agents (DDD docs available for Coder)
- Simplifies agent implementation (just read filesystem)
- Supports Claude SDK's filesystem-based tools
- Enables debugging (inspect workspace before/after)

### Decision 2: Single Coder Agent

**Chosen:** One Coder agent for the PoC

**Rationale:**
- Simpler to implement and test
- Validates orchestration model
- Can expand to multiple specialized coders later (DomainCoder, InfrastructureCoder)

### Decision 3: Persistent Workspaces

**Chosen:** Keep workspaces after execution with CLI commands to manage them

**Rationale:**
- Essential for debugging during PoC development
- Allows inspection of agent outputs
- CLI commands (`list-workspaces`, `clean-workspaces`) provide control

## Installation & Setup

### Prerequisites

```bash
# 1. Install dependencies
uv sync

# 2. Install openai package (required for LLM generators)
pip install openai

# 3. Start Ollama
ollama serve

# 4. Pull model
ollama pull qwen2.5-coder:14b
```

### Verify Installation

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check model is available
ollama list | grep qwen2.5-coder
```

## Usage

### Run Workflow

```bash
# Run with default sample input
uv run python -m examples.advanced.sdlc_multi_agent.demo run

# Run with custom input
uv run python -m examples.advanced.sdlc_multi_agent.demo run --input path/to/requirements.md

# Run with different model
uv run python -m examples.advanced.sdlc_multi_agent.demo run --model llama2:13b
```

### List Workspaces

```bash
# List all persisted workspaces
uv run python -m examples.advanced.sdlc_multi_agent.demo list-workspaces
```

Output:
```
PERSISTED WORKSPACES
============================================================

Found 6 workspace(s):

  g_coder_attempt_1
    Path: output/workspaces/g_coder_attempt_1
    Size: 0.05 MB
    Files: 8 (*.py, *.md)

  g_ddd_attempt_1
    Path: output/workspaces/g_ddd_attempt_1
    Size: 0.02 MB
    Files: 4 (*.py, *.md)

  ...
```

### Clean Workspaces

```bash
# Delete only workspaces (keeps artifacts)
uv run python -m examples.advanced.sdlc_multi_agent.demo clean-workspaces

# Delete everything (workspaces + artifacts)
uv run python -m examples.advanced.sdlc_multi_agent.demo clean
```

## Project Structure

```
examples/advanced/sdlc_multi_agent/
├── README.md                      # This file
├── __init__.py                    # Package initialization
├── demo.py                        # CLI interface
├── workflow.json                  # Workflow configuration
├── prompts.json                   # LLM prompts for each phase
├── interfaces.py                  # Interface definitions (SoC)
├── workspace_service.py           # Filesystem ↔ DAG synchronization
├── orchestrator.py                # Phase sequencing & retry management
├── generators/
│   ├── __init__.py
│   ├── base.py                    # BaseGenerator (LLM client)
│   ├── ddd_generator.py           # DDD documentation generation
│   ├── coder_generator.py         # Implementation generation
│   └── identity_generator.py      # Pass-through (for Tester)
├── guards/
│   ├── __init__.py
│   ├── documentation_guard.py     # Validate DDD docs
│   └── all_tests_pass_guard.py    # Run pytest
├── sample_input/
│   └── requirements.md            # Example: Library management system
└── output/                        # Generated during execution
    ├── artifact_dag/              # DAG storage (source of truth)
    └── workspaces/                # Ephemeral workspaces (persisted for debugging)
        ├── g_ddd_attempt_1/
        ├── g_coder_attempt_1/
        └── g_tester_attempt_1/
```

## Workflow Execution Flow

### Phase 1: DDD Agent

1. Orchestrator creates workspace: `output/workspaces/g_ddd_attempt_1/`
2. No dependencies to materialize (first phase)
3. DDDGenerator calls LLM with user intent
4. LLM generates 4 markdown files
5. Generator writes files to workspace
6. DocumentationGuard validates:
   - All required files exist
   - Each file has required sections
   - Extracts metadata (entities, gates)
7. If passed:
   - WorkspaceService captures files to JSON
   - Artifact stored in DAG
   - Move to Phase 2
8. If failed:
   - Feedback added to prompt
   - Retry (up to 5 attempts)

### Phase 2: Coder Agent

1. Orchestrator creates workspace: `output/workspaces/g_coder_attempt_1/`
2. **WorkspaceService materializes DDD artifact** (Option C: eager)
   - Workspace now contains: `docs/domain_model.md`, etc.
3. CoderGenerator calls LLM with instruction to read docs
4. LLM generates implementation files
5. Generator writes files to workspace
6. AllTestsPassGuard validates:
   - Tests directory exists
   - Test files exist
   - Runs pytest
7. If passed:
   - WorkspaceService captures files to JSON
   - Artifact stored in DAG
   - Move to Phase 3
8. If failed:
   - Feedback includes pytest output
   - Retry (up to 11 attempts)

### Phase 3: Tester Agent

1. Orchestrator creates workspace: `output/workspaces/g_tester_attempt_1/`
2. **WorkspaceService materializes Coder artifact**
   - Workspace now contains: `src/`, `tests/`
3. IdentityGenerator returns pass-through (no LLM call)
4. AllTestsPassGuard runs pytest
5. If passed:
   - Workflow SUCCESS
6. If failed:
   - Feedback to user (no retry for Tester in PoC)

## Example Output

### Successful Workflow

```
============================================================
MULTI-AGENT SDLC WORKFLOW - Proof of Concept
============================================================

Three-phase workflow:
  1. DDD Agent: Generate domain documentation
  2. Coder Agent: Generate implementation from docs
  3. Tester Agent: Validate tests pass

LLM Host: http://localhost:11434
Model: qwen2.5-coder:14b
Total Retry Budget: 23

User Intent: 1483 chars

============================================================
EXECUTING WORKFLOW
============================================================

============================================================
[SUCCESS]
============================================================

Completed Phases: ['g_ddd', 'g_coder', 'g_tester']

Attempts Used: 3 / 23
Budget Remaining: 20

------------------------------------------------------------
PHASE DETAILS
------------------------------------------------------------

✓ g_ddd: SUCCESS
  Attempts: 1
  Artifact ID: a1b2c3d4e5f6...

✓ g_coder: SUCCESS
  Attempts: 1
  Artifact ID: f6e5d4c3b2a1...

✓ g_tester: SUCCESS
  Attempts: 1
  Artifact ID: 123456789abc...

------------------------------------------------------------
OUTPUT LOCATIONS
------------------------------------------------------------
Artifacts: output/artifact_dag
Workspaces: output/workspaces

------------------------------------------------------------
NEXT STEPS
------------------------------------------------------------
1. Explore workspaces:
   ls output/workspaces
2. View generated code:
   ls output/workspaces/g_coder_attempt_1/src/
3. List all workspaces:
   uv run python -m examples.advanced.sdlc_multi_agent.demo list-workspaces
```

## Configuration

### workflow.json

```json
{
  "name": "multi_agent_sdlc_poc",
  "model": "qwen2.5-coder:14b",
  "phases": {
    "g_ddd": {
      "generator": "DDDGenerator",
      "guard": "DocumentationGuard",
      "retry_budget": 5
    },
    "g_coder": {
      "generator": "CoderGenerator",
      "guard": "AllTestsPassGuard",
      "dependencies": ["g_ddd"],
      "retry_budget": 11
    },
    "g_tester": {
      "generator": "IdentityGenerator",
      "guard": "AllTestsPassGuard",
      "dependencies": ["g_coder"],
      "retry_budget": 7
    }
  },
  "total_retry_budget": 23
}
```

### prompts.json

Defines the prompt template for each phase. Prompts can reference:
- `{user_intent}` - User requirements
- Workspace files (accessed by agents)

## Limitations & Future Work

### Current Limitations

1. **No TDD/BDD Agents**: Coder generates both implementation and tests
2. **No Architecture Tests**: No pytest-arch validation
3. **No Structure Audit**: No filesystem validation against docs
4. **Single Coder**: One agent for all implementation
5. **No Learning**: Fixed retry budgets (no learning loop)

### Future Enhancements

1. **Full 6-Agent System**:
   - ADD Agent: Generate pytest-arch tests
   - TDD Agent: Generate pytest unit tests
   - BDD Agent: Generate pytest-bdd scenarios
   - Separate test generation from implementation

2. **Project Structure Agent**:
   - Phase 1: Skeleton creation (blocks ADD/TDD)
   - Phase 3: File creation tool for Coder
   - Phase 5: Structure audit

3. **Gap Analysis**:
   - Aggregate violations from 4 streams (Architecture, Unit, BDD, Structure)
   - Remediation phase with targeted fixes

4. **Learning Loop**:
   - Track retry budget consumption per phase
   - Adjust budgets based on historical data
   - Optimize workflow efficiency

5. **Extension Integration**:
   - Extension 01: Versioned Environment (W_ref, config_ref)
   - Extension 02: Artifact Extraction (predicate queries)
   - Extension 07: Incremental Execution (skip unchanged phases)

## Troubleshooting

### Ollama Connection Error

```
[ERROR] LLM call failed: Connection refused
```

**Solution:**
```bash
ollama serve
```

### Model Not Found

```
[ERROR] Model 'qwen2.5-coder:14b' not found
```

**Solution:**
```bash
ollama pull qwen2.5-coder:14b
```

### Tests Fail (Coder Phase)

```
✗ g_coder: FAILED
  Feedback: Tests failed:
  FAILED tests/test_book.py::test_book_creation ...
```

**Solution:**
- Check workspace: `ls output/workspaces/g_coder_attempt_*/tests/`
- Review test output in feedback
- LLM will retry with feedback (up to 11 attempts)
- If exhausted, inspect workspace and debug manually

### Workspace Disk Space

```
[WARNING] Workspaces using 500 MB
```

**Solution:**
```bash
# Clean old workspaces
uv run python -m examples.advanced.sdlc_multi_agent.demo clean-workspaces
```

## Contributing

This is a Proof-of-Concept. For production use, see the full specification:
`docs/design/multi_agent_sdlc_specification.md`

## License

See repository LICENSE file.
