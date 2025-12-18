# TDD Workflow Benchmark Design

This document describes the design of the TDD workflow benchmark system used to evaluate the Dual-State Framework's ability to generate test-driven code.

## Overview

The benchmark evaluates LLM code generation through a two-step TDD workflow:

1. **G_test**: Generate pytest tests from a specification
2. **G_impl**: Generate implementation that passes those tests

This demonstrates the core AtomicGuard pattern: generators produce artifacts, guards validate them, and the cycle repeats until acceptance or r_max exhaustion.

## Architecture

```
benchmarks/
├── workflow_benchmark.py   # CLI benchmark runner
└── workflows.json          # Task definitions (action_pairs)
```

### Workflow JSON Schema

```json
{
  "task_id": {
    "name": "Human-readable name",
    "specification": "Task description",
    "action_pairs": {
      "g_test": {
        "prompt": "Test generation prompt...",
        "guard": "syntax"
      },
      "g_impl": {
        "prompt": "Implementation prompt with {test_code}...",
        "guard": "dynamic_test",
        "requires": ["g_test"]
      }
    }
  }
}
```

### Action Pair Structure

Each action pair couples:

- **Generator**: LLM that produces code artifacts
- **Guard**: Validator that accepts or rejects artifacts

| Field | Description |
|-------|-------------|
| `prompt` | Template for LLM generation |
| `guard` | Guard type: `syntax`, `dynamic_test`, `human` |
| `requires` | Dependencies (artifacts from prior steps) |

## Guard Types

### G8: SyntaxGuard

Validates Python syntax via `ast.parse()`. Used for test generation step.

```
ast_parse_succeeds ∧ no_syntax_errors
```

### G10: DynamicTestGuard

Executes generated tests against generated implementation in sandboxed subprocess.

```
unit_tests_pass ∧ no_runtime_errors
```

## Benchmark Tasks

### Difficulty Levels

| Task | Difficulty | Key Challenge |
|------|------------|---------------|
| `tdd_stack` | Easy | Basic data structure |
| `tdd_calculator` | Easy | Arithmetic + error handling |
| `tdd_queue` | Easy | FIFO ordering |
| `tdd_lrucache` | Medium | State management, eviction order |
| `tdd_template` | Medium-Hard | String parsing, error semantics |

### Task Selection Criteria

Good TDD benchmark tasks should have:

1. **Unambiguous specification** - Single correct interpretation
2. **Testable behavior** - Clear pass/fail criteria
3. **Enumerable edge cases** - Finite test coverage
4. **No external dependencies** - Self-contained

### LRUCache (Medium)

Tests state management with access-order tracking:

- Capacity enforcement
- LRU eviction semantics
- Access order updates on get/put

### SimpleTemplate (Medium-Hard)

Tests string manipulation with strict error semantics:

- Placeholder substitution `{varname}`
- Escaped braces `{{` → `{`
- `KeyError` on missing variables

The SimpleTemplate task demonstrates specification precision requirements. Ambiguous specs (e.g., "what happens on missing variable?") lead to test/implementation mismatches.

## Execution Model

```
┌─────────────────────────────────────────────────────┐
│                  run_tdd_workflow()                  │
├─────────────────────────────────────────────────────┤
│  1. Load task from workflows.json                   │
│  2. Create ActionPair(generator, guard) per step    │
│  3. Execute g_test with SyntaxGuard                 │
│  4. Execute g_impl with DynamicTestGuard            │
│     - Inject test_artifact as dependency            │
│  5. Return TDDWorkflowResult                        │
└─────────────────────────────────────────────────────┘
```

### Retry Semantics

Each action pair has r_max retries. On guard failure:

1. Guard returns `(passed=False, feedback=error_message)`
2. Feedback appended to next generation prompt
3. DualStateAgent retries until success or r_max exhaustion
4. `RmaxExhausted` raised on failure

## Artifact Provenance

Each artifact tracks:

- `artifact_id`: UUID
- `action_pair_id`: Which step produced it (`g_test`, `g_impl`)
- `attempt_number`: Which retry attempt
- `dependency_ids`: Artifacts this depends on
- `status`: `pending`, `accepted`, `rejected`
- `feedback`: Guard rejection message

Artifacts are persisted to `FilesystemArtifactDAG` for post-hoc analysis.

## CLI Usage

```bash
# Run single task
uv run python benchmarks/workflow_benchmark.py \
  --task tdd_stack \
  --trials 10 \
  --model qwen2.5-coder:7b

# Run all tasks
uv run python benchmarks/workflow_benchmark.py \
  --task all \
  --trials 50 \
  --output results.csv \
  --artifact-dir ./artifacts

# Custom workflow file
uv run python benchmarks/workflow_benchmark.py \
  --workflow custom_tasks.json \
  --task my_task
```

## Output

### CSV Results

```csv
model_name,task,trial_num,success,total_attempts,duration_seconds,failed_step,error_message,timestamp
qwen2.5-coder:7b,tdd_stack,1,True,2,6.5,,,2025-12-17 19:30:16
qwen2.5-coder:7b,tdd_template,1,False,5,22.3,g_impl,Failed after 3 retries,2025-12-17 19:52:42
```

### Artifact Storage

```
artifacts/
└── tdd_stack/
    └── trial_0/
        ├── index.json          # DAG index
        └── objects/
            ├── ab/ab12...json  # Test artifact
            └── cd/cd34...json  # Impl artifact
```

## Design Decisions

### Why JSON for Workflows?

1. **Separation of concerns**: Task content separate from benchmark logic
2. **Schema validation**: Catch errors before execution
3. **Extensibility**: Add tasks without code changes
4. **Reproducibility**: Version control task definitions

### Why action_pairs not steps?

Aligns with `ActionPair` class terminology in the codebase. Each entry represents a generator+guard coupling.

### Why r_max per step?

Allows step-specific retry budgets. Test generation (syntax-only) typically needs fewer retries than implementation (dynamic testing).

## Future Extensions

1. **Multi-model comparison**: Compare models on same tasks
2. **Guard chain analysis**: Track which guards fail most often
3. **Prompt optimization**: A/B test prompt variations
4. **Complexity metrics**: Correlate task complexity with success rate
