# Design: Persistent Feedback Across Escalation Cycles

## Problem Statement

When escalation triggers in a workflow, agents lose their accumulated feedback history:

1. `ap_gen_patch` attempts 1-5, accumulating `feedback_history` with each rejection
2. Stagnation detected → escalation triggers
3. Upstream steps (`ap_localise_issue`, `ap_context_read`) re-run with downstream failure context
4. `ap_gen_patch` re-runs with **NEW agent** → `feedback_history = []`
5. Learning from attempts 1-5 is lost; model repeats same mistakes

**Evidence from experiment:** DAG shows `feedback_history` length reset from 4 to 0 after escalation.

---

## Design Principles

1. **DAG is source of truth** - All feedback is stored in artifacts, not agent memory
2. **Agents remain stateless** - No internal state between `execute()` calls
3. **Feedback is reconstructible** - Query DAG at execution start
4. **Two feedback types**:
   - **Retry feedback**: This step's own prior rejections
   - **Escalation feedback**: Downstream failure context sent to upstream steps

---

## Solution Overview

### 1. Add DAG Query Method

Add `get_all_for_action_pair()` to interface for querying prior attempts.

### 2. Agent Reconstructs Feedback from DAG

At start of `execute()`, agent queries DAG for its prior artifacts in this workflow.

### 3. Store Escalation Feedback in Context

Add `escalation_feedback` field to `Context`/`ContextSnapshot` for downstream failure context.

### 4. Add Prompt Template Wrapper

Add `escalation_feedback_wrapper` to `PromptTemplate` for formatting escalation context.

---

## File Changes

### 1. `src/atomicguard/domain/interfaces.py`

Add method to `ArtifactDAGInterface`:

```python
@abstractmethod
def get_all_for_action_pair(
    self, action_pair_id: str, workflow_id: str
) -> list["Artifact"]:
    """
    Get all artifacts for an action pair in a specific workflow.

    Args:
        action_pair_id: The action pair identifier (e.g., 'ap_gen_patch')
        workflow_id: UUID of the workflow execution instance

    Returns:
        List of artifacts sorted by created_at ascending
    """
    pass
```

### 2. `src/atomicguard/infrastructure/persistence/filesystem.py`

Implement the new method:

```python
def get_all_for_action_pair(
    self, action_pair_id: str, workflow_id: str
) -> list[Artifact]:
    """Get all artifacts for an action pair in a specific workflow."""
    if action_pair_id not in self._index.get("action_pairs", {}):
        return []

    artifacts = []
    for artifact_id in self._index["action_pairs"][action_pair_id]:
        artifact_info = self._index["artifacts"].get(artifact_id, {})
        if artifact_info.get("workflow_id") == workflow_id:
            artifacts.append(self.get_artifact(artifact_id))

    # Sort by created_at ascending
    artifacts.sort(key=lambda a: a.created_at)
    return artifacts
```

### 3. `src/atomicguard/infrastructure/persistence/memory.py`

Implement same method for in-memory DAG (used in tests).

### 4. `src/atomicguard/domain/models.py`

Add `escalation_feedback` to `Context` and `ContextSnapshot`:

```python
@dataclass(frozen=True)
class ContextSnapshot:
    """Immutable context C that conditioned generation (Definition 5)."""
    workflow_id: str
    specification: str
    constraints: str
    feedback_history: tuple[FeedbackEntry, ...]
    dependency_artifacts: tuple[tuple[str, str], ...] = ()
    escalation_feedback: tuple[str, ...] = ()  # NEW: One entry per escalation cycle

@dataclass(frozen=True)
class Context:
    """Immutable hierarchical context composition (Definition 5)."""
    ambient: AmbientEnvironment
    specification: str
    current_artifact: str | None = None
    feedback_history: tuple[tuple[str, str], ...] = ()
    dependency_artifacts: tuple[tuple[str, str], ...] = ()
    workflow_id: str | None = None
    escalation_feedback: tuple[str, ...] = ()  # NEW: Downstream failure summaries
```

### 5. `src/atomicguard/domain/prompts.py`

Add `escalation_feedback_wrapper` to `PromptTemplate`:

```python
@dataclass(frozen=True)
class PromptTemplate:
    role: str
    constraints: str
    task: str
    feedback_wrapper: str  # REQUIRED - from prompts.json (no default)
    escalation_feedback_wrapper: str  # REQUIRED - from prompts.json (no default)
```

Update `render()` to include escalation feedback:

```python
def render(self, context: "Context") -> str:
    # ... existing code ...

    # Add escalation feedback if present (NEW)
    if context.escalation_feedback:
        parts.append("# ESCALATION HISTORY")
        for i, feedback in enumerate(context.escalation_feedback):
            wrapped = self.escalation_feedback_wrapper.format(feedback=feedback)
            parts.append(f"--- Escalation Cycle {i + 1} ---\n{wrapped}")

    # Add feedback history (existing)
    if context.feedback_history:
        parts.append("# RETRY HISTORY (This Cycle)")
        for i, (_artifact_content, feedback) in enumerate(context.feedback_history):
            wrapped = self.feedback_wrapper.format(feedback=feedback)
            parts.append(f"--- Attempt {i + 1} ---\n{wrapped}")

    parts.append(f"# TASK\n{task}")
    return "\n\n".join(parts)
```

### 6. `src/atomicguard/application/agent.py`

Reconstruct prior feedback at start of `execute()`:

```python
def execute(
    self,
    specification: str,
    dependencies: dict[str, Artifact] | None = None,
    escalation_feedback: tuple[str, ...] = (),  # NEW: From workflow
) -> Artifact:
    dependencies = dependencies or {}

    # NEW: Reconstruct prior feedback from DAG
    prior_artifacts = self._artifact_dag.get_all_for_action_pair(
        action_pair_id=self._action_pair_id,
        workflow_id=self._workflow_id
    )

    # Initialize feedback_history from prior rejected artifacts
    feedback_history: list[tuple[Artifact, str]] = [
        (a, a.guard_result.feedback)
        for a in prior_artifacts
        if a.status == ArtifactStatus.REJECTED and a.guard_result
    ]

    # Compose context with escalation feedback
    context = self._compose_context(
        specification,
        dependencies,
        escalation_feedback=escalation_feedback  # Pass through
    )

    # ... rest of retry loop unchanged ...
```

Update `_compose_context()`:

```python
def _compose_context(
    self,
    specification: str,
    dependencies: dict[str, Artifact],
    escalation_feedback: tuple[str, ...] = (),  # NEW
) -> Context:
    ambient = AmbientEnvironment(
        repository=self._artifact_dag,
        constraints=self._constraints,
    )
    return Context(
        ambient=ambient,
        specification=specification,
        current_artifact=None,
        feedback_history=(),
        dependency_artifacts=tuple(
            (k, v.artifact_id) for k, v in dependencies.items()
        ),
        workflow_id=self._workflow_id,
        escalation_feedback=escalation_feedback,  # NEW
    )
```

### 7. `src/atomicguard/application/workflow.py`

Track escalation feedback per step and pass to agent:

```python
class Workflow:
    def __init__(self, ...):
        # ... existing ...
        self._escalation_feedback: dict[str, list[str]] = defaultdict(list)  # NEW

    def execute(self, specification: str) -> WorkflowResult:
        # ... in the main loop ...

        # Get escalation feedback for this step (NEW)
        step_escalation_feedback = tuple(self._escalation_feedback.get(step.guard_id, []))

        agent = DualStateAgent(...)

        try:
            artifact = agent.execute(
                specification,
                dependencies,
                escalation_feedback=step_escalation_feedback  # NEW
            )
            # ...

        except StagnationDetected as e:
            if self._escalation_count[step.guard_id] < step.e_max:
                # Store escalation feedback for the FAILED step itself (NEW)
                if e.failure_summary:
                    self._escalation_feedback[step.guard_id].append(e.failure_summary)

                # Existing: inject context to upstream targets
                for target_id in e.escalate_to:
                    self._invalidate_dependents(target_id)
                    if e.failure_summary:
                        self._inject_failure_context(target_id, e.failure_summary)
                        # Also track as escalation feedback (NEW)
                        self._escalation_feedback[target_id].append(e.failure_summary)

                # ... rest unchanged ...
```

### 8. `src/atomicguard/schemas/prompts.schema.json`

Add `escalation_feedback_wrapper` to schema:

```json
{
  "$defs": {
    "PromptTemplate": {
      "properties": {
        "role": { "type": "string" },
        "constraints": { "type": "string" },
        "task": { "type": "string" },
        "feedback_wrapper": { "type": "string" },
        "escalation_feedback_wrapper": {
          "type": "string",
          "description": "Template for wrapping escalation cycle feedback. Use {feedback} placeholder."
        }
      },
      "required": ["role", "constraints", "task", "feedback_wrapper", "escalation_feedback_wrapper"]
    }
  }
}
```

### 9. Update serialization in `filesystem.py`

Add `escalation_feedback` to `ContextSnapshot` serialization:

```python
def _artifact_to_dict(self, artifact: Artifact) -> dict:
    # ...
    "context": {
        # ... existing fields ...
        "escalation_feedback": list(artifact.context.escalation_feedback),  # NEW
    }

def _dict_to_artifact(self, data: dict) -> Artifact:
    context = ContextSnapshot(
        # ... existing fields ...
        escalation_feedback=tuple(data["context"].get("escalation_feedback", [])),  # NEW
    )
```

---

## Data Flow

```
Escalation Cycle 1:
  ap_gen_patch creates agent
  Agent queries DAG → [] (no prior artifacts)
  Attempts 1-5 fail, stored to DAG
  StagnationDetected with failure_summary

  Workflow stores: _escalation_feedback["ap_gen_patch"] = [summary1]
  Workflow stores: _escalation_feedback["ap_localise_issue"] = [summary1]

  Upstream steps re-run with escalation_feedback

Escalation Cycle 2:
  ap_gen_patch creates NEW agent
  Agent queries DAG → [artifact_1..5] (prior attempts!)
  Agent initializes feedback_history from DAG
  Agent receives escalation_feedback=(summary1,) from workflow

  Attempts 6+ have full context of prior failures
```

---

## Verification

### Unit Tests

```python
def test_agent_reconstructs_feedback_from_dag():
    """Agent should query DAG for prior attempts at execute() start."""
    # Setup: DAG with 3 rejected artifacts for ap_gen_patch
    # Execute: Create new agent, call execute()
    # Assert: feedback_history initialized from DAG artifacts

def test_escalation_feedback_persisted():
    """Escalation feedback should be stored in artifact context."""
    # Setup: Workflow with escalation
    # Execute: Run until escalation triggers
    # Assert: Artifacts after escalation have escalation_feedback in context

def test_prompt_renders_escalation_feedback():
    """PromptTemplate should include escalation_feedback section."""
    # Setup: Context with escalation_feedback=("summary1", "summary2")
    # Execute: template.render(context)
    # Assert: Output contains "Escalation Cycle 1" and "Escalation Cycle 2"
```

### Integration Test

Re-run the s1_decomposed experiment and verify:

```bash
# After escalation, artifact should have prior feedback in context
jq '.context.feedback_history | length' \
  output/*/artifact_dags/*/07_s1_decomposed/objects/*/21*.json

# Should be > 0 (not reset to empty)
```

---

## Architecture Constraints

### No Hardcoded Prompts

**All prompt templates must come from `prompts.json`** - no defaults in code.

**Current issue**: `feedback_wrapper` has a default in code:
```python
feedback_wrapper: str = (
    "GUARD REJECTION:\n{feedback}\nInstruction: Address the rejection above."
)
```

Update `PromptTemplate` to have NO defaults for wrapper fields:

```python
@dataclass(frozen=True)
class PromptTemplate:
    role: str
    constraints: str
    task: str
    feedback_wrapper: str  # REQUIRED - from prompts.json
    escalation_feedback_wrapper: str  # REQUIRED - from prompts.json
```

Each prompt in `prompts.json` must include both wrappers:

```json
{
  "ap_gen_patch": {
    "role": "You are a code patch generator...",
    "constraints": "...",
    "task": "...",
    "feedback_wrapper": "GUARD REJECTION:\n{feedback}\nInstruction: Address the rejection above.",
    "escalation_feedback_wrapper": "## Previous Escalation Cycle\nThe following summarizes what went wrong:\n{feedback}\nAvoid these mistakes."
  }
}
```

### DDD/Hexagonal Architecture Tests

Add architecture tests to ensure boundaries are maintained:

```python
# tests/architecture/test_hexagonal_boundaries.py

import ast
import importlib
from pathlib import Path

def test_domain_has_no_infrastructure_imports():
    """Domain layer must not import from infrastructure."""
    domain_files = Path("src/atomicguard/domain").glob("*.py")

    forbidden = ["atomicguard.infrastructure", "atomicguard.application"]

    for file in domain_files:
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, 'module', '') or ''
                for name in getattr(node, 'names', []):
                    full_name = f"{module}.{name.name}" if module else name.name
                    for f in forbidden:
                        assert f not in full_name, \
                            f"{file.name} imports {full_name} - domain must not depend on {f}"

def test_no_hardcoded_prompts_in_code():
    """Prompts must come from prompts.json, not be hardcoded."""
    code_files = list(Path("src/atomicguard").rglob("*.py"))

    # Patterns that suggest hardcoded prompts
    forbidden_patterns = [
        "You are a",  # Role prompts
        "GUARD REJECTION",  # Feedback wrapper
        "Previous Escalation",  # Escalation wrapper
        "# ROLE\n",  # Prompt structure
    ]

    for file in code_files:
        content = file.read_text()
        for pattern in forbidden_patterns:
            # Allow in test files and prompts.py (which loads from JSON)
            if "test" not in str(file) and pattern in content:
                # Check if it's in a string literal (crude but effective)
                if f'"{pattern}' in content or f"'{pattern}" in content:
                    raise AssertionError(
                        f"{file}: Found hardcoded prompt pattern '{pattern}'. "
                        "Prompts must come from prompts.json"
                    )

def test_prompt_template_requires_all_wrappers():
    """PromptTemplate must not have default values for wrappers."""
    from atomicguard.domain.prompts import PromptTemplate
    import inspect

    sig = inspect.signature(PromptTemplate)

    # These fields should NOT have defaults
    required_fields = ["feedback_wrapper", "escalation_feedback_wrapper"]

    for field in required_fields:
        param = sig.parameters.get(field)
        assert param is not None, f"PromptTemplate missing field: {field}"
        assert param.default is inspect.Parameter.empty, \
            f"PromptTemplate.{field} has a default value - it should be required from prompts.json"
```

---

## Files to Modify

| File | Change |
|------|--------|
| `src/atomicguard/domain/interfaces.py` | Add `get_all_for_action_pair()` |
| `src/atomicguard/infrastructure/persistence/filesystem.py` | Implement method + serialization |
| `src/atomicguard/infrastructure/persistence/memory.py` | Implement method |
| `src/atomicguard/domain/models.py` | Add `escalation_feedback` to Context/ContextSnapshot |
| `src/atomicguard/domain/prompts.py` | Add `escalation_feedback_wrapper` (no default), update `render()` |
| `src/atomicguard/application/agent.py` | Query DAG, accept escalation_feedback param |
| `src/atomicguard/application/workflow.py` | Track and pass escalation_feedback |
| `src/atomicguard/schemas/prompts.schema.json` | Add `escalation_feedback_wrapper` (required) |
| `examples/*/prompts.json` (6 files) | Add `escalation_feedback_wrapper` to all prompts |
| `tests/architecture/test_hexagonal_boundaries.py` | **NEW**: Architecture constraint tests |
| `tests/domain/test_prompts.py` | Add tests for escalation feedback rendering |

---

## Backwards Compatibility

- New fields have defaults (`escalation_feedback=()`)
- Existing DAG artifacts without `escalation_feedback` will deserialize with empty tuple
- No migration needed
