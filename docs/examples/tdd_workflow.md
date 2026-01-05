# Dual-State Agent: TDD Multi-Step Workflow Example

> **Runnable Example**: For a complete runnable implementation of this TDD pattern with proper guard composition, see [examples/basics/03_tdd_import_guard/](../../examples/basics/03_tdd_import_guard/) which demonstrates defense-in-depth with SyntaxGuard → ImportGuard → HumanReviewGuard.

## Workflow Overview

Test-Driven Development workflow with two sequential steps:

| Step | Name | Guard | Dependencies |
|------|------|-------|--------------|
| 1 | Generate Tests | `g_test` (syntax only) | None |
| 2 | Generate Implementation | `g_impl` (TDD) | `g_test` artifact |

---

## Specifications

```python
TEST_SPECIFICATION = """
Write pytest-style unit tests for an LRU Cache implementation.

Requirements for tests:
1. Test class named `TestLRUCache`
2. Test basic get/put operations
3. Test capacity eviction (LRU item removed)
4. Test that get() updates recency
5. Test updating existing keys
6. Use assert statements

Return only the Python test code, no implementation.
"""

IMPL_SPECIFICATION = """
Implement an LRU (Least Recently Used) Cache in Python.

Requirements:
1. Class named `LRUCache`
2. Constructor takes `capacity: int`
3. Method `get(key: int) -> int`: Return value if key exists, else -1
4. Method `put(key: int, value: int) -> None`: Insert or update key-value pair
5. When capacity exceeded, evict least recently used item

Return only the Python code, no explanations.
"""
```

---

## Guards

```python
class TestSyntaxGuard(GuardInterface):
    """
    Step 1 Guard: Validates test code is syntactically correct
    and contains expected test structure.
    No dependencies required.
    """

    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        try:
            tree = ast.parse(artifact.content)

            # Check for test class
            has_test_class = any(
                isinstance(node, ast.ClassDef) and node.name.startswith('Test')
                for node in ast.walk(tree)
            )

            if not has_test_class:
                return GuardResult(
                    passed=False,
                    feedback="Missing test class (expected class starting with 'Test')"
                )

            # Check for test methods
            has_test_methods = any(
                isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
                for node in ast.walk(tree)
            )

            if not has_test_methods:
                return GuardResult(
                    passed=False,
                    feedback="Missing test methods (expected methods starting with 'test_')"
                )

            return GuardResult(passed=True)

        except SyntaxError as e:
            return GuardResult(
                passed=False,
                feedback=f"Syntax error at line {e.lineno}: {e.msg}"
            )


class TDDImplementationGuard(GuardInterface):
    """
    Step 2 Guard: Validates implementation against generated tests.
    Requires 'test' dependency from step 1.
    """

    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        test_artifact = dependencies.get('test')

        if not test_artifact:
            return GuardResult(
                passed=False,
                feedback="TDD Guard requires 'test' dependency from prior step"
            )

        namespace = {}
        try:
            # Load implementation
            exec(artifact.content, namespace)

            if 'LRUCache' not in namespace:
                return GuardResult(
                    passed=False,
                    feedback="Missing LRUCache class definition"
                )

            # Load and execute tests
            exec(test_artifact.content, namespace)

            # Find and run test class
            test_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and name.startswith('Test'):
                    test_class = obj
                    break

            if not test_class:
                return GuardResult(
                    passed=False,
                    feedback="No test class found in test artifact"
                )

            # Run test methods
            instance = test_class()
            for method_name in dir(instance):
                if method_name.startswith('test_'):
                    method = getattr(instance, method_name)
                    method()

            return GuardResult(passed=True)

        except AssertionError as e:
            return GuardResult(
                passed=False,
                feedback=f"Test failed: {e}"
            )
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"{type(e).__name__}: {e}"
            )
```

---

## Prompt Templates

```python
TEST_PROMPT_TEMPLATE = PromptTemplate(
    role="""You are a Test Engineer practicing TDD.
Your goal is to write comprehensive unit tests BEFORE implementation.""",

    constraints="""1. Use pytest-style test classes (class TestXxx)
2. Use assert statements for validation
3. Do NOT write implementation code
4. Cover edge cases: empty input, capacity limits, update existing
5. Return only the Python test code""",

    task="""Write unit tests for an LRU Cache implementation.

Required test coverage:
- Basic get/put operations
- Capacity eviction (LRU item removed)
- get() updates recency
- Updating existing keys""",

    feedback_wrapper="""GUARD REJECTION:
{feedback}

Instruction: Your test code has structural issues. Fix the specific problem above."""
)

IMPL_PROMPT_TEMPLATE = PromptTemplate(
    role="""You are a Python Developer implementing code to pass existing tests.
The tests are provided by a prior step and cannot be modified.""",

    constraints="""1. Class must be named exactly as expected by tests
2. All methods must match test expectations
3. Do NOT modify or redefine the tests
4. Return only the implementation code""",

    task="""Implement an LRU (Least Recently Used) Cache in Python.

Requirements:
- Class named `LRUCache`
- Constructor takes `capacity: int`
- Method `get(key: int) -> int`: Return value if exists, else -1
- Method `put(key: int, value: int) -> None`: Insert or update
- Evict least recently used when capacity exceeded""",

    feedback_wrapper="""TEST FAILURE:
{feedback}

Instruction: The implementation failed the test suite. Analyze the failure above and fix your code."""
)
```

---

## Workflow Configuration

```python
from atomicguard import (
    OllamaGenerator,
    ActionPair,
    Workflow,
    WorkflowStep,
    InMemoryArtifactDAG,
    ArtifactDAGInterface,
)

def create_tdd_workflow(artifact_dag: ArtifactDAGInterface = None) -> Workflow:
    """Configure TDD workflow with two steps and structured prompts."""

    if artifact_dag is None:
        artifact_dag = InMemoryArtifactDAG()  # Default for testing

    generator = OllamaGenerator(model="qwen2.5-coder:7b")

    # Step 1: Generate Tests (with structured prompt)
    test_action_pair = ActionPair(
        precondition=lambda sw: True,
        generator=generator,
        guard=TestSyntaxGuard(),
        prompt_template=TEST_PROMPT_TEMPLATE
    )

    # Step 2: Generate Implementation (with structured prompt)
    impl_action_pair = ActionPair(
        precondition=lambda sw: sw.is_satisfied('g_test'),
        generator=generator,
        guard=TDDImplementationGuard(),
        prompt_template=IMPL_PROMPT_TEMPLATE
    )

    steps = [
        WorkflowStep(
            name="Generate Tests",
            action_pair=test_action_pair,
            guard_id='g_test',
            guard_artifact_deps=()
        ),
        WorkflowStep(
            name="Generate Implementation",
            action_pair=impl_action_pair,
            guard_id='g_impl',
            guard_artifact_deps=('g_test',)
        )
    ]

    return Workflow(
        steps=steps,
        artifact_dag=artifact_dag,
        rmax=3
    )
```

---

## Example Execution

```python
from atomicguard import FilesystemArtifactDAG

def main():
    artifact_dag = FilesystemArtifactDAG(base_path="./tdd_workflow_artifacts")
    workflow = create_tdd_workflow(artifact_dag)

    # Execute TDD workflow
    # Step 1 uses TEST_SPECIFICATION
    # Step 2 uses IMPL_SPECIFICATION
    # For simplicity, we pass combined spec; in production, each step would have its own

    result = workflow.execute(TEST_SPECIFICATION)  # Step 1

    if result.status == WorkflowStatus.SUCCESS:
        print("=== TDD WORKFLOW SUCCESS ===")
        print(f"\n--- Tests (g_test) ---")
        print(result.artifacts['g_test'].content)
        print(f"\n--- Implementation (g_impl) ---")
        print(result.artifacts['g_impl'].content)
    elif result.status == WorkflowStatus.ESCALATION:
        print(f"=== ESCALATION REQUIRED at {result.failed_step} ===")
        print(f"Reason: {result.escalation_feedback}")
    else:
        print(f"=== WORKFLOW FAILED at {result.failed_step} ===")
        print("\nProvenance:")
        for artifact, feedback in result.provenance:
            print(f"  Feedback: {feedback}")

if __name__ == "__main__":
    main()
```

---

## Expected Percept Sequences

### Happy Path

| Step | Percept | Action |
|------|---------|--------|
| 1 | [Ψ_test] | QUERY-LLM |
| 1 | [a_test: "class TestLRUCache..."] | EXECUTE-GUARD(a_test, deps={}) |
| 1 | [(⊤, ∅)] | ADVANCE-STATE → sw[g_test ↦ ⊤] |
| 2 | [Ψ_impl] | QUERY-LLM |
| 2 | [a_impl: "class LRUCache..."] | EXECUTE-GUARD(a_impl, deps={test: a_test}) |
| 2 | [(⊤, ∅)] | ADVANCE-STATE → sw[g_impl ↦ ⊤] |
| - | [sw: {g_test ↦ ⊤, g_impl ↦ ⊤}] | REPLY-TO-USER(artifacts) |

### Refinement at Step 2

| Step | Percept | Action |
|------|---------|--------|
| 1 | [Ψ_test] | QUERY-LLM |
| 1 | [a_test, (⊤, ∅)] | ADVANCE-STATE |
| 2 | [Ψ_impl] | QUERY-LLM |
| 2 | [a_impl₁] | EXECUTE-GUARD(a_impl₁, deps={test: a_test}) |
| 2 | [(⊥, "Test failed: get(2) should return -1")] | REFINE-CONTEXT |
| 2 | [H: {(a_impl₁, φ₁)}] | QUERY-LLM |
| 2 | [a_impl₂] | EXECUTE-GUARD(a_impl₂, deps={test: a_test}) |
| 2 | [(⊤, ∅)] | ADVANCE-STATE |

---

## Workflow State Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    WORKFLOW STATE                        │
├─────────────────────────────────────────────────────────┤
│  sw = { g_test: ⊥, g_impl: ⊥ }                          │
│                      │                                   │
│                      ▼                                   │
│  ┌─────────────────────────────────────┐                │
│  │ Step 1: Generate Tests              │                │
│  │ precondition: True                  │                │
│  │ guard_artifact_deps: ()             │                │
│  └─────────────────────────────────────┘                │
│                      │                                   │
│                      ▼ (on g_test = ⊤)                  │
│  sw = { g_test: ⊤, g_impl: ⊥ }                          │
│  artifacts = { g_test: a_test }                         │
│                      │                                   │
│                      ▼                                   │
│  ┌─────────────────────────────────────┐                │
│  │ Step 2: Generate Implementation     │                │
│  │ precondition: sw.is_satisfied('g_test') │            │
│  │ guard_artifact_deps: ('g_test',)    │                │
│  │ deps extracted: {test: a_test}      │                │
│  └─────────────────────────────────────┘                │
│                      │                                   │
│                      ▼ (on g_impl = ⊤)                  │
│  sw = { g_test: ⊤, g_impl: ⊤ }                          │
│  artifacts = { g_test: a_test, g_impl: a_impl }         │
│                      │                                   │
│                      ▼                                   │
│                   SUCCESS                                │
└─────────────────────────────────────────────────────────┘
```

---

## Sample Generated Artifacts

### Test Artifact (Step 1)

```python
class TestLRUCache:
    def test_basic_operations(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        assert cache.get(1) == 1
        assert cache.get(2) == 2

    def test_capacity_eviction(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.put(3, 3)  # Evicts key 1
        assert cache.get(1) == -1
        assert cache.get(3) == 3

    def test_get_updates_recency(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.get(1)  # Makes key 1 recent
        cache.put(3, 3)  # Should evict key 2, not key 1
        assert cache.get(1) == 1
        assert cache.get(2) == -1

    def test_update_existing(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(1, 10)
        assert cache.get(1) == 10
```

### Implementation Artifact (Step 2)

```python
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            lru = self.order.pop(0)
            del self.cache[lru]
        self.cache[key] = value
        self.order.append(key)
```
