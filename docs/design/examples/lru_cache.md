# Dual-State Agent: LRU Cache Example

> **Runnable Example**: For a simpler runnable version of this pattern, see [examples/basics/01_mock.py](../../examples/basics/01_mock.py) which demonstrates the same DualStateAgent retry loop with a mock generator.

## Task Overview

From paper Table 1:

- **Task**: LRU Cache
- **Prior Knowledge**: High (standard Hash Map + Linked List pattern)
- **Guard Role**: Drift prevention
- **Expected Behavior**: High baseline success; guard catches stochastic drift

---

## Specification (Ψ)

```python
LRU_CACHE_SPECIFICATION = """
Implement an LRU (Least Recently Used) Cache in Python.

Requirements:
1. Class named `LRUCache`
2. Constructor takes `capacity: int`
3. Method `get(key: int) -> int`: Return value if key exists, else -1
4. Method `put(key: int, value: int) -> None`: Insert or update key-value pair
5. When capacity exceeded, evict least recently used item
6. Both operations must be O(1) time complexity

Return only the Python code, no explanations.
"""
```

---

## Prompt Template

```python
LRU_CACHE_PROMPT_TEMPLATE = PromptTemplate(
    role="""You are a Python Developer implementing a standard data structure.
Focus on correctness and clarity.""",

    constraints="""1. Class must be named `LRUCache`
2. Constructor takes `capacity: int`
3. Both get() and put() must be O(1) time complexity
4. Return only Python code, no explanations""",

    task="""Implement an LRU (Least Recently Used) Cache.

Methods:
- `get(key: int) -> int`: Return value if key exists, else -1
- `put(key: int, value: int) -> None`: Insert or update key-value pair
- When capacity exceeded, evict least recently used item""",

    feedback_wrapper="""TEST FAILURE:
{feedback}

Instruction: Your implementation failed. Analyze the error and fix the specific issue."""
)
```

---

## Guard Implementation

```python
class LRUCacheTestGuard(GuardInterface):
    """Validates LRU Cache implementation via test execution."""

    TEST_CODE = '''
# Test basic operations
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1, "Failed: get(1) should return 1"

# Test capacity eviction
cache.put(3, 3)  # Evicts key 2
assert cache.get(2) == -1, "Failed: get(2) should return -1 after eviction"
assert cache.get(3) == 3, "Failed: get(3) should return 3"

# Test LRU ordering update on get
cache.put(4, 4)  # Evicts key 1 (not 3, because get(3) made it recent)
assert cache.get(1) == -1, "Failed: get(1) should return -1 after eviction"
assert cache.get(3) == 3, "Failed: get(3) should still return 3"
assert cache.get(4) == 4, "Failed: get(4) should return 4"

# Test update existing key
cache.put(3, 30)
assert cache.get(3) == 30, "Failed: get(3) should return updated value 30"
'''

    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        """No dependencies required for this guard."""
        namespace = {}
        try:
            exec(artifact.content, namespace)

            if 'LRUCache' not in namespace:
                return GuardResult(
                    passed=False,
                    feedback="Missing LRUCache class definition"
                )

            exec(self.TEST_CODE, namespace)
            return GuardResult(passed=True)

        except AssertionError as e:
            return GuardResult(passed=False, feedback=str(e))
        except SyntaxError as e:
            return GuardResult(
                passed=False,
                feedback=f"Syntax error at line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"{type(e).__name__}: {e}"
            )
```

---

## Example Execution

```python
from atomicguard import (
    DualStateAgent,
    ActionPair,
    OllamaGenerator,
    InMemoryArtifactDAG,
    RmaxExhausted,
    PromptTemplate,
)

def main():
    # Precondition: always true for single action pair
    precondition = lambda sw: True

    # Initialize components
    generator = OllamaGenerator(model="qwen2.5-coder:7b")
    guard = LRUCacheTestGuard()
    artifact_dag = InMemoryArtifactDAG()  # Simple for testing

    # Create atomic action pair with structured prompt
    action_pair = ActionPair(
        precondition=precondition,
        generator=generator,
        guard=guard,
        prompt_template=LRU_CACHE_PROMPT_TEMPLATE
    )

    # Initialize agent
    agent = DualStateAgent(
        action_pair=action_pair,
        artifact_dag=artifact_dag,
        rmax=3
    )

    # Execute (no dependencies for this simple case)
    try:
        artifact = agent.execute(LRU_CACHE_PROMPT_TEMPLATE.task)
        print("SUCCESS")
        print(f"Artifact ID: {artifact.artifact_id}")
        print(f"Content:\n{artifact.content}")

    except RmaxExhausted as e:
        print(f"FAILED: {e}")
        print("\nProvenance:")
        for i, (artifact, feedback) in enumerate(e.provenance):
            print(f"\n--- Attempt {i+1} ---")
            print(f"Feedback: {feedback}")

if __name__ == "__main__":
    main()
```

---

## Expected Percept Sequences

### Happy Path (Pass on first attempt)

| Percept | Action |
|---------|--------|
| [Ψ: LRU_CACHE_SPECIFICATION] | QUERY-LLM |
| [Ψ, a₁: "class LRUCache:..."] | EXECUTE-GUARD |
| [Ψ, a₁, (⊤, ∅)] | ADVANCE-STATE |
| [sw: {lru_guard ↦ ⊤}] | REPLY-TO-USER(a₁) |

### Refinement (Missing method)

| Percept | Action |
|---------|--------|
| [Ψ: LRU_CACHE_SPECIFICATION] | QUERY-LLM |
| [Ψ, a₁: "class LRUCache: def **init**..."] | EXECUTE-GUARD |
| [Ψ, a₁, (⊥, "AttributeError: 'LRUCache' object has no attribute 'get'")] | REFINE-CONTEXT |
| [Ψ, H: {(a₁, φ₁)}] | QUERY-LLM |
| [Ψ, a₂: "class LRUCache: def **init**... def get... def put..."] | EXECUTE-GUARD |
| [Ψ, a₂, (⊤, ∅)] | ADVANCE-STATE |
| [sw: {lru_guard ↦ ⊤}] | REPLY-TO-USER(a₂) |

### Refinement (Incorrect eviction logic)

| Percept | Action |
|---------|--------|
| [Ψ: LRU_CACHE_SPECIFICATION] | QUERY-LLM |
| [Ψ, a₁: "class LRUCache:..."] | EXECUTE-GUARD |
| [Ψ, a₁, (⊥, "Failed: get(2) should return -1 after eviction")] | REFINE-CONTEXT |
| [Ψ, H: {(a₁, φ₁)}] | QUERY-LLM |
| [Ψ, a₂: "class LRUCache:... (fixed eviction)"] | EXECUTE-GUARD |
| [Ψ, a₂, (⊤, ∅)] | ADVANCE-STATE |
| [sw: {lru_guard ↦ ⊤}] | REPLY-TO-USER(a₂) |

---

## Sample Valid Output

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

Note: This naive implementation is O(n) for order updates. A production implementation would use OrderedDict or doubly-linked list for O(1). The guard tests correctness, not complexity.
