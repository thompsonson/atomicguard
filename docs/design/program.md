# Dual-State Agent: Agent Program

## Background for Implementation Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.12 | Project standard |
| Async | No | PoC simplicity |
| LLM | LiteLLM → Ollama | Infrastructure abstraction |
| Architecture | DDD/SOLID/Clean | Inspectable, testable |
| Artifact Storage | Git-backed DAG | Full provenance |
| Error Handling | Exception on Rmax | Fail with provenance |

---

## Complexity Analysis

From paper Section 5:

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Planning | O(\|T\|) where \|T\| ≤ N × Rmax | N = guards, Rmax = retry limit |
| Per transition | O(1) + generator cost | Generator (LLM) dominates |
| Total worst-case | O(\|Sreach\| × Rmax × \|G\|) | Tractable for sparse workflows |

**Bottleneck**: LLM inference (seconds) dominates all other operations (microseconds).

---

## Domain Objects

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any

@dataclass(frozen=True)
class Artifact:
    """Immutable generated output with version tracking."""
    content: str
    version: int
    artifact_id: str                          # Unique identifier for DAG lookup
    parent_version: Optional[int] = None

@dataclass(frozen=True)
class GuardResult:
    """Immutable guard validation outcome."""
    passed: bool          # v ∈ {⊥, ⊤}
    feedback: str = ""    # φ ∈ Σ*
    fatal: bool = False   # ⊥_fatal - skip retry, escalate to human

@dataclass(frozen=True)
class AmbientEnvironment:
    """
    Ambient Environment E = ⟨R, Ω⟩
    Provides read-only access to finalized artifacts.
    """
    repository: 'ArtifactDAGInterface'  # R - versioned artifact DAG
    constraints: str = ""                # Ω - global constraints

@dataclass(frozen=True)
class PromptTemplate:
    """
    Structured prompt template for generator.
    Separates role, constraints, and task with semantic feedback wrapping.
    """
    role: str                                    # System role description
    constraints: str                             # Architectural/task constraints
    task: str                                    # Current task instruction
    feedback_wrapper: str = "GUARD REJECTION:\n{feedback}\nInstruction: Address the rejection above."

    def render(self, context: 'Context') -> str:
        """Render template with context, including feedback history."""
        parts = [
            f"# ROLE\n{self.role}",
            f"# CONSTRAINTS\n{self.constraints}",
        ]

        # Add ambient context if present
        if context.ambient.constraints:
            parts.append(f"# CONTEXT\n{context.ambient.constraints}")

        # Add feedback history with semantic wrapping
        if context.feedback_history:
            parts.append("# HISTORY (Context Refinement)")
            for i, (artifact_content, feedback) in enumerate(context.feedback_history):
                wrapped = self.feedback_wrapper.format(feedback=feedback)
                parts.append(f"--- Attempt {i+1} ---\n{wrapped}")

        parts.append(f"# TASK\n{self.task}")
        return "\n\n".join(parts)

@dataclass(frozen=True)
class Context:
    """
    Immutable hierarchical context composition.
    Paper Definition 3: C = ⟨E, Clocal, Hfeedback⟩
    """
    ambient: AmbientEnvironment                    # E
    specification: str                             # Ψ (part of Clocal)
    current_artifact: Optional[str] = None         # ak (part of Clocal)
    feedback_history: Tuple[Tuple[str, str], ...] = ()  # Hfeedback

@dataclass
class WorkflowState:
    """Mutable workflow state tracking guard satisfaction."""
    guards: Dict[str, bool] = field(default_factory=dict)  # σ : G → {⊥, ⊤}
    artifact_ids: Dict[str, str] = field(default_factory=dict)  # guard_id → artifact_id

    def is_satisfied(self, guard_id: str) -> bool:
        return self.guards.get(guard_id, False)

    def satisfy(self, guard_id: str, artifact_id: str) -> None:
        self.guards[guard_id] = True
        self.artifact_ids[guard_id] = artifact_id

    def get_artifact_id(self, guard_id: str) -> Optional[str]:
        return self.artifact_ids.get(guard_id)

@dataclass
class EnvironmentState:
    """Mutable environment state container."""
    artifact: Optional[Artifact] = None
    context: Optional[Context] = None
    retry_count: int = 0
```

---

## Interfaces

```python
from abc import ABC, abstractmethod

class GeneratorInterface(ABC):
    """Abstract interface for LLM generation."""

    @abstractmethod
    def generate(self, context: Context) -> Artifact:
        """Generate artifact from context."""
        pass

class GuardInterface(ABC):
    """Abstract interface for artifact validation."""

    @abstractmethod
    def validate(self, artifact: Artifact) -> GuardResult:
        """Validate artifact, return (passed, feedback)."""
        pass
```

---

## ActionPair (Executable Unit)

```python
from typing import Callable, Tuple, Dict, Optional

class PreconditionNotMet(Exception):
    """Raised when action pair precondition fails."""
    pass

class ActionPair:
    """
    Atomic generation-verification transaction.
    Paper Definition 6: A = ⟨ρ, agen, G⟩

    Extended with PromptTemplate for structured context composition.
    """

    def __init__(
        self,
        precondition: Callable[[WorkflowState], bool],  # ρ
        generator: GeneratorInterface,                   # agen
        guard: GuardInterface,                           # G
        prompt_template: Optional[PromptTemplate] = None # Structured prompt
    ):
        self._precondition = precondition
        self._generator = generator
        self._guard = guard
        self._prompt_template = prompt_template

    def can_execute(self, workflow_state: WorkflowState) -> bool:
        """Check if precondition satisfied."""
        return self._precondition(workflow_state)

    def execute(
        self,
        workflow_state: WorkflowState,
        context: Context,
        dependencies: Dict[str, Artifact] = None
    ) -> Tuple[Artifact, GuardResult]:
        """
        Atomic execution: generate then validate.
        No partial execution possible.

        Args:
            workflow_state: Current workflow state
            context: Generation context
            dependencies: Scoped artifacts for guard validation
        """
        if not self._precondition(workflow_state):
            raise PreconditionNotMet(
                f"Precondition not satisfied for action pair"
            )

        dependencies = dependencies or {}

        artifact = self._generator.generate(context, self._prompt_template)
        result = self._guard.validate(artifact, **dependencies)
        return artifact, result
```

---

## Main Agent Class

```python
from typing import List, Tuple as TypingTuple, Dict
import uuid

class RmaxExhausted(Exception):
    """Raised when retry budget exhausted."""
    def __init__(self, message: str, provenance: List[TypingTuple[Artifact, str]]):
        super().__init__(message)
        self.provenance = provenance

class EscalationRequired(Exception):
    """Raised when guard returns ⊥_fatal - human intervention needed."""
    def __init__(self, artifact: Artifact, feedback: str):
        super().__init__(feedback)
        self.artifact = artifact
        self.feedback = feedback

class DualStateAgent:
    """
    Dual-State Agent implementing guard-validated generation.
    Paper: Managing the Stochastic (Thompson, 2025)
    """

    def __init__(
        self,
        action_pair: ActionPair,
        artifact_dag: ArtifactDAGInterface,
        rmax: int = 3,
        constraints: str = ""
    ):
        self._action_pair = action_pair
        self._artifact_dag = artifact_dag
        self._rmax = rmax
        self._constraints = constraints

        # Dual state initialization
        self._workflow_state = WorkflowState()
        self._environment_state = EnvironmentState()
        self._feedback_history: List[TypingTuple[Artifact, str]] = []

    def execute(
        self,
        specification: str,
        dependencies: Dict[str, Artifact] = None
    ) -> Artifact:
        """
        Execute action pair with retry loop.

        Args:
            specification: Task specification Ψ
            dependencies: Scoped artifacts for guard validation

        Returns:
            Verified artifact or raises RmaxExhausted
        """
        dependencies = dependencies or {}
        context = self._compose_context(specification)

        while self._environment_state.retry_count <= self._rmax:
            # Atomic execution: generate + validate
            artifact, result = self._action_pair.execute(
                self._workflow_state,
                context,
                dependencies
            )

            # Store in DAG with metadata
            self._artifact_dag.store(
                artifact,
                metadata="" if result.passed else result.feedback
            )

            if result.passed:
                # Definition 8: Advance workflow state
                self._advance_state(artifact)
                return artifact
            elif result.fatal:
                # ⊥_fatal: Non-recoverable failure - escalate immediately
                raise EscalationRequired(artifact, result.feedback)
            else:
                # Definition 5: Refine context, workflow stable
                context = self._refine_context(
                    specification,
                    artifact,
                    result.feedback
                )

        # Rmax exhausted
        raise RmaxExhausted(
            f"Failed after {self._rmax} retries",
            provenance=self._feedback_history
        )

    def _compose_context(self, specification: str) -> Context:
        """Definition 3: Hierarchical context composition."""
        ambient = AmbientEnvironment(
            repository=self._artifact_dag,
            constraints=self._constraints
        )
        return Context(
            ambient=ambient,
            specification=specification,
            current_artifact=None,
            feedback_history=tuple()
        )

    def _refine_context(
        self,
        specification: str,
        artifact: Artifact,
        feedback: str
    ) -> Context:
        """Definition 5: Context refinement on guard failure."""
        self._feedback_history.append((artifact, feedback))
        self._environment_state.retry_count += 1

        ambient = AmbientEnvironment(
            repository=self._artifact_dag,
            constraints=self._constraints
        )
        return Context(
            ambient=ambient,
            specification=specification,
            current_artifact=artifact.content,
            feedback_history=tuple(
                (a.content, f) for a, f in self._feedback_history
            )
        )

    def _advance_state(self, artifact: Artifact) -> None:
        """Definition 8: State transition on guard success."""
        self._environment_state.artifact = artifact
        self._environment_state.retry_count = 0
        self._feedback_history.clear()
```

---

## Interfaces

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class GeneratorInterface(ABC):
    """
    Abstract interface for LLM generation.

    Note (Hierarchical Composition & Semantic Agency):
        The generator is not constrained to a single inference step. It may be
        instantiated as an autonomous Semantic Agent (ReAct loop, CoT reasoning,
        multi-tool orchestration) operating within the stochastic environment.
        From the workflow's perspective, this agentic process is atomic — the
        Workflow State tracks only the final artifact's validity via the Guard.
    """

    @abstractmethod
    def generate(self, context: Context, template: Optional[PromptTemplate] = None) -> Artifact:
        """
        Generate artifact from context.

        Args:
            context: Hierarchical context (E, Clocal, Hfeedback)
            template: Optional structured prompt template
        """
        pass

class GuardInterface(ABC):
    """
    Abstract interface for artifact validation.
    Guards receive scoped dependencies extracted by workflow.
    """

    @abstractmethod
    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        """
        Validate artifact with scoped dependencies.

        Args:
            artifact: The artifact to validate
            **dependencies: Named artifacts from R, scoped by workflow

        Returns:
            GuardResult with validation outcome and feedback
        """
        pass

class ArtifactDAGInterface(ABC):
    """Abstract interface for artifact version storage."""

    @abstractmethod
    def store(self, artifact: Artifact, metadata: str) -> str:
        """Store artifact, return version identifier."""
        pass

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Artifact:
        """Retrieve artifact by ID."""
        pass

    @abstractmethod
    def get_provenance(self, artifact_id: str) -> List[Artifact]:
        """Get full history leading to artifact."""
        pass
```

**Remark (Hierarchical Composition & Semantic Agency)**: The generator function `agen` is not constrained to be a single LLM inference step. It may be instantiated as an autonomous **Semantic Agent** (e.g., a ReAct loop, Chain-of-Thought reasoning, or multi-tool orchestration) that operates purely within the stochastic environment state. From the Dual-State Workflow's perspective:

1. The semantic agent's internal reasoning is **opaque** to the Workflow State
2. The generator invocation is **atomic** — the workflow sees only the final artifact
3. The Guard `G` validates the output regardless of how it was produced

This enables hierarchical composition: a workflow step's generator can itself be a complete agentic system, with the Guard serving as the interface contract.

---

## Infrastructure Implementations

### In-Memory DAG (Testing)

```python
class InMemoryArtifactDAG(ArtifactDAGInterface):
    """Simple in-memory DAG for testing. No persistence."""

    def __init__(self):
        self._artifacts: Dict[str, Artifact] = {}
        self._metadata: Dict[str, str] = {}

    def store(self, artifact: Artifact, metadata: str) -> str:
        self._artifacts[artifact.artifact_id] = artifact
        self._metadata[artifact.artifact_id] = metadata
        return artifact.artifact_id

    def get_artifact(self, artifact_id: str) -> Artifact:
        if artifact_id not in self._artifacts:
            raise KeyError(f"Artifact not found: {artifact_id}")
        return self._artifacts[artifact_id]

    def get_provenance(self, artifact_id: str) -> List[Artifact]:
        result = []
        current = self._artifacts.get(artifact_id)
        while current:
            result.append(current)
            if current.parent_version is None:
                break
            # Find parent by version
            parent = next(
                (a for a in self._artifacts.values()
                 if a.version == current.parent_version),
                None
            )
            current = parent
        return list(reversed(result))
```

### Git-Backed DAG (Production)

```python
from git import Repo
from pathlib import Path
import uuid

class GitArtifactDAG(ArtifactDAGInterface):
    """Git-backed artifact DAG implementation."""

    def __init__(self, repo_path: str, artifact_dir: str = "artifacts"):
        self._repo_path = Path(repo_path)
        self._artifact_dir = artifact_dir
        self._repo = Repo.init(self._repo_path)
        (self._repo_path / artifact_dir).mkdir(exist_ok=True)

    def store(self, artifact: Artifact, metadata: str) -> str:
        artifact_path = self._repo_path / self._artifact_dir / f"{artifact.artifact_id}.py"
        artifact_path.write_text(artifact.content)
        self._repo.index.add([str(artifact_path.relative_to(self._repo_path))])
        message = f"{artifact.artifact_id}: {metadata or 'passed'}"
        commit = self._repo.index.commit(message)
        return commit.hexsha

    def get_artifact(self, artifact_id: str) -> Artifact:
        artifact_path = self._repo_path / self._artifact_dir / f"{artifact_id}.py"
        content = artifact_path.read_text()
        return Artifact(
            content=content,
            version=0,  # Version tracking via git
            artifact_id=artifact_id
        )

    def get_provenance(self, artifact_id: str) -> List[Artifact]:
        artifact_path = f"{self._artifact_dir}/{artifact_id}.py"
        artifacts = []
        for i, commit in enumerate(self._repo.iter_commits(paths=artifact_path)):
            content = self._repo.git.show(f"{commit.hexsha}:{artifact_path}")
            artifacts.append(Artifact(
                content=content,
                version=i,
                artifact_id=artifact_id,
                parent_version=i - 1 if i > 0 else None
            ))
        return list(reversed(artifacts))
```

```python
from litellm import completion
import uuid

class LiteLLMGenerator(GeneratorInterface):
    """LiteLLM implementation for Ollama backend."""

    def __init__(self, model: str = "ollama/qwen2.5-coder:7b"):
        self._model = model
        self._version_counter = 0

    def generate(self, context: Context, template: Optional[PromptTemplate] = None) -> Artifact:
        """Generate artifact using template if provided, else fallback to basic prompt."""

        if template:
            prompt = template.render(context)
        else:
            prompt = self._build_basic_prompt(context)

        response = completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}]
        )

        self._version_counter += 1
        parent = self._version_counter - 1 if self._version_counter > 1 else None

        return Artifact(
            content=response.choices[0].message.content,
            version=self._version_counter,
            artifact_id=str(uuid.uuid4()),
            parent_version=parent
        )

    def _build_basic_prompt(self, context: Context) -> str:
        """Fallback prompt builder when no template provided."""
        parts = [context.specification]

        if context.current_artifact:
            parts.append(f"\nPrevious attempt:\n{context.current_artifact}")

        if context.feedback_history:
            feedback_text = "\n".join(
                f"Attempt {i+1} feedback: {f}"
                for i, (_, f) in enumerate(context.feedback_history)
            )
            parts.append(f"\nFeedback history:\n{feedback_text}")

        return "\n".join(parts)
```

---

## Guard Library (Shared)

```python
import ast

class SyntaxGuard(GuardInterface):
    """Validates Python syntax. No dependencies required."""

    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        try:
            ast.parse(artifact.content)
            return GuardResult(passed=True)
        except SyntaxError as e:
            return GuardResult(
                passed=False,
                feedback=f"Syntax error at line {e.lineno}: {e.msg}"
            )

class TestGuard(GuardInterface):
    """
    Validates via test execution.
    Can use static test_code OR dynamic test artifact from dependencies.
    """

    def __init__(self, test_code: str = None):
        self._static_test_code = test_code

    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        # Use dependency test artifact if provided, else static
        test_artifact = dependencies.get('test')
        test_code = test_artifact.content if test_artifact else self._static_test_code

        if not test_code:
            return GuardResult(
                passed=False,
                feedback="No test code provided (static or via dependency)"
            )

        namespace = {}
        try:
            exec(artifact.content, namespace)
            exec(test_code, namespace)
            return GuardResult(passed=True)
        except AssertionError as e:
            return GuardResult(passed=False, feedback=str(e))
        except Exception as e:
            return GuardResult(passed=False, feedback=f"{type(e).__name__}: {e}")

class TDDTestGuard(GuardInterface):
    """
    TDD Guard: requires test artifact from prior step.
    Validates implementation against generated tests.
    """

    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        test_artifact = dependencies.get('test')

        if not test_artifact:
            return GuardResult(
                passed=False,
                feedback="TDD Guard requires 'test' dependency"
            )

        namespace = {}
        try:
            exec(artifact.content, namespace)
            exec(test_artifact.content, namespace)
            return GuardResult(passed=True)
        except AssertionError as e:
            return GuardResult(passed=False, feedback=f"Test failed: {e}")
        except Exception as e:
            return GuardResult(passed=False, feedback=f"{type(e).__name__}: {e}")
```

---

## Workflow Orchestration

```python
from enum import Enum
from typing import List, Tuple as TypingTuple, Dict
from dataclasses import dataclass

@dataclass(frozen=True)
class WorkflowStep:
    """Immutable workflow step configuration."""
    name: str
    action_pair: ActionPair
    guard_id: str
    guard_artifact_deps: Tuple[str, ...] = ()  # guard_ids of required artifacts

class WorkflowStatus(Enum):
    """Workflow execution outcome."""
    SUCCESS = "success"      # All steps completed
    FAILED = "failed"        # Rmax exhausted on a step
    ESCALATION = "escalation"  # Fatal guard triggered

@dataclass(frozen=True)
class WorkflowResult:
    """Immutable workflow execution result."""
    status: WorkflowStatus
    artifacts: Dict[str, Artifact]  # guard_id → artifact
    failed_step: Optional[str] = None
    provenance: TypingTuple[TypingTuple[Artifact, str], ...] = ()
    escalation_artifact: Optional[Artifact] = None  # Artifact that triggered escalation
    escalation_feedback: str = ""  # Fatal feedback message

class Workflow:
    """
    Orchestrates sequential ActionPair execution.
    Extracts scoped dependencies for each guard.
    """

    def __init__(
        self,
        steps: List[WorkflowStep],
        artifact_dag: ArtifactDAGInterface,
        rmax: int = 3,
        constraints: str = ""
    ):
        self._steps = steps
        self._artifact_dag = artifact_dag
        self._rmax = rmax
        self._constraints = constraints
        self._workflow_state = WorkflowState()

    def execute(self, specification: str) -> WorkflowResult:
        """
        Execute all steps sequentially.
        Returns WorkflowResult with artifacts or failure provenance.
        """
        artifacts: Dict[str, Artifact] = {}

        for step in self._steps:
            # Check precondition
            if not step.action_pair.can_execute(self._workflow_state):
                return WorkflowResult(
                    status=WorkflowStatus.FAILED,
                    artifacts=artifacts,
                    failed_step=step.name,
                    provenance=(("Precondition not met", ""),)
                )

            # Extract scoped dependencies for this step's guard
            dependencies = self._extract_dependencies(step.guard_artifact_deps, artifacts)

            # Create step agent
            agent = DualStateAgent(
                action_pair=step.action_pair,
                artifact_dag=self._artifact_dag,
                rmax=self._rmax,
                constraints=self._constraints
            )

            try:
                artifact = agent.execute(specification, dependencies)
                artifacts[step.guard_id] = artifact

                # Advance workflow state with artifact tracking
                self._workflow_state.satisfy(step.guard_id, artifact.artifact_id)

            except EscalationRequired as e:
                return WorkflowResult(
                    status=WorkflowStatus.ESCALATION,
                    artifacts=artifacts,
                    failed_step=step.guard_id,
                    escalation_artifact=e.artifact,
                    escalation_feedback=e.feedback,
                )

            except RmaxExhausted as e:
                return WorkflowResult(
                    status=WorkflowStatus.FAILED,
                    artifacts=artifacts,
                    failed_step=step.name,
                    provenance=tuple(e.provenance)
                )

        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            artifacts=artifacts
        )

    def _extract_dependencies(
        self,
        dep_guard_ids: Tuple[str, ...],
        artifacts: Dict[str, Artifact]
    ) -> Dict[str, Artifact]:
        """
        Extract scoped artifacts for guard validation.
        Maps dependency names to artifacts from completed steps.
        """
        dependencies = {}
        for guard_id in dep_guard_ids:
            if guard_id in artifacts:
                # Use simple name (e.g., 'test' from 'g_test')
                dep_name = guard_id.replace('g_', '')
                dependencies[dep_name] = artifacts[guard_id]
        return dependencies
```

---

## Architectural Decisions Summary

| Decision | Choice | Paper Reference |
|----------|--------|-----------------|
| Dual state separation | WorkflowState + EnvironmentState | Definition 1 |
| Atomic transactions | ActionPair.execute() | Definition 6 |
| Context composition | AmbientEnvironment + Clocal + Hfeedback | Definition 3 |
| Workflow stability on failure | retry_count in EnvironmentState | Definition 4 |
| Context refinement | feedback_history accumulation | Definition 5 |
| Artifact provenance | ArtifactDAGInterface (git-backed) | Definition 2 |
| Fail with provenance | RmaxExhausted exception, WorkflowResult | Algorithm 1, line 9 |
| Fatal escalation | GuardResult.fatal + EscalationRequired | Definition 6 (⊥_fatal) |
| Workflow outcome | WorkflowStatus enum (SUCCESS, FAILED, ESCALATION) | Definition 6 |
| Infrastructure abstraction | Interface + Implementation pattern | SOLID/DDD |
| Workflow orchestration | Workflow class (SRP) | Definition 9 |
| Guard input scoping | Explicit dependencies via WorkflowStep | Remark (Guard Input Scoping) |
| Artifact tracking | WorkflowState.artifact_ids | Definition 2 |
| Structured prompts | PromptTemplate with semantic feedback | Section 8.1.1 (Coach) |

**Remark (Guard Input Scoping)**: While Definition 6 provides guards access to the full context C, well-designed guards accept only minimal required inputs. The Workflow extracts specific artifacts from R based on `guard_artifact_deps` and passes them explicitly, preserving guard simplicity and testability.

**Remark (Prompt Templates)**: PromptTemplate separates role, constraints, and task with semantic feedback wrapping. This implements the "Coach" concept from Section 8.1.1, transforming raw guard feedback into actionable refinement instructions.
