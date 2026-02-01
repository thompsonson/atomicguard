"""
AtomicGuard: Dual-State Framework for LLM Output Management.

A framework for managing stochastic LLM outputs through deterministic
control flow, implementing the formal model from Thompson (2025).

Example:
    from atomicguard import Workflow, ActionPair, PromptTemplate
    from atomicguard.guards import SyntaxGuard
    from atomicguard.infrastructure import OllamaGenerator

    generator = OllamaGenerator(model="qwen2.5-coder:7b")
    guard = SyntaxGuard()
    template = PromptTemplate(role="code generator", constraints="write clean Python code", task="generate code")
    action_pair = ActionPair(generator=generator, guard=guard, prompt_template=template)

    workflow = Workflow(rmax=3)
    workflow.add_step('g_code', action_pair=action_pair)
    result = workflow.execute("Write a function that adds two numbers")
"""

# Domain models (most commonly used)
# Application layer (orchestration)
from atomicguard.application.action_pair import ActionPair
from atomicguard.application.agent import DualStateAgent
from atomicguard.application.workflow import Workflow, WorkflowStep

# Domain exceptions
from atomicguard.domain.exceptions import EscalationRequired, RmaxExhausted

# Domain interfaces (for type hints and custom implementations)
from atomicguard.domain.interfaces import (
    ArtifactDAGInterface,
    GeneratorInterface,
    GuardInterface,
)
from atomicguard.domain.models import (
    AmbientEnvironment,
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    FeedbackEntry,
    GuardResult,
    WorkflowResult,
    WorkflowState,
    WorkflowStatus,
)

# Prompts and tasks (structures only - content defined by calling applications)
from atomicguard.domain.prompts import (
    PromptTemplate,
    StepDefinition,
    TaskDefinition,
)

# Guards (commonly composed)
from atomicguard.guards import (
    CompositeGuard,
    DynamicTestGuard,
    HumanReviewGuard,
    ImportGuard,
    SyntaxGuard,
    TestGuard,
)
from atomicguard.infrastructure.llm import (
    MockGenerator,
    OllamaGenerator,
)

# Infrastructure (explicit import encouraged for dependency injection)
from atomicguard.infrastructure.persistence import (
    FilesystemArtifactDAG,
    InMemoryArtifactDAG,
)

__version__ = "1.2.0"

__all__ = [
    # Version
    "__version__",
    # Domain models
    "Artifact",
    "ArtifactStatus",
    "ContextSnapshot",
    "FeedbackEntry",
    "Context",
    "AmbientEnvironment",
    "GuardResult",
    "WorkflowState",
    "WorkflowResult",
    "WorkflowStatus",
    # Prompts and tasks (structures only)
    "PromptTemplate",
    "StepDefinition",
    "TaskDefinition",
    # Domain interfaces
    "GeneratorInterface",
    "GuardInterface",
    "ArtifactDAGInterface",
    # Domain exceptions
    "RmaxExhausted",
    "EscalationRequired",
    # Application layer
    "ActionPair",
    "DualStateAgent",
    "Workflow",
    "WorkflowStep",
    # Infrastructure - Persistence
    "InMemoryArtifactDAG",
    "FilesystemArtifactDAG",
    # Infrastructure - LLM
    "MockGenerator",
    "OllamaGenerator",
    # Guards
    "CompositeGuard",
    "SyntaxGuard",
    "ImportGuard",
    "TestGuard",
    "DynamicTestGuard",
    "HumanReviewGuard",
]
