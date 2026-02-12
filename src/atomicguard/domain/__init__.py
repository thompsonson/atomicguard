"""
Domain layer for the Dual-State Framework.

Contains core business logic with no external dependencies.
"""

from atomicguard.domain.exceptions import (
    EscalationRequired,
    RmaxExhausted,
    StagnationDetected,
)
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
    SubGuardOutcome,
    WorkflowResult,
    WorkflowState,
    WorkflowStatus,
)
from atomicguard.domain.prompts import (
    PromptTemplate,
    StepDefinition,
    TaskDefinition,
)

__all__ = [
    # Models
    "Artifact",
    "ArtifactStatus",
    "ContextSnapshot",
    "FeedbackEntry",
    "Context",
    "AmbientEnvironment",
    "GuardResult",
    "SubGuardOutcome",
    "WorkflowState",
    "WorkflowResult",
    "WorkflowStatus",
    # Prompts and Tasks (structures only, no content)
    "PromptTemplate",
    "StepDefinition",
    "TaskDefinition",
    # Interfaces
    "GeneratorInterface",
    "GuardInterface",
    "ArtifactDAGInterface",
    # Exceptions
    "RmaxExhausted",
    "EscalationRequired",
    "StagnationDetected",
]
