"""Event types for workflow monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WorkflowEvent:
    """Base event class for all workflow events."""

    timestamp: str
    event_type: str

    @staticmethod
    def now() -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()


@dataclass(frozen=True)
class WorkflowStartedEvent(WorkflowEvent):
    """Emitted when workflow execution begins."""

    workflow_name: str
    step_count: int
    model: str
    specification_preview: str  # First N chars of specification


@dataclass(frozen=True)
class StepStartedEvent(WorkflowEvent):
    """Emitted when a workflow step begins execution."""

    step_id: str
    guard_type: str
    requires: tuple[str, ...]
    attempt: int


@dataclass(frozen=True)
class GenerationEvent(WorkflowEvent):
    """Emitted when an artifact is generated."""

    step_id: str
    artifact_id: str
    attempt: int
    content_size: int


@dataclass(frozen=True)
class GuardResultEvent(WorkflowEvent):
    """Emitted after guard validation."""

    step_id: str
    artifact_id: str
    passed: bool
    feedback: str
    fatal: bool
    attempt: int


@dataclass(frozen=True)
class StepCompletedEvent(WorkflowEvent):
    """Emitted when a step completes (success or failure)."""

    step_id: str
    success: bool
    artifact_id: str | None
    total_attempts: int
    duration_seconds: float


@dataclass(frozen=True)
class WorkflowCompletedEvent(WorkflowEvent):
    """Emitted when workflow execution completes."""

    status: str  # SUCCESS, FAILED, ESCALATION
    failed_step: str | None
    total_duration: float
    total_artifacts: int


@dataclass(frozen=True)
class LogEvent(WorkflowEvent):
    """Emitted for log messages."""

    level: str  # DEBUG, INFO, WARNING, ERROR
    message: str
    logger_name: str


# Type alias for all event types
AnyEvent = (
    WorkflowStartedEvent
    | StepStartedEvent
    | GenerationEvent
    | GuardResultEvent
    | StepCompletedEvent
    | WorkflowCompletedEvent
    | LogEvent
)
