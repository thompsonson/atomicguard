"""Workflow execution trace models (Extension 10, Definition 50-51)."""

from dataclasses import dataclass
from enum import Enum


class WorkflowEventType(str, Enum):
    """Types of workflow execution events."""

    STEP_START = "STEP_START"
    STEP_PASS = "STEP_PASS"
    STEP_FAIL = "STEP_FAIL"
    STAGNATION = "STAGNATION"
    ESCALATE = "ESCALATE"
    CASCADE_INVALIDATE = "CASCADE_INVALIDATE"


@dataclass(frozen=True)
class EscalationEventRecord:
    """Escalation event details (Definition 51)."""

    targets: tuple[str, ...]
    invalidated: tuple[str, ...]
    e_count: int
    e_max: int
    failure_summary: str
    trigger: str  # "STAGNATION" or "RMAX_EXHAUSTED"
    stagnant_guard: str | None = None


@dataclass(frozen=True)
class WorkflowEvent:
    """Single workflow state transition (Definition 50).

    Represents an atomic event in the workflow execution trace,
    capturing state changes for observability and debugging.
    """

    event_id: str
    event_type: WorkflowEventType
    action_pair_id: str
    workflow_id: str
    guard_name: str | None = None
    verdict: str | None = None  # "PASS", "FAIL", "FATAL"
    attempt: int | None = None
    e_count: int | None = None
    escalation: EscalationEventRecord | None = None
    summary: str = ""
    created_at: str = ""  # ISO 8601
