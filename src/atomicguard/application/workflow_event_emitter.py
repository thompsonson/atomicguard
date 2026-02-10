"""Workflow event emission service (Extension 10)."""

import uuid
from datetime import datetime, timezone

from atomicguard.domain.interfaces import WorkflowEventStoreInterface
from atomicguard.domain.workflow_event import (
    EscalationEventRecord,
    WorkflowEvent,
    WorkflowEventType,
)


class WorkflowEventEmitter:
    """Emits workflow events to a store.

    Provides convenience methods for emitting common workflow events
    during execution, handling ID generation and timestamps.
    """

    def __init__(
        self, event_store: WorkflowEventStoreInterface, workflow_id: str
    ) -> None:
        self._store = event_store
        self._workflow_id = workflow_id

    def _emit(self, event: WorkflowEvent) -> str:
        return self._store.store_event(event)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def step_start(self, ap_id: str, attempt: int) -> None:
        """Emit STEP_START event when beginning a step execution."""
        self._emit(
            WorkflowEvent(
                event_id=str(uuid.uuid4()),
                event_type=WorkflowEventType.STEP_START,
                action_pair_id=ap_id,
                workflow_id=self._workflow_id,
                attempt=attempt,
                created_at=self._now(),
            )
        )

    def step_pass(self, ap_id: str, guard_name: str | None, attempt: int) -> None:
        """Emit STEP_PASS event when step completes successfully."""
        self._emit(
            WorkflowEvent(
                event_id=str(uuid.uuid4()),
                event_type=WorkflowEventType.STEP_PASS,
                action_pair_id=ap_id,
                workflow_id=self._workflow_id,
                guard_name=guard_name,
                verdict="PASS",
                attempt=attempt,
                created_at=self._now(),
            )
        )

    def step_fail(
        self, ap_id: str, guard_name: str | None, attempt: int, feedback: str
    ) -> None:
        """Emit STEP_FAIL event when step fails validation."""
        self._emit(
            WorkflowEvent(
                event_id=str(uuid.uuid4()),
                event_type=WorkflowEventType.STEP_FAIL,
                action_pair_id=ap_id,
                workflow_id=self._workflow_id,
                guard_name=guard_name,
                verdict="FAIL",
                attempt=attempt,
                summary=feedback[:500],
                created_at=self._now(),
            )
        )

    def stagnation(self, ap_id: str, guard_name: str | None, summary: str) -> None:
        """Emit STAGNATION event when stagnation is detected."""
        self._emit(
            WorkflowEvent(
                event_id=str(uuid.uuid4()),
                event_type=WorkflowEventType.STAGNATION,
                action_pair_id=ap_id,
                workflow_id=self._workflow_id,
                guard_name=guard_name,
                summary=summary,
                created_at=self._now(),
            )
        )

    def escalate(self, ap_id: str, record: EscalationEventRecord) -> None:
        """Emit ESCALATE event when escalation occurs."""
        self._emit(
            WorkflowEvent(
                event_id=str(uuid.uuid4()),
                event_type=WorkflowEventType.ESCALATE,
                action_pair_id=ap_id,
                workflow_id=self._workflow_id,
                e_count=record.e_count,
                escalation=record,
                created_at=self._now(),
            )
        )

    def cascade_invalidate(self, ap_id: str) -> None:
        """Emit CASCADE_INVALIDATE event when a step is invalidated."""
        self._emit(
            WorkflowEvent(
                event_id=str(uuid.uuid4()),
                event_type=WorkflowEventType.CASCADE_INVALIDATE,
                action_pair_id=ap_id,
                workflow_id=self._workflow_id,
                created_at=self._now(),
            )
        )
