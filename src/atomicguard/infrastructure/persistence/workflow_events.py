"""Workflow event store implementations (Extension 10)."""

import json
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import WorkflowEventStoreInterface
from atomicguard.domain.workflow_event import (
    EscalationEventRecord,
    WorkflowEvent,
    WorkflowEventType,
)


class InMemoryWorkflowEventStore(WorkflowEventStoreInterface):
    """In-memory implementation for testing."""

    def __init__(self) -> None:
        self._events: list[WorkflowEvent] = []

    def store_event(self, event: WorkflowEvent) -> str:
        self._events.append(event)
        return event.event_id

    def get_events(
        self,
        workflow_id: str,
        event_type: WorkflowEventType | None = None,
        action_pair_id: str | None = None,
    ) -> list[WorkflowEvent]:
        return sorted(
            [
                e
                for e in self._events
                if e.workflow_id == workflow_id
                and (event_type is None or e.event_type == event_type)
                and (action_pair_id is None or e.action_pair_id == action_pair_id)
            ],
            key=lambda e: e.created_at,
        )

    def get_escalation_events(self, workflow_id: str) -> list[WorkflowEvent]:
        return self.get_events(workflow_id, WorkflowEventType.ESCALATE)


class FilesystemWorkflowEventStore(WorkflowEventStoreInterface):
    """Filesystem implementation storing events as JSONL."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.events_dir = base_path / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def _get_workflow_file(self, workflow_id: str) -> Path:
        return self.events_dir / f"{workflow_id}.jsonl"

    def store_event(self, event: WorkflowEvent) -> str:
        path = self._get_workflow_file(event.workflow_id)
        with open(path, "a") as f:
            f.write(json.dumps(self._event_to_dict(event)) + "\n")
        return event.event_id

    def get_events(
        self,
        workflow_id: str,
        event_type: WorkflowEventType | None = None,
        action_pair_id: str | None = None,
    ) -> list[WorkflowEvent]:
        path = self._get_workflow_file(workflow_id)
        if not path.exists():
            return []
        events: list[WorkflowEvent] = []
        with open(path) as f:
            for line in f:
                event = self._dict_to_event(json.loads(line))
                if event_type and event.event_type != event_type:
                    continue
                if action_pair_id and event.action_pair_id != action_pair_id:
                    continue
                events.append(event)
        return sorted(events, key=lambda e: e.created_at)

    def get_escalation_events(self, workflow_id: str) -> list[WorkflowEvent]:
        return self.get_events(workflow_id, WorkflowEventType.ESCALATE)

    def _event_to_dict(self, event: WorkflowEvent) -> dict[str, Any]:
        """Serialize event to dict."""
        data: dict[str, Any] = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "action_pair_id": event.action_pair_id,
            "workflow_id": event.workflow_id,
            "guard_name": event.guard_name,
            "verdict": event.verdict,
            "attempt": event.attempt,
            "e_count": event.e_count,
            "summary": event.summary,
            "created_at": event.created_at,
        }
        if event.escalation:
            data["escalation"] = {
                "targets": list(event.escalation.targets),
                "invalidated": list(event.escalation.invalidated),
                "e_count": event.escalation.e_count,
                "e_max": event.escalation.e_max,
                "failure_summary": event.escalation.failure_summary,
                "trigger": event.escalation.trigger,
                "stagnant_guard": event.escalation.stagnant_guard,
            }
        return data

    def _dict_to_event(self, data: dict[str, Any]) -> WorkflowEvent:
        """Deserialize dict to event."""
        escalation = None
        if data.get("escalation"):
            esc_data = data["escalation"]
            escalation = EscalationEventRecord(
                targets=tuple(esc_data["targets"]),
                invalidated=tuple(esc_data["invalidated"]),
                e_count=esc_data["e_count"],
                e_max=esc_data["e_max"],
                failure_summary=esc_data["failure_summary"],
                trigger=esc_data["trigger"],
                stagnant_guard=esc_data.get("stagnant_guard"),
            )
        return WorkflowEvent(
            event_id=data["event_id"],
            event_type=WorkflowEventType(data["event_type"]),
            action_pair_id=data["action_pair_id"],
            workflow_id=data["workflow_id"],
            guard_name=data.get("guard_name"),
            verdict=data.get("verdict"),
            attempt=data.get("attempt"),
            e_count=data.get("e_count"),
            escalation=escalation,
            summary=data.get("summary", ""),
            created_at=data.get("created_at", ""),
        )
