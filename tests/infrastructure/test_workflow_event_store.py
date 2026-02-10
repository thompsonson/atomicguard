"""Tests for workflow event store implementations (Extension 10)."""

import tempfile
from pathlib import Path

from atomicguard.domain.workflow_event import (
    EscalationEventRecord,
    WorkflowEvent,
    WorkflowEventType,
)
from atomicguard.infrastructure.persistence.workflow_events import (
    FilesystemWorkflowEventStore,
    InMemoryWorkflowEventStore,
)


def make_event(
    workflow_id: str = "wf-1",
    event_type: WorkflowEventType = WorkflowEventType.STEP_START,
    action_pair_id: str = "step1",
    created_at: str = "2024-01-01T00:00:00Z",
    **kwargs,
) -> WorkflowEvent:
    """Create a test workflow event."""
    return WorkflowEvent(
        event_id=f"evt-{created_at}",
        event_type=event_type,
        action_pair_id=action_pair_id,
        workflow_id=workflow_id,
        created_at=created_at,
        **kwargs,
    )


class TestInMemoryWorkflowEventStore:
    """Tests for InMemoryWorkflowEventStore."""

    def test_store_and_retrieve_event(self):
        """Store and retrieve events."""
        store = InMemoryWorkflowEventStore()
        event = make_event()

        event_id = store.store_event(event)

        assert event_id == event.event_id
        events = store.get_events("wf-1")
        assert len(events) == 1
        assert events[0] == event

    def test_get_events_filters_by_workflow_id(self):
        """get_events filters by workflow_id."""
        store = InMemoryWorkflowEventStore()
        store.store_event(make_event(workflow_id="wf-1"))
        store.store_event(make_event(workflow_id="wf-2"))

        events = store.get_events("wf-1")

        assert len(events) == 1
        assert events[0].workflow_id == "wf-1"

    def test_get_events_filters_by_event_type(self):
        """get_events filters by event_type."""
        store = InMemoryWorkflowEventStore()
        store.store_event(make_event(event_type=WorkflowEventType.STEP_START))
        store.store_event(make_event(event_type=WorkflowEventType.STEP_PASS))

        events = store.get_events("wf-1", event_type=WorkflowEventType.STEP_START)

        assert len(events) == 1
        assert events[0].event_type == WorkflowEventType.STEP_START

    def test_get_events_filters_by_action_pair_id(self):
        """get_events filters by action_pair_id."""
        store = InMemoryWorkflowEventStore()
        store.store_event(make_event(action_pair_id="step1"))
        store.store_event(make_event(action_pair_id="step2"))

        events = store.get_events("wf-1", action_pair_id="step1")

        assert len(events) == 1
        assert events[0].action_pair_id == "step1"

    def test_events_ordered_by_created_at(self):
        """Events returned in chronological order."""
        store = InMemoryWorkflowEventStore()
        store.store_event(make_event(created_at="2024-01-03T00:00:00Z"))
        store.store_event(make_event(created_at="2024-01-01T00:00:00Z"))
        store.store_event(make_event(created_at="2024-01-02T00:00:00Z"))

        events = store.get_events("wf-1")

        assert events[0].created_at == "2024-01-01T00:00:00Z"
        assert events[1].created_at == "2024-01-02T00:00:00Z"
        assert events[2].created_at == "2024-01-03T00:00:00Z"

    def test_get_escalation_events(self):
        """get_escalation_events returns only ESCALATE events."""
        store = InMemoryWorkflowEventStore()
        store.store_event(make_event(event_type=WorkflowEventType.STEP_START))
        store.store_event(make_event(event_type=WorkflowEventType.ESCALATE))
        store.store_event(make_event(event_type=WorkflowEventType.STEP_PASS))

        events = store.get_escalation_events("wf-1")

        assert len(events) == 1
        assert events[0].event_type == WorkflowEventType.ESCALATE


class TestFilesystemWorkflowEventStore:
    """Tests for FilesystemWorkflowEventStore."""

    def test_store_and_retrieve_event(self):
        """Store and retrieve events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemWorkflowEventStore(Path(tmpdir))
            event = make_event()

            event_id = store.store_event(event)

            assert event_id == event.event_id
            events = store.get_events("wf-1")
            assert len(events) == 1
            assert events[0].event_id == event.event_id

    def test_events_persist_across_instances(self):
        """Events persist across store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Store event with first instance
            store1 = FilesystemWorkflowEventStore(path)
            store1.store_event(make_event())

            # Retrieve with second instance
            store2 = FilesystemWorkflowEventStore(path)
            events = store2.get_events("wf-1")

            assert len(events) == 1

    def test_events_ordered_by_created_at(self):
        """Events returned in chronological order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemWorkflowEventStore(Path(tmpdir))
            store.store_event(make_event(created_at="2024-01-03T00:00:00Z"))
            store.store_event(make_event(created_at="2024-01-01T00:00:00Z"))
            store.store_event(make_event(created_at="2024-01-02T00:00:00Z"))

            events = store.get_events("wf-1")

            assert events[0].created_at == "2024-01-01T00:00:00Z"
            assert events[1].created_at == "2024-01-02T00:00:00Z"
            assert events[2].created_at == "2024-01-03T00:00:00Z"

    def test_serializes_escalation_record(self):
        """EscalationEventRecord is correctly serialized and deserialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemWorkflowEventStore(Path(tmpdir))

            escalation = EscalationEventRecord(
                targets=("target1", "target2"),
                invalidated=("inv1",),
                e_count=2,
                e_max=5,
                failure_summary="Test failure",
                trigger="STAGNATION",
                stagnant_guard="Guard1",
            )

            event = make_event(
                event_type=WorkflowEventType.ESCALATE,
                escalation=escalation,
            )
            store.store_event(event)

            events = store.get_events("wf-1")
            assert len(events) == 1
            assert events[0].escalation is not None
            assert events[0].escalation.targets == ("target1", "target2")
            assert events[0].escalation.stagnant_guard == "Guard1"

    def test_get_events_for_missing_workflow(self):
        """get_events returns empty list for missing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemWorkflowEventStore(Path(tmpdir))

            events = store.get_events("nonexistent")

            assert events == []

    def test_get_events_filters_correctly(self):
        """get_events applies all filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemWorkflowEventStore(Path(tmpdir))
            store.store_event(
                make_event(
                    event_type=WorkflowEventType.STEP_START, action_pair_id="step1"
                )
            )
            store.store_event(
                make_event(
                    event_type=WorkflowEventType.STEP_PASS, action_pair_id="step1"
                )
            )
            store.store_event(
                make_event(
                    event_type=WorkflowEventType.STEP_START, action_pair_id="step2"
                )
            )

            events = store.get_events(
                "wf-1",
                event_type=WorkflowEventType.STEP_START,
                action_pair_id="step1",
            )

            assert len(events) == 1
            assert events[0].event_type == WorkflowEventType.STEP_START
            assert events[0].action_pair_id == "step1"
