"""Tests for WorkflowEventEmitter (Extension 10)."""

from atomicguard.application.workflow_event_emitter import WorkflowEventEmitter
from atomicguard.domain.workflow_event import (
    EscalationEventRecord,
    WorkflowEventType,
)
from atomicguard.infrastructure.persistence.workflow_events import (
    InMemoryWorkflowEventStore,
)


class TestWorkflowEventEmitter:
    """Tests for WorkflowEventEmitter."""

    def test_step_start_creates_event(self):
        """step_start creates STEP_START event."""
        store = InMemoryWorkflowEventStore()
        emitter = WorkflowEventEmitter(store, "wf-1")

        emitter.step_start("step1", attempt=1)

        events = store.get_events("wf-1")
        assert len(events) == 1
        assert events[0].event_type == WorkflowEventType.STEP_START
        assert events[0].action_pair_id == "step1"
        assert events[0].attempt == 1
        assert events[0].workflow_id == "wf-1"

    def test_step_pass_creates_event(self):
        """step_pass creates STEP_PASS event with guard_name."""
        store = InMemoryWorkflowEventStore()
        emitter = WorkflowEventEmitter(store, "wf-1")

        emitter.step_pass("step1", guard_name="TestGuard", attempt=2)

        events = store.get_events("wf-1")
        assert len(events) == 1
        assert events[0].event_type == WorkflowEventType.STEP_PASS
        assert events[0].guard_name == "TestGuard"
        assert events[0].verdict == "PASS"
        assert events[0].attempt == 2

    def test_step_fail_creates_event(self):
        """step_fail creates STEP_FAIL event with feedback."""
        store = InMemoryWorkflowEventStore()
        emitter = WorkflowEventEmitter(store, "wf-1")

        emitter.step_fail("step1", guard_name="TestGuard", attempt=3, feedback="Error!")

        events = store.get_events("wf-1")
        assert len(events) == 1
        assert events[0].event_type == WorkflowEventType.STEP_FAIL
        assert events[0].verdict == "FAIL"
        assert events[0].summary == "Error!"

    def test_step_fail_truncates_long_feedback(self):
        """step_fail truncates feedback to 500 chars."""
        store = InMemoryWorkflowEventStore()
        emitter = WorkflowEventEmitter(store, "wf-1")

        long_feedback = "x" * 1000
        emitter.step_fail("step1", None, 1, long_feedback)

        events = store.get_events("wf-1")
        assert len(events[0].summary) == 500

    def test_stagnation_creates_event(self):
        """stagnation creates STAGNATION event."""
        store = InMemoryWorkflowEventStore()
        emitter = WorkflowEventEmitter(store, "wf-1")

        emitter.stagnation("step1", guard_name="SyntaxGuard", summary="Repeated errors")

        events = store.get_events("wf-1")
        assert len(events) == 1
        assert events[0].event_type == WorkflowEventType.STAGNATION
        assert events[0].guard_name == "SyntaxGuard"
        assert events[0].summary == "Repeated errors"

    def test_escalate_creates_event(self):
        """escalate creates ESCALATE with EscalationEventRecord."""
        store = InMemoryWorkflowEventStore()
        emitter = WorkflowEventEmitter(store, "wf-1")

        record = EscalationEventRecord(
            targets=("target1", "target2"),
            invalidated=("inv1",),
            e_count=1,
            e_max=3,
            failure_summary="Test failure",
            trigger="STAGNATION",
            stagnant_guard="Guard1",
        )
        emitter.escalate("step1", record)

        events = store.get_events("wf-1")
        assert len(events) == 1
        assert events[0].event_type == WorkflowEventType.ESCALATE
        assert events[0].e_count == 1
        assert events[0].escalation == record
        assert events[0].escalation.targets == ("target1", "target2")

    def test_cascade_invalidate_creates_event(self):
        """cascade_invalidate creates CASCADE_INVALIDATE event."""
        store = InMemoryWorkflowEventStore()
        emitter = WorkflowEventEmitter(store, "wf-1")

        emitter.cascade_invalidate("step2")

        events = store.get_events("wf-1")
        assert len(events) == 1
        assert events[0].event_type == WorkflowEventType.CASCADE_INVALIDATE
        assert events[0].action_pair_id == "step2"

    def test_events_have_timestamps(self):
        """All events have created_at timestamps."""
        store = InMemoryWorkflowEventStore()
        emitter = WorkflowEventEmitter(store, "wf-1")

        emitter.step_start("step1", 1)
        emitter.step_pass("step1", None, 1)

        events = store.get_events("wf-1")
        for event in events:
            assert event.created_at != ""
            # Should be ISO format
            assert "T" in event.created_at

    def test_events_have_unique_ids(self):
        """Each event has a unique event_id."""
        store = InMemoryWorkflowEventStore()
        emitter = WorkflowEventEmitter(store, "wf-1")

        emitter.step_start("step1", 1)
        emitter.step_start("step1", 2)
        emitter.step_pass("step1", None, 2)

        events = store.get_events("wf-1")
        event_ids = [e.event_id for e in events]
        assert len(event_ids) == len(set(event_ids))  # All unique

    def test_multiple_workflows_isolated(self):
        """Events for different workflows are isolated."""
        store = InMemoryWorkflowEventStore()
        emitter1 = WorkflowEventEmitter(store, "wf-1")
        emitter2 = WorkflowEventEmitter(store, "wf-2")

        emitter1.step_start("step1", 1)
        emitter2.step_start("step2", 1)

        events1 = store.get_events("wf-1")
        events2 = store.get_events("wf-2")

        assert len(events1) == 1
        assert len(events2) == 1
        assert events1[0].action_pair_id == "step1"
        assert events2[0].action_pair_id == "step2"
