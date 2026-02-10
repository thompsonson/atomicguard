"""Tests for workflow event models (Extension 10)."""

import pytest

from atomicguard.domain.workflow_event import (
    EscalationEventRecord,
    WorkflowEvent,
    WorkflowEventType,
)


class TestWorkflowEventType:
    """Tests for WorkflowEventType enum."""

    def test_event_type_values(self):
        """Event types have correct string values."""
        assert WorkflowEventType.STEP_START.value == "STEP_START"
        assert WorkflowEventType.STEP_PASS.value == "STEP_PASS"
        assert WorkflowEventType.STEP_FAIL.value == "STEP_FAIL"
        assert WorkflowEventType.STAGNATION.value == "STAGNATION"
        assert WorkflowEventType.ESCALATE.value == "ESCALATE"
        assert WorkflowEventType.CASCADE_INVALIDATE.value == "CASCADE_INVALIDATE"

    def test_event_type_is_str_enum(self):
        """WorkflowEventType is a string enum."""
        assert isinstance(WorkflowEventType.STEP_START, str)
        assert WorkflowEventType.STEP_START == "STEP_START"


class TestWorkflowEvent:
    """Tests for WorkflowEvent dataclass."""

    def test_workflow_event_frozen(self):
        """WorkflowEvent is immutable."""
        event = WorkflowEvent(
            event_id="evt-1",
            event_type=WorkflowEventType.STEP_START,
            action_pair_id="step1",
            workflow_id="wf-1",
        )

        with pytest.raises(AttributeError):
            event.event_id = "evt-2"  # type: ignore

    def test_workflow_event_all_fields(self):
        """WorkflowEvent stores all fields."""
        escalation = EscalationEventRecord(
            targets=("t1", "t2"),
            invalidated=("i1",),
            e_count=1,
            e_max=3,
            failure_summary="Test summary",
            trigger="STAGNATION",
            stagnant_guard="Guard1",
        )

        event = WorkflowEvent(
            event_id="evt-1",
            event_type=WorkflowEventType.ESCALATE,
            action_pair_id="step1",
            workflow_id="wf-1",
            guard_name="TestGuard",
            verdict="FAIL",
            attempt=3,
            e_count=1,
            escalation=escalation,
            summary="Test summary",
            created_at="2024-01-01T00:00:00Z",
        )

        assert event.event_id == "evt-1"
        assert event.event_type == WorkflowEventType.ESCALATE
        assert event.action_pair_id == "step1"
        assert event.workflow_id == "wf-1"
        assert event.guard_name == "TestGuard"
        assert event.verdict == "FAIL"
        assert event.attempt == 3
        assert event.e_count == 1
        assert event.escalation == escalation
        assert event.summary == "Test summary"
        assert event.created_at == "2024-01-01T00:00:00Z"

    def test_workflow_event_defaults(self):
        """WorkflowEvent has correct defaults."""
        event = WorkflowEvent(
            event_id="evt-1",
            event_type=WorkflowEventType.STEP_START,
            action_pair_id="step1",
            workflow_id="wf-1",
        )

        assert event.guard_name is None
        assert event.verdict is None
        assert event.attempt is None
        assert event.e_count is None
        assert event.escalation is None
        assert event.summary == ""
        assert event.created_at == ""


class TestEscalationEventRecord:
    """Tests for EscalationEventRecord dataclass."""

    def test_escalation_event_record_frozen(self):
        """EscalationEventRecord is immutable."""
        record = EscalationEventRecord(
            targets=("t1",),
            invalidated=("i1",),
            e_count=1,
            e_max=3,
            failure_summary="Test",
            trigger="STAGNATION",
        )

        with pytest.raises(AttributeError):
            record.e_count = 2  # type: ignore

    def test_escalation_event_record_fields(self):
        """EscalationEventRecord stores all fields."""
        record = EscalationEventRecord(
            targets=("target1", "target2"),
            invalidated=("inv1", "inv2", "inv3"),
            e_count=2,
            e_max=5,
            failure_summary="Failed due to repeated syntax errors",
            trigger="STAGNATION",
            stagnant_guard="SyntaxGuard",
        )

        assert record.targets == ("target1", "target2")
        assert record.invalidated == ("inv1", "inv2", "inv3")
        assert record.e_count == 2
        assert record.e_max == 5
        assert record.failure_summary == "Failed due to repeated syntax errors"
        assert record.trigger == "STAGNATION"
        assert record.stagnant_guard == "SyntaxGuard"

    def test_escalation_event_record_default_stagnant_guard(self):
        """EscalationEventRecord defaults stagnant_guard to None."""
        record = EscalationEventRecord(
            targets=(),
            invalidated=(),
            e_count=1,
            e_max=1,
            failure_summary="",
            trigger="RMAX_EXHAUSTED",
        )

        assert record.stagnant_guard is None
