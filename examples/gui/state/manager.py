"""Thread-safe state management for the workflow monitor."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomicguard import WorkflowResult

from .events import (
    AnyEvent,
    GenerationEvent,
    GuardResultEvent,
    LogEvent,
    StepCompletedEvent,
    StepStartedEvent,
    WorkflowCompletedEvent,
    WorkflowStartedEvent,
)

if TYPE_CHECKING:
    pass


@dataclass
class StepStatus:
    """Status information for a single workflow step."""

    step_id: str
    guard_type: str
    requires: tuple[str, ...]
    status: str = "pending"  # pending, running, success, failed
    current_attempt: int = 0
    max_attempts: int = 3
    last_feedback: str = ""
    artifact_id: str | None = None


@dataclass
class ExecutionState:
    """Complete state snapshot for the GUI."""

    # Workflow info
    workflow_name: str = ""
    model: str = ""
    specification_preview: str = ""
    is_running: bool = False

    # Step tracking
    steps: dict[str, StepStatus] = field(default_factory=dict)
    current_step: str | None = None

    # Events and logs
    events: list[AnyEvent] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    # Results
    result: WorkflowResult | None = None
    error: str | None = None
    duration: float = 0.0


class StateManager:
    """
    Thread-safe state manager for the workflow monitor GUI.

    Provides methods to update state based on events and
    retrieve snapshots for UI rendering.
    """

    def __init__(self, max_logs: int = 500) -> None:
        """
        Initialize the state manager.

        Args:
            max_logs: Maximum number of log entries to keep
        """
        self._state = ExecutionState()
        self._lock = threading.Lock()
        self._max_logs = max_logs
        self._subscribers: list[Callable[[AnyEvent], None]] = []
        self._log_filter: str = "All"
        self._artifacts_refreshed: bool = False

    def reset(self) -> None:
        """Reset state for a new execution."""
        with self._lock:
            self._state = ExecutionState()
            self._artifacts_refreshed = False

    def subscribe(self, callback: Callable[[AnyEvent], None]) -> None:
        """Register an event subscriber."""
        self._subscribers.append(callback)

    def update(self, event: AnyEvent) -> None:
        """
        Update state based on an event.

        This method is thread-safe and can be called from
        background execution threads.
        """
        with self._lock:
            self._state.events.append(event)
            self._dispatch_event(event)

        # Notify subscribers outside the lock
        for subscriber in self._subscribers:
            subscriber(event)

    def _dispatch_event(self, event: AnyEvent) -> None:
        """Dispatch event to appropriate handler (called within lock)."""
        if isinstance(event, WorkflowStartedEvent):
            self._handle_workflow_started(event)
        elif isinstance(event, StepStartedEvent):
            self._handle_step_started(event)
        elif isinstance(event, GenerationEvent):
            self._handle_generation(event)
        elif isinstance(event, GuardResultEvent):
            self._handle_guard_result(event)
        elif isinstance(event, StepCompletedEvent):
            self._handle_step_completed(event)
        elif isinstance(event, WorkflowCompletedEvent):
            self._handle_workflow_completed(event)
        elif isinstance(event, LogEvent):
            self._handle_log(event)

    def _handle_workflow_started(self, event: WorkflowStartedEvent) -> None:
        """Handle workflow started event."""
        self._state.workflow_name = event.workflow_name
        self._state.model = event.model
        self._state.specification_preview = event.specification_preview
        self._state.is_running = True
        self._state.error = None
        self._state.result = None

    def _handle_step_started(self, event: StepStartedEvent) -> None:
        """Handle step started event."""
        self._state.current_step = event.step_id

        # Create or update step status
        if event.step_id not in self._state.steps:
            self._state.steps[event.step_id] = StepStatus(
                step_id=event.step_id,
                guard_type=event.guard_type,
                requires=event.requires,
            )

        step = self._state.steps[event.step_id]
        step.status = "running"
        step.current_attempt = event.attempt

    def _handle_generation(self, event: GenerationEvent) -> None:
        """Handle artifact generation event."""
        if event.step_id in self._state.steps:
            step = self._state.steps[event.step_id]
            step.current_attempt = event.attempt
            step.artifact_id = event.artifact_id

    def _handle_guard_result(self, event: GuardResultEvent) -> None:
        """Handle guard validation result event."""
        if event.step_id in self._state.steps:
            step = self._state.steps[event.step_id]
            step.current_attempt = event.attempt
            if not event.passed:
                step.last_feedback = event.feedback

    def _handle_step_completed(self, event: StepCompletedEvent) -> None:
        """Handle step completed event."""
        if event.step_id in self._state.steps:
            step = self._state.steps[event.step_id]
            step.status = "success" if event.success else "failed"
            step.artifact_id = event.artifact_id
            step.current_attempt = event.total_attempts

        # Clear current step if it's the one that completed
        if self._state.current_step == event.step_id:
            self._state.current_step = None

    def _handle_workflow_completed(self, event: WorkflowCompletedEvent) -> None:
        """Handle workflow completed event."""
        self._state.is_running = False
        self._state.duration = event.total_duration
        self._state.current_step = None

    def _handle_log(self, event: LogEvent) -> None:
        """Handle log event."""
        log_line = f"[{event.timestamp[11:19]}] {event.level:8s} | {event.message}"
        self._state.logs.append(log_line)

        # Trim logs if over limit
        if len(self._state.logs) > self._max_logs:
            self._state.logs = self._state.logs[-self._max_logs :]

    def set_result(self, result: WorkflowResult) -> None:
        """Set the workflow result (called after execution completes)."""
        with self._lock:
            self._state.result = result

    def set_error(self, error: str) -> None:
        """Set an execution error."""
        with self._lock:
            self._state.error = error
            self._state.is_running = False

    def set_steps_from_config(
        self, action_pairs: dict[str, dict], rmax: int = 3
    ) -> None:
        """Initialize step statuses from workflow configuration."""
        with self._lock:
            self._state.steps = {}
            for step_id, config in action_pairs.items():
                guard_type = config.get("guard", "unknown")
                if guard_type == "composite":
                    guards = config.get("guards", [])
                    guard_type = f"composite({', '.join(guards)})"
                requires = tuple(config.get("requires", []))

                self._state.steps[step_id] = StepStatus(
                    step_id=step_id,
                    guard_type=guard_type,
                    requires=requires,
                    max_attempts=rmax,
                )

    def get_snapshot(self) -> ExecutionState:
        """
        Get a deep copy of the current state.

        This is safe to use for UI rendering as it won't
        be modified by background threads.
        """
        with self._lock:
            return copy.deepcopy(self._state)

    def get_logs(self, level_filter: str | None = None) -> list[str]:
        """Get log entries, optionally filtered by level."""
        with self._lock:
            if level_filter and level_filter != "All":
                return [
                    log
                    for log in self._state.logs
                    if f"] {level_filter}" in log or f"]{level_filter}" in log
                ]
            return list(self._state.logs)

    def set_log_filter(self, level: str) -> None:
        """Set the current log filter level."""
        with self._lock:
            self._log_filter = level

    def get_log_filter(self) -> str:
        """Get the current log filter level."""
        with self._lock:
            return self._log_filter

    def should_refresh_artifacts(self) -> bool:
        """Check if artifacts need refreshing after workflow completion."""
        with self._lock:
            # Only refresh once after workflow completes with a result
            if self._state.result is not None and not self._artifacts_refreshed:
                self._artifacts_refreshed = True
                return True
            return False
