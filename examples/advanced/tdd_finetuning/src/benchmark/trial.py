"""Trial data structures and result tracking."""

from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass
class AttemptResult:
    """Result of a single generation attempt."""
    attempt_number: int
    generated_code: str
    guard_passed: bool
    guard_feedback: str
    timestamp: str = ""


@dataclass
class PhaseResult:
    """Result of a single phase (g_test or g_impl)."""
    phase_name: str  # "g_test" or "g_impl"
    success: bool
    attempts: list[AttemptResult] = field(default_factory=list)
    final_code: str = ""
    total_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase_name": self.phase_name,
            "success": self.success,
            "attempts": [asdict(a) for a in self.attempts],
            "final_code": self.final_code,
            "total_attempts": self.total_attempts,
        }


@dataclass
class TrialResult:
    """Result of a complete TDD workflow trial."""
    task_id: str
    trial_number: int
    workflow_success: bool  # True if both g_test and g_impl succeeded

    # Phase results
    g_test_result: PhaseResult | None = None
    g_impl_result: PhaseResult | None = None

    # Summary statistics
    total_attempts: int = 0
    execution_time_seconds: float = 0.0

    # Quality flags for training data extraction
    first_attempt_test_success: bool = False  # g_test passed on first attempt
    validated_success: bool = False  # g_test succeeded AND g_impl succeeded

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "trial_number": self.trial_number,
            "workflow_success": self.workflow_success,
            "g_test_result": self.g_test_result.to_dict() if self.g_test_result else None,
            "g_impl_result": self.g_impl_result.to_dict() if self.g_impl_result else None,
            "total_attempts": self.total_attempts,
            "execution_time_seconds": self.execution_time_seconds,
            "first_attempt_test_success": self.first_attempt_test_success,
            "validated_success": self.validated_success,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialResult":
        """Create TrialResult from dictionary."""
        # Reconstruct PhaseResult objects
        g_test_data = data.get("g_test_result")
        g_impl_data = data.get("g_impl_result")

        g_test_result = None
        if g_test_data:
            attempts = [
                AttemptResult(**attempt) for attempt in g_test_data["attempts"]
            ]
            g_test_result = PhaseResult(
                phase_name=g_test_data["phase_name"],
                success=g_test_data["success"],
                attempts=attempts,
                final_code=g_test_data["final_code"],
                total_attempts=g_test_data["total_attempts"],
            )

        g_impl_result = None
        if g_impl_data:
            attempts = [
                AttemptResult(**attempt) for attempt in g_impl_data["attempts"]
            ]
            g_impl_result = PhaseResult(
                phase_name=g_impl_data["phase_name"],
                success=g_impl_data["success"],
                attempts=attempts,
                final_code=g_impl_data["final_code"],
                total_attempts=g_impl_data["total_attempts"],
            )

        return cls(
            task_id=data["task_id"],
            trial_number=data["trial_number"],
            workflow_success=data["workflow_success"],
            g_test_result=g_test_result,
            g_impl_result=g_impl_result,
            total_attempts=data["total_attempts"],
            execution_time_seconds=data["execution_time_seconds"],
            first_attempt_test_success=data["first_attempt_test_success"],
            validated_success=data["validated_success"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class BenchmarkSummary:
    """Summary statistics for a complete benchmark run."""
    model_name: str
    num_tasks: int
    num_trials_per_task: int
    total_trials: int

    # Success rates
    workflow_success_rate: float  # % of trials where both phases succeeded
    g_test_success_rate: float    # % of trials where g_test succeeded
    g_impl_success_rate: float    # % of trials where g_impl succeeded

    # Attempt statistics
    avg_attempts_per_trial: float
    avg_g_test_attempts: float
    avg_g_impl_attempts: float

    # Quality metrics for training data
    first_attempt_test_successes: int
    validated_successes: int

    # Execution time
    total_execution_time: float
    avg_time_per_trial: float

    # Per-task breakdown
    task_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
