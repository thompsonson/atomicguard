"""Shared data models for SWE-bench experiment runners."""

from dataclasses import dataclass, field


@dataclass
class ArmResult:
    """Result of running one arm on one instance."""

    instance_id: str
    arm: str
    patch_content: str = ""
    total_tokens: int = 0
    per_step_tokens: dict[str, int] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    init_time_seconds: float = 0.0  # Repo clone + checkout time
    workflow_time_seconds: float = 0.0  # Action pair execution time
    error: str | None = None
    resolved: bool | None = None  # Evaluation result (None = not evaluated)
    failed_step: str | None = None  # Which action pair failed (None = success)
    failed_guard: str | None = None  # Which guard in the step failed (for composite)
    retry_count: int = 0  # How many retries before failure
