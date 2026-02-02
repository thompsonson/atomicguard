"""Pydantic models for structured LLM output in SWE-bench ablation study.

These are OUTPUT schemas for LLM extraction - NOT core infrastructure.
Core types (Artifact, Context, GuardResult, etc.) are imported from atomicguard.domain.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# Bug Analysis Output (Experiment 7.2)
# =============================================================================


class BugType(str, Enum):
    """Classification of bug type."""

    LOGIC = "logic"
    TYPE_ERROR = "type_error"
    OFF_BY_ONE = "off_by_one"
    NULL_REFERENCE = "null_reference"
    API_MISUSE = "api_misuse"
    MISSING_CHECK = "missing_check"
    WRONG_RETURN = "wrong_return"
    CONCURRENCY = "concurrency"
    PERFORMANCE = "performance"
    OTHER = "other"


class Analysis(BaseModel):
    """Structured output from bug analysis (Experiment 7.2 S1/S1-TDD arms)."""

    bug_type: BugType
    root_cause_hypothesis: str
    affected_components: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list, min_length=1)
    fix_approach: str
    confidence: Literal["low", "medium", "high"] = "medium"


class GeneratedTest(BaseModel):
    """Structured output from test generation (Experiment 7.2 S1-TDD arm)."""

    test_code: str
    target_behavior: str
    expected_to_fail: bool = True


# =============================================================================
# Issue Parsing Output
# =============================================================================


class IssueParsed(BaseModel):
    """Structured output from issue parsing."""

    expected_behavior: str | None = None
    actual_behavior: str | None = None
    reproduction_steps: str | None = None
    error_messages: str | None = None
    affected_components: list[str] = Field(default_factory=list)


# =============================================================================
# Bug Characterization Output
# =============================================================================


class StackFrame(BaseModel):
    """Single frame in a stack trace."""

    file: str
    line: int
    function: str


class BugCharacterization(BaseModel):
    """Structured output from bug characterization (test execution)."""

    test_command: str
    test_passed: bool
    error_type: str | None = None
    error_message: str | None = None
    stack_trace: list[StackFrame] = Field(default_factory=list)
    relevant_files: list[str] = Field(default_factory=list)
    relevant_functions: list[str] = Field(default_factory=list)


# =============================================================================
# Hypothesis Formation Output
# =============================================================================


class Hypothesis(BaseModel):
    """Structured output from hypothesis formation."""

    root_cause: str
    fix_approach: str
    files: list[str] = Field(default_factory=list)
    likely_functions: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


# =============================================================================
# Localization Output
# =============================================================================


class FunctionLocation(BaseModel):
    """A function location reference."""

    name: str
    file: str
    line: int | None = None


class Localization(BaseModel):
    """Structured output from localization."""

    files: list[str]
    functions: list[FunctionLocation] = Field(default_factory=list)
    reasoning: str | None = None


# =============================================================================
# Patch Generation Output
# =============================================================================


class SearchReplaceEdit(BaseModel):
    """A single search-replace edit.

    The LLM specifies exact code blocks to find and replace.
    This is converted to unified diff format programmatically.
    """

    file: str  # File path relative to repo root
    search: str  # Exact code to find (must match exactly)
    replace: str  # Code to replace with


class Patch(BaseModel):
    """Patch output from patch generation.

    Uses search-replace format which is more natural for LLMs
    than unified diff format.
    """

    edits: list[SearchReplaceEdit]
    reasoning: str | None = None


# =============================================================================
# Test Results Output
# =============================================================================


class TestResults(BaseModel):
    """Structured output from test execution."""

    fail_to_pass_passed: bool = False
    pass_to_pass_passed: bool = False
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""


# =============================================================================
# Analysis Models (for results tracking)
# =============================================================================


class ActionPairMetrics(BaseModel):
    """Metrics for a single action pair execution."""

    action_pair_id: str
    attempts: int
    passed: bool
    tokens_used: int = 0
    time_seconds: float = 0.0


class ProblemResult(BaseModel):
    """Result of running a workflow on a single problem."""

    problem_id: str
    workflow_variant: str
    success: bool
    action_pair_metrics: list[ActionPairMetrics] = Field(default_factory=list)
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    error: str | None = None


class AblationResults(BaseModel):
    """Aggregated results across all problems for a variant."""

    variant: str
    problems_attempted: int
    problems_passed: int
    pass_rate: float
    avg_tokens_per_problem: float = 0.0
    avg_time_per_problem_seconds: float = 0.0
    problem_results: list[ProblemResult] = Field(default_factory=list)
