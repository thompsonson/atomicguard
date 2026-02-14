"""Shared data models for SWE-bench experiment runners.

Contains:
1. ArmResult dataclass - Result tracking for experiment runs
2. Pydantic models - Structured LLM output schemas for generators

These are OUTPUT schemas for LLM extraction - NOT core infrastructure.
Core types (Artifact, Context, GuardResult, etc.) are imported from atomicguard.domain.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# =============================================================================
# Experiment Result Tracking
# =============================================================================


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
# Decomposed Workflow Outputs (Arm 07)
# =============================================================================


class ProjectStructure(BaseModel):
    """Structured output from project structure analysis (ap_structure).

    Captures high-level understanding of project layout, dependencies,
    and conventions to inform downstream steps.
    """

    root_modules: list[str] = Field(
        default_factory=list,
        description="Top-level Python modules/packages in the project",
    )
    test_framework: str = Field(
        default="pytest",
        description="Test framework used (pytest, unittest, nose, etc.)",
    )
    test_directories: list[str] = Field(
        default_factory=list,
        description="Directories containing test files",
    )
    import_conventions: str | None = Field(
        default=None,
        description="Import style used (absolute, relative, etc.)",
    )
    key_dependencies: list[str] = Field(
        default_factory=list,
        description="Key third-party dependencies",
    )
    reasoning: str | None = None


class RootCause(BaseModel):
    """Structured output from root cause analysis (ap_root_cause).

    Deeper analysis of the bug's root cause, building on classification.
    """

    cause_type: str = Field(
        description="Specific type of root cause (e.g., 'incorrect_condition', 'missing_null_check')"
    )
    cause_description: str = Field(
        description="Detailed explanation of why the bug occurs"
    )
    triggering_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that trigger the bug",
    )
    affected_code_paths: list[str] = Field(
        default_factory=list,
        description="Code paths affected by the bug",
    )
    confidence: Literal["low", "medium", "high"] = "medium"


class ContextSummary(BaseModel):
    """Structured output from context reading (ap_context_read).

    Summarizes the relevant code context around the bug location.
    """

    file_path: str = Field(description="Primary file containing the bug")
    relevant_functions: list[str] = Field(
        default_factory=list,
        description="Functions relevant to the bug",
    )
    relevant_classes: list[str] = Field(
        default_factory=list,
        description="Classes relevant to the bug",
    )
    imports_used: list[str] = Field(
        default_factory=list,
        description="Imports used by the buggy code",
    )
    code_snippet: str = Field(
        description="The relevant code snippet containing or near the bug"
    )
    summary: str = Field(description="Summary of what the code is doing")


class TestLocalization(BaseModel):
    """Structured output from test localization (ap_localise_tests).

    Comprehensive test infrastructure analysis to guide test generation.
    """

    # Test file locations (may be empty when proposed_test_file is used)
    test_files: list[str] = Field(
        default_factory=list,
        description="Existing test files related to the buggy code",
    )

    # Proposed new test file (for TDD when no existing test file exists)
    proposed_test_file: str | None = Field(
        default=None,
        description=(
            "Path for a new test file to create when no existing test file "
            "covers the buggy code. An ancestor directory must exist in the repo."
        ),
    )

    @model_validator(mode="after")
    def check_test_files_or_proposed(self) -> "TestLocalization":
        """Ensure at least one of test_files or proposed_test_file is provided."""
        if not self.test_files and not self.proposed_test_file:
            msg = "At least one of test_files or proposed_test_file must be provided"
            raise ValueError(msg)
        return self

    # Test discovery patterns
    test_patterns: list[str] = Field(
        default_factory=list,
        description="File naming patterns (e.g., 'test_*.py', '*_test.py')",
    )

    # Test framework info
    test_library: str = Field(
        default="pytest",
        description="Test framework: pytest, unittest, nose, doctest",
    )
    test_plugins: list[str] = Field(
        default_factory=list,
        description="Test plugins used (pytest-qt, pytest-asyncio, etc.)",
    )

    # Test infrastructure
    test_fixtures: list[str] = Field(
        default_factory=list,
        description="Available fixtures or setup functions",
    )
    conftest_files: list[str] = Field(
        default_factory=list,
        description="Relevant conftest.py files in scope",
    )

    # Test style
    test_style: Literal["function-based", "class-based", "bdd", "mixed"] = Field(
        default="function-based",
        description="Testing style used in this area of the codebase",
    )

    # Invocation
    test_invocation: str = Field(
        description="Example command to run the relevant tests",
    )

    # Reasoning
    reasoning: str | None = None


class FixApproach(BaseModel):
    """Structured output from fix approach design (ap_fix_approach).

    Detailed strategy for how to fix the bug.
    """

    approach_summary: str = Field(description="One-line summary of the fix approach")
    steps: list[str] = Field(
        default_factory=list,
        description="Ordered steps to implement the fix",
    )
    files_to_modify: list[str] = Field(
        default_factory=list,
        description="Files that need modification",
    )
    functions_to_modify: list[str] = Field(
        default_factory=list,
        description="Functions that need modification",
    )
    edge_cases: list[str] = Field(
        default_factory=list,
        description="Edge cases the fix should handle",
    )
    reasoning: str


class FileEdit(BaseModel):
    """A planned edit to a single file (part of EditPlan)."""

    file: str = Field(description="File path relative to repo root")
    change_description: str = Field(description="What needs to change in this file")
    functions_to_modify: list[str] = Field(
        default_factory=list,
        description="Functions that will be modified in this file",
    )


class EditPlan(BaseModel):
    """Structured output from edit planning (ap_edit_plan).

    Bridges fix_approach and gen_patch by specifying exact files and
    changes needed, validated against the repository before patch generation.
    """

    files_to_edit: list[FileEdit] = Field(
        min_length=1,
        description="Files to edit with descriptions of changes",
    )
    import_changes: list[str] = Field(
        default_factory=list,
        description="Import additions/removals needed",
    )
    rationale: str = Field(description="Why these specific edits fix the bug")


class ImpactAnalysis(BaseModel):
    """Structured output from impact analysis (ap_impact_analysis).

    Analyzes the impact of the proposed fix on other code and tests.
    """

    affected_tests: list[str] = Field(
        default_factory=list,
        description="Existing tests that may be affected by the fix",
    )
    affected_functions: list[str] = Field(
        default_factory=list,
        description="Other functions that may be affected by the fix",
    )
    potential_regressions: list[str] = Field(
        default_factory=list,
        description="Potential regressions to watch for",
    )
    api_changes: list[str] = Field(
        default_factory=list,
        description="Any API changes introduced by the fix",
    )
    risk_level: Literal["low", "medium", "high"] = "low"
    reasoning: str


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
# Diff Review Output (Arms 17-18)
# =============================================================================


class DiffReviewVerdict(str, Enum):
    """Verdict from the diff review action pair."""

    APPROVE = "approve"
    REVISE = "revise"
    BACKTRACK = "backtrack"


class DiffReviewIssue(BaseModel):
    """A single issue identified in the diff review."""

    severity: Literal["critical", "minor"] = "critical"
    description: str
    location: str | None = None


class DiffReview(BaseModel):
    """Structured output from diff review (ap_diff_review).

    The reviewer reads analysis + test + patch and produces a structured
    critique. The backtrack_target field is the heuristic signal consumed
    by the BacktrackOrchestrator in Arm 18.
    """

    verdict: DiffReviewVerdict
    issues: list[DiffReviewIssue] = Field(default_factory=list)
    backtrack_target: str | None = Field(
        default=None,
        description=(
            "Which action pair to backtrack to when verdict is 'backtrack'. "
            "One of: ap_gen_patch, ap_gen_test, ap_analysis, or null."
        ),
    )
    reasoning: str


# =============================================================================
# Problem Classification Output (Arms 20-21)
# =============================================================================


class ProblemCategory(str, Enum):
    """Classification of problem complexity."""

    TRIVIAL_FIX = "trivial_fix"
    SINGLE_FILE_BUG = "single_file_bug"
    MULTI_FILE_BUG = "multi_file_bug"
    API_CHANGE = "api_change"
    REFACTOR = "refactor"


class ProblemClassification(BaseModel):
    """Structured output from problem classification (ap_classify_problem)."""

    category: ProblemCategory
    estimated_complexity: int = Field(ge=1, le=5)
    reasoning: str


# =============================================================================
# Generated Workflow Output (Arms 20-21)
# =============================================================================


class GeneratedActionPairSpec(BaseModel):
    """Specification for a single action pair in a generated workflow."""

    generator: str
    guard: str
    guard_config: dict[str, Any] = Field(default_factory=dict)
    requires: list[str] = Field(default_factory=list)
    description: str = ""
    prompt_override: str | None = None


class BacktrackConfig(BaseModel):
    """Per-step backtracking budget for generated workflows (Arm 21)."""

    backtrack_budget: dict[str, int] = Field(
        default_factory=dict,
        description="Map of action_pair_id -> backtrack budget",
    )
    include_diff_review: bool = False


class GeneratedWorkflow(BaseModel):
    """Structured output from workflow generation (ap_generate_workflow).

    Uses the same schema as static workflow configs, enabling direct
    execution by the existing workflow infrastructure.
    """

    name: str
    description: str = ""
    rmax: int = Field(default=4, ge=1, le=10)
    action_pairs: dict[str, GeneratedActionPairSpec]
    backtrack_config: BacktrackConfig | None = None
    reasoning: str | None = None


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
