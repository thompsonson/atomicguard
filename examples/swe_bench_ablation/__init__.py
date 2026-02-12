"""SWE-bench Ablation Study Example.

This example demonstrates using AtomicGuard for SWE-bench problem solving
with different workflow variants to measure component contribution.

This module re-exports models from swe_bench_common for backward compatibility.
New code should import from examples.swe_bench_common.models directly.
"""

# Re-export models from common for backward compatibility
from examples.swe_bench_common.models import (
    Analysis,
    BugCharacterization,
    BugType,
    ContextSummary,
    FixApproach,
    FunctionLocation,
    GeneratedTest,
    Hypothesis,
    ImpactAnalysis,
    IssueParsed,
    Localization,
    Patch,
    ProjectStructure,
    RootCause,
    SearchReplaceEdit,
    StackFrame,
    TestLocalization,
)

__all__ = [
    "Analysis",
    "BugCharacterization",
    "BugType",
    "ContextSummary",
    "FixApproach",
    "FunctionLocation",
    "GeneratedTest",
    "Hypothesis",
    "ImpactAnalysis",
    "IssueParsed",
    "Localization",
    "Patch",
    "ProjectStructure",
    "RootCause",
    "SearchReplaceEdit",
    "StackFrame",
    "TestLocalization",
]
