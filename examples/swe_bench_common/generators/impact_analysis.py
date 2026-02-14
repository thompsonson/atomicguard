"""ImpactAnalysisGenerator: Analyzes the impact of a proposed fix.

Evaluates how the fix approach will affect other code and tests,
identifying potential regressions and API changes.
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator
from examples.swe_bench_common.models import ImpactAnalysis


class ImpactAnalysisGenerator(PydanticAIGenerator[ImpactAnalysis]):
    """Generator that analyzes the impact of a proposed fix.

    Uses the fix approach to identify affected tests, functions,
    potential regressions, and API changes.

    Context comes from prompt templates via {ap_fix_approach} placeholder.
    """

    output_type = ImpactAnalysis

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
