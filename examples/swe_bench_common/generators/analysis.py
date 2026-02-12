"""AnalysisGenerator: Analyzes bug type, root cause, and fix approach.

Uses PydanticAI structured output to classify the bug and produce a
structured analysis that downstream generators (patch or test) consume
via dependency artifacts.
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator

from examples.swe_bench_common.models import Analysis


class AnalysisGenerator(PydanticAIGenerator[Analysis]):
    """Generator that produces structured bug analysis.

    Outputs an Analysis JSON classifying the bug type, identifying
    root cause, affected components, likely files, and fix approach.

    Context comes from prompt templates via {specification} placeholder.
    """

    output_type = Analysis

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
