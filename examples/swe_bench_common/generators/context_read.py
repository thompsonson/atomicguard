"""ContextReadGenerator: Reads and summarizes relevant code context.

Builds on localization to read the actual code files and provide
a summary of the relevant context around the bug location.
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator

from examples.swe_bench_common.models import ContextSummary


class ContextReadGenerator(PydanticAIGenerator[ContextSummary]):
    """Generator that reads and summarizes code context.

    Uses the localization from ap_localise_issue to read the actual
    file contents and provide a contextual summary.

    Context comes from prompt templates via {ap_localise_issue} placeholder.
    """

    output_type = ContextSummary

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
