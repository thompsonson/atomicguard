"""LocalizationGenerator: Identifies files and functions to modify.

Uses PydanticAI structured output to analyze the problem statement
and identify which files and functions need modification to fix the bug.
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator

from examples.swe_bench_common.models import Localization


class LocalizationGenerator(PydanticAIGenerator[Localization]):
    """Generator that identifies files and functions to modify.

    Uses PydanticAI structured output to analyze the problem statement
    and identify the most likely locations that need modification.

    Context comes from prompt templates via {specification} placeholder.
    """

    output_type = Localization

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
