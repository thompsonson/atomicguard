"""FixApproachGenerator: Designs the fix strategy.

Synthesizes information from root cause analysis, context reading,
and localization to design a concrete fix approach.
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator

from examples.swe_bench_common.models import FixApproach


class FixApproachGenerator(PydanticAIGenerator[FixApproach]):
    """Generator that designs the fix strategy.

    Uses root cause, context, and localization to produce a detailed
    fix approach with ordered steps and edge case handling.

    Context comes from prompt templates via {ap_root_cause},
    {ap_context_read}, and {ap_localise_issue} placeholders.
    """

    output_type = FixApproach

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
