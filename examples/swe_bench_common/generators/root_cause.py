"""RootCauseGenerator: Identifies the root cause of a bug based on classification.

Builds on the classification step to provide deeper analysis of why
the bug occurs and what conditions trigger it.
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator

from examples.swe_bench_common.models import RootCause


class RootCauseGenerator(PydanticAIGenerator[RootCause]):
    """Generator that identifies the root cause of a bug.

    Uses the classification from ap_classify to produce a detailed
    root cause analysis including triggering conditions and
    affected code paths.

    Context comes from prompt templates via {ap_classify} placeholder.
    """

    output_type = RootCause

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
