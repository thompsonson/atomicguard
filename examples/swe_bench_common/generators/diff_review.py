"""DiffReviewGenerator: LLM-as-reviewer that critiques generated patches.

Reviews the full context (analysis + test + patch) and produces a structured
critique with verdict, issues, and a backtrack_target heuristic for the
BacktrackOrchestrator (Arm 18).

Used by: ap_diff_review in Arms 17, 18
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator

from examples.swe_bench_common.models import DiffReview


class DiffReviewGenerator(PydanticAIGenerator[DiffReview]):
    """Generator that produces structured code review of a patch.

    Reads analysis, test code, and patch from dependency artifacts.
    Outputs a verdict (approve/revise/backtrack) with a backtrack_target
    field that the BacktrackOrchestrator uses as a search heuristic.

    Context comes from prompt templates via {ap_analysis}, {ap_gen_test},
    and {ap_gen_patch} placeholders.
    """

    output_type = DiffReview

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
