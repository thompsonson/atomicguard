"""EditPlanGenerator: Plans exact file edits before patch generation.

Bridges the gap between fix_approach (strategy) and gen_patch (code)
by specifying which files and functions to modify with descriptions.
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator

from examples.swe_bench_common.models import EditPlan


class EditPlanGenerator(PydanticAIGenerator[EditPlan]):
    """Generator that plans exact file edits.

    Takes fix_approach and context_read as inputs and produces a
    validated edit plan that gen_patch can follow precisely.

    Context comes from prompt templates via {ap_fix_approach}
    and {ap_context_read} placeholders.
    """

    output_type = EditPlan

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
