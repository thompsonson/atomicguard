"""ClassificationGenerator: Classifies problem instances by complexity.

Part of the meta-level pipeline in Arms 20-21. Classifies the problem
into a category (trivial_fix, single_file_bug, multi_file_bug, api_change,
refactor) that determines which workflow template to generate.

Used by: ap_classify_problem in Arms 20, 21
"""

import logging
from typing import Any

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate
from examples.base.generators import PydanticAIGenerator

from ..models import ProblemClassification

logger = logging.getLogger("swe_bench_ablation.generators")


class ClassificationGenerator(PydanticAIGenerator[ProblemClassification]):
    """Generator that classifies problem instances by type and complexity.

    Reads the problem statement and repository file listing. Outputs a
    category and complexity estimate used by ap_generate_workflow to
    select the appropriate pipeline.
    """

    output_type = ProblemClassification

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.1)
        super().__init__(**kwargs)

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for problem classification."""
        parts = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)
