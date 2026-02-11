"""LocalizationGenerator: Identifies files and functions to modify.

Uses PydanticAI structured output to analyze the problem statement
and identify which files and functions need modification to fix the bug.
"""

import logging
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import Localization

logger = logging.getLogger("swe_bench_ablation.generators")


class LocalizationGenerator(PydanticAIGenerator[Localization]):
    """Generator that identifies files and functions to modify.

    Uses PydanticAI structured output to analyze the problem statement
    and identify the most likely locations that need modification.
    """

    output_type = Localization

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for localization."""
        parts = []

        # Task from template
        if template:
            parts.append(template.task)

        # Problem statement
        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Constraints from template
        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        # Add feedback if retry
        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)
