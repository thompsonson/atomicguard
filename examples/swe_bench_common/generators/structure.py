"""StructureGenerator: Analyzes project structure, imports, and dependencies.

Provides high-level understanding of the codebase layout to inform
downstream steps like localization and test generation.
"""

import logging
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import ProjectStructure

logger = logging.getLogger("swe_bench_ablation.generators")


class StructureGenerator(PydanticAIGenerator[ProjectStructure]):
    """Generator that analyzes project structure.

    Outputs a ProjectStructure JSON identifying top-level modules,
    test framework, test directories, import conventions, and
    key dependencies.
    """

    output_type = ProjectStructure

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.1)
        super().__init__(**kwargs)

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for structure analysis."""
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
