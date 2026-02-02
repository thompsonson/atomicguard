"""LocalizationGenerator: Identifies files and functions to modify.

Uses PydanticAI structured output to analyze the problem statement
and identify which files and functions need modification to fix the bug.
"""

import logging
from typing import Any

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate
from examples.base.generators import PydanticAIGenerator

from ..models import Localization

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

        # Output format instruction
        parts.append(
            """

## Output Format
Return a JSON object with this structure:
```json
{
  "files": ["path/to/file1.py", "path/to/file2.py"],
  "functions": [
    {"name": "function_name", "file": "path/to/file.py", "line": null}
  ],
  "reasoning": "Brief explanation of why these locations were identified"
}
```

IMPORTANT:
- Maximum 5 files
- File paths should be relative to repository root
- Function line numbers can be null if unknown
"""
        )

        # Add feedback if retry
        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)
