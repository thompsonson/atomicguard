"""AnalysisGenerator: Analyzes bug type, root cause, and fix approach.

Uses PydanticAI structured output to classify the bug and produce a
structured analysis that downstream generators (patch or test) consume
via dependency artifacts.
"""

import logging
from typing import Any

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate
from examples.base.generators import PydanticAIGenerator

from ..models import Analysis

logger = logging.getLogger("swe_bench_ablation.generators")


class AnalysisGenerator(PydanticAIGenerator[Analysis]):
    """Generator that produces structured bug analysis.

    Outputs an Analysis JSON classifying the bug type, identifying
    root cause, affected components, likely files, and fix approach.
    """

    output_type = Analysis

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for analysis."""
        parts = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        parts.append(
            """

## Output Format
Return a JSON object with this structure:
```json
{
  "bug_type": "logic|type_error|off_by_one|null_reference|api_misuse|missing_check|wrong_return|concurrency|performance|other",
  "root_cause_hypothesis": "Detailed explanation of the likely root cause",
  "affected_components": ["module.submodule", "class.method"],
  "files": ["path/to/file1.py", "path/to/file2.py"],
  "fix_approach": "Detailed description of how to fix the bug",
  "confidence": "low|medium|high"
}
```

IMPORTANT:
- Identify at least one likely file
- Provide a specific, actionable root cause hypothesis
- The fix approach should be concrete enough to guide patch generation
"""
        )

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)
