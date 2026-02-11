"""Multi-language test generator.

Subclasses :class:`TestGenerator` from swe_bench_common to add
language-specific framework instructions to the prompt.  PydanticAI
handles structured output parsing via the ``GeneratedTest`` model.
"""

import logging
from typing import Any

from examples.swe_bench_common.generators import TestGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from ..language import LanguageConfig

logger = logging.getLogger("swe_bench_pro.generators")

_FRAMEWORK_INSTRUCTIONS: dict[str, str] = {
    "python": (
        "Use pytest style (``test_`` functions or ``Test`` classes).\n"
        "The test should FAIL on the current buggy code and PASS after the fix.\n"
        "Import only from the project's own modules."
    ),
    "go": (
        "Use the Go ``testing`` package (``func TestXxx(t *testing.T)``).\n"
        "The test should FAIL on the current buggy code and PASS after the fix.\n"
        "Import only packages available in the project."
    ),
    "javascript": (
        "Use Jest or Mocha style (``describe``/``it``/``test`` blocks).\n"
        "The test should FAIL on the current buggy code and PASS after the fix.\n"
        "Import or require only modules available in the project."
    ),
    "typescript": (
        "Use Jest or Mocha style (``describe``/``it``/``test`` blocks).\n"
        "The test should FAIL on the current buggy code and PASS after the fix.\n"
        "Import only modules available in the project."
    ),
}


class MultiLangTestGenerator(TestGenerator):
    """Language-aware test generator.

    Overrides ``_build_prompt`` to add language-specific test framework
    instructions to the constraints.
    """

    def __init__(self, language_config: LanguageConfig, **kwargs: Any):
        super().__init__(**kwargs)
        self._lang = language_config

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        parts: list[str] = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Analysis dependency
        analysis = self._get_analysis(context)
        if analysis:
            parts.append("\n\n## Bug Analysis")
            parts.append(f"Bug type: {analysis.bug_type.value}")
            parts.append(f"Root cause: {analysis.root_cause_hypothesis}")
            parts.append(
                f"Affected components: {', '.join(analysis.affected_components)}"
            )
            parts.append(f"Files: {', '.join(analysis.files)}")
            parts.append(f"Fix approach: {analysis.fix_approach}")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        # Language-specific framework instructions
        if self._lang.name in _FRAMEWORK_INSTRUCTIONS:
            parts.append(
                f"\n\n## Test Framework\n{_FRAMEWORK_INSTRUCTIONS[self._lang.name]}"
            )

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n"
                f"{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)
