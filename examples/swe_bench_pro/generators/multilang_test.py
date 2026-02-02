"""Multi-language test generator.

Subclasses :class:`TestGenerator` from the ablation example to replace
pytest-specific prompt text with language-appropriate alternatives.
"""

import logging
import re
from typing import Any

logger = logging.getLogger("swe_bench_pro.generators")

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate
from examples.swe_bench_ablation.generators import TestGenerator

from ..language import LanguageConfig

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

    Overrides ``_build_prompt`` and ``_extract_code`` so that test-framework
    references and code-fence tags match the target language.
    """

    def __init__(self, language_config: LanguageConfig, **kwargs: Any):
        super().__init__(**kwargs)
        self._lang = language_config

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

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

        # Output format – language-adapted
        tag = self._lang.code_block_tag
        if self._lang.name not in _FRAMEWORK_INSTRUCTIONS:
            raise ValueError(
                f"No test framework instructions for language {self._lang.name!r}. "
                f"Supported: {', '.join(sorted(_FRAMEWORK_INSTRUCTIONS))}"
            )
        fw_instructions = _FRAMEWORK_INSTRUCTIONS[self._lang.name]

        parts.append(
            f"""

## Output Format
Return test code in a markdown code block:
```{tag}
// your test code here
```

REQUIREMENTS:
{fw_instructions}
- Be specific about the expected behaviour.
"""
        )

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n"
                f"{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Code extraction
    # ------------------------------------------------------------------

    def _extract_code(self, content: str) -> str:
        """Extract code from an LLM response, preferring the target language tag."""
        tag = self._lang.code_block_tag

        # Try language-specific code block first
        match = re.search(rf"```{tag}\s*([\s\S]*?)```", content)
        if match:
            return match.group(1).strip()

        # Fall back to generic code block
        match = re.search(r"```\s*([\s\S]*?)```", content)
        if match:
            return match.group(1).strip()

        # If content looks like test code, return as-is
        pattern = self._lang.test_function_pattern
        if re.search(pattern, content):
            logger.warning(
                "No code fence found in LLM response for %s; "
                "using raw content that matched test pattern",
                self._lang.name,
            )
            return content.strip()

        logger.warning(
            "Could not extract %s test code from LLM response (%d chars). "
            "No code fences or test patterns found.",
            self._lang.name,
            len(content),
        )
        return (
            f"// Could not extract test code from response\n"
            f"// Raw: {content[:500]}"
        )
