"""Multi-language patch generator.

Subclasses :class:`PatchGenerator` from the ablation example to replace
Python-specific prompt text with language-appropriate alternatives.
"""

from typing import Any

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate
from examples.swe_bench_ablation.generators import PatchGenerator

from ..language import LanguageConfig


class MultiLangPatchGenerator(PatchGenerator):
    """Language-aware patch generator.

    Overrides ``_build_prompt`` so that code-fence language tags, valid-code
    labels, and file-content annotations match the target language instead of
    being hard-coded to Python.
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
        repo_root: str | None,
    ) -> str:
        """Build a language-aware prompt for patch generation.

        The structure mirrors the parent class but substitutes:
        * ``python`` code fences → ``self._lang.code_block_tag``
        * ``VALID PYTHON`` → ``self._lang.valid_code_label``
        * File extension filtering for context snippets
        """
        parts: list[str] = []

        # Task from template
        if template:
            parts.append(template.task)

        # Problem statement
        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Analysis dependency
        analysis = self._get_analysis(context)
        if analysis:
            parts.append("\n\n## Bug Analysis")
            parts.append(f"Bug type: {analysis.bug_type.value}")
            parts.append(f"Root cause: {analysis.root_cause_hypothesis}")
            parts.append(f"Likely files: {', '.join(analysis.likely_files)}")
            parts.append(f"Fix approach: {analysis.fix_approach}")

            if self._include_file_content and repo_root:
                parts.append("\n\n## Current File Content")
                tag = self._lang.code_block_tag
                for file_path in analysis.likely_files[:3]:
                    content = self._read_file(repo_root, file_path)
                    if content:
                        parts.append(f"\n### {file_path}\n```{tag}\n{content}\n```")

        # Localization dependency
        localization = self._get_localization(context)
        if localization and not analysis:
            parts.append("\n\n## Files to Modify")
            parts.append(f"Files: {', '.join(localization.files)}")
            if localization.functions:
                funcs = [f"{f.name} in {f.file}" for f in localization.functions]
                parts.append(f"Functions: {', '.join(funcs)}")
            if localization.reasoning:
                parts.append(f"Reasoning: {localization.reasoning}")

            if self._include_file_content and repo_root:
                parts.append("\n\n## Current File Content")
                tag = self._lang.code_block_tag
                for file_path in localization.files[:3]:
                    content = self._read_file(repo_root, file_path)
                    if content:
                        parts.append(f"\n### {file_path}\n```{tag}\n{content}\n```")

        # Test code dependency
        test_code = self._get_test_code(context)
        if test_code:
            tag = self._lang.code_block_tag
            parts.append(f"\n\n## Failing Test\n```{tag}\n{test_code}\n```")

        # Constraints
        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        # Output format – language-adapted
        lang_label = self._lang.valid_code_label
        parts.append(
            f"""

## Output Format
Return a JSON object with search-replace edits:
```json
{{
  "edits": [
    {{
      "file": "path/to/file",
      "search": "exact code to find\\nincluding multiple lines",
      "replace": "new code to replace with\\nincluding multiple lines"
    }}
  ],
  "reasoning": "Brief explanation of the fix"
}}
```

CRITICAL REQUIREMENTS:
1. EXACT MATCH: The 'search' string must match the file content EXACTLY (including whitespace/indentation)
2. {lang_label}: The 'replace' code must be syntactically valid
3. MINIMAL CHANGE: Only change what's necessary to fix the bug
4. PRESERVE STYLE: Match existing code style

TIPS:
- Include enough surrounding context in 'search' to ensure uniqueness
- Copy the exact indentation from the file
- Include 1-2 lines before/after your target if the match is ambiguous
"""
        )

        # Feedback on retry
        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n"
                f"{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)
