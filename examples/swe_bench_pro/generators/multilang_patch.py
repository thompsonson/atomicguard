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
    ) -> str:
        """Build a language-aware prompt for patch generation.

        The structure mirrors the parent class but substitutes:
        * ``python`` code fences → ``self._lang.code_block_tag``
        * File extension filtering for context snippets
        """
        repo_root = self._resolve_repo_root(context)
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
            parts.append(f"Files: {', '.join(analysis.files)}")
            parts.append(f"Fix approach: {analysis.fix_approach}")

            if self._include_file_content and repo_root:
                parts.append("\n\n## Current File Content")
                tag = self._lang.code_block_tag
                for file_path in analysis.files[:3]:
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

        # Singleshot fallback: include content of files referenced in the problem
        if not analysis and not localization and self._include_file_content and repo_root:
            repo_files = self._list_repo_files(
                repo_root, extensions=self._lang.file_extensions
            )
            referenced = [f for f in repo_files if f in context.specification]
            if referenced:
                parts.append("\n\n## Current File Content")
                tag = self._lang.code_block_tag
                for file_path in referenced[:5]:
                    content = self._read_file(repo_root, file_path)
                    if content:
                        parts.append(f"\n### {file_path}\n```{tag}\n{content}\n```")

        # Test code dependency
        test_code = self._get_test_code(context)
        if test_code:
            tag = self._lang.code_block_tag
            parts.append(
                f"\n\n## Failing Test (for guidance only — do NOT patch this)\n"
                f"The following test demonstrates the expected behavior. "
                f"Your patch should fix the SOURCE files so this test would pass. "
                f"Do NOT create or modify test files.\n"
                f"```{tag}\n{test_code}\n```"
            )

        # Constraints
        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        # Feedback on retry
        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n"
                f"{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)
