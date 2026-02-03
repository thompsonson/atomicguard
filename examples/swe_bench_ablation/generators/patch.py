"""PatchGenerator: Generates patches using search-replace format.

Uses PydanticAI structured output to generate code changes as
search-replace blocks, which are then converted to unified diff
format programmatically. This approach is more reliable than asking
LLMs to generate diff format directly.
"""

import difflib
import json
import logging
from pathlib import Path
from typing import Any

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate
from examples.base.generators import PydanticAIGenerator

from ..models import Analysis, Localization, Patch, SearchReplaceEdit

logger = logging.getLogger("swe_bench_ablation.generators")


class PatchGenerator(PydanticAIGenerator[Patch]):
    """Generator that produces patches via search-replace blocks.

    The LLM specifies exact code blocks to find and replace.
    This is converted to unified diff format programmatically using difflib.
    """

    output_type = Patch

    def __init__(
        self,
        *,
        include_file_content: bool = True,
        max_context_lines: int = 100,
        repo_root: str | None = None,
        code_block_tag: str = "python",
        **kwargs: Any,
    ):
        kwargs.setdefault("temperature", 0.3)
        super().__init__(**kwargs)
        self._include_file_content = include_file_content
        self._max_context_lines = max_context_lines
        self._repo_root = repo_root
        self._code_block_tag = code_block_tag

    def _resolve_repo_root(self, context: Context) -> str | None:
        """Resolve repo_root from context metadata or constructor fallback."""
        repo_root = None
        if hasattr(context.ambient.repository, "metadata"):
            repo_root = context.ambient.repository.metadata.get("repo_root")
        if not repo_root:
            repo_root = self._repo_root
        return repo_root

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for patch generation."""
        repo_root = self._resolve_repo_root(context)
        parts = []

        # Task from template
        if template:
            parts.append(template.task)

        # Problem statement
        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Get analysis from dependencies (S1-direct and S1-TDD arms)
        analysis = self._get_analysis(context)
        if analysis:
            parts.append("\n\n## Bug Analysis")
            parts.append(f"Bug type: {analysis.bug_type.value}")
            parts.append(f"Root cause: {analysis.root_cause_hypothesis}")
            parts.append(f"Files: {', '.join(analysis.files)}")
            parts.append(f"Fix approach: {analysis.fix_approach}")

            # Include file content from analysis
            if self._include_file_content and repo_root:
                parts.append("\n\n## Current File Content")
                tag = self._code_block_tag
                for file_path in analysis.files[:3]:
                    content = self._read_file(repo_root, file_path)
                    if content:
                        parts.append(f"\n### {file_path}\n```{tag}\n{content}\n```")

        # Get localization from dependencies (baseline arm)
        localization = self._get_localization(context)
        if localization and not analysis:
            parts.append("\n\n## Files to Modify")
            parts.append(f"Files: {', '.join(localization.files)}")
            if localization.functions:
                funcs = [f"{f.name} in {f.file}" for f in localization.functions]
                parts.append(f"Functions: {', '.join(funcs)}")
            if localization.reasoning:
                parts.append(f"Reasoning: {localization.reasoning}")

            # Include file content
            if self._include_file_content and repo_root:
                parts.append("\n\n## Current File Content")
                tag = self._code_block_tag
                for file_path in localization.files[:3]:  # Limit to 3 files
                    content = self._read_file(repo_root, file_path)
                    if content:
                        parts.append(f"\n### {file_path}\n```{tag}\n{content}\n```")

        # Singleshot fallback: include content of files referenced in the problem
        if not analysis and not localization and self._include_file_content and repo_root:
            repo_files = self._list_repo_files(repo_root)
            referenced = [f for f in repo_files if f in context.specification]
            if referenced:
                parts.append("\n\n## Current File Content")
                tag = self._code_block_tag
                for file_path in referenced[:5]:
                    content = self._read_file(repo_root, file_path)
                    if content:
                        parts.append(f"\n### {file_path}\n```{tag}\n{content}\n```")

        # Get test code from dependencies (S1-TDD arm)
        test_code = self._get_test_code(context)
        if test_code:
            tag = self._code_block_tag
            parts.append(
                f"\n\n## Failing Test (for guidance only — do NOT patch this)\n"
                f"The following test demonstrates the expected behavior. "
                f"Your patch should fix the SOURCE files so this test would pass. "
                f"Do NOT create or modify test files.\n"
                f"```{tag}\n{test_code}\n```"
            )

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

    def _process_output(self, output: Patch, context: Context) -> str:
        """Convert validated Patch to JSON, generating unified diff if possible."""
        repo_root = self._resolve_repo_root(context)

        if repo_root and output.edits:
            unified_diff = self._create_unified_diff(output.edits, repo_root)
            return json.dumps(
                {
                    "patch": unified_diff,
                    "edits": [e.model_dump() for e in output.edits],
                    "reasoning": output.reasoning,
                }
            )

        return output.model_dump_json(indent=2)

    def _get_analysis(self, context: Context) -> Analysis | None:
        """Extract analysis from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "analysis" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return Analysis.model_validate(data)
                    except Exception:
                        pass
        return None

    def _get_localization(self, context: Context) -> Localization | None:
        """Extract localization from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "localize" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return Localization.model_validate(data)
                    except Exception:
                        pass
        return None

    def _get_test_code(self, context: Context) -> str | None:
        """Extract test code from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "test" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact and artifact.content.strip():
                    return artifact.content
        return None

    def _list_repo_files(
        self,
        repo_root: str,
        extensions: tuple[str, ...] = (".py",),
        max_files: int = 80,
    ) -> list[str]:
        """Return source file paths relative to *repo_root*.

        Walks the repository tree, filtering by *extensions* and skipping
        common non-source directories (``__pycache__``, ``.git``, ``node_modules``,
        ``vendor``, ``venv``).  Returns at most *max_files* entries sorted
        alphabetically.
        """
        skip_dirs = {"__pycache__", ".git", "node_modules", "vendor", "venv", ".venv", ".tox"}
        root = Path(repo_root)
        found: list[str] = []
        for dirpath, dirnames, filenames in root.walk():
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in sorted(filenames):
                if any(fname.endswith(ext) for ext in extensions):
                    rel = (dirpath / fname).relative_to(root)
                    found.append(str(rel))
                    if len(found) >= max_files:
                        return found
        return found

    def _read_file(self, repo_root: str, file_path: str) -> str | None:
        """Read file content from repository."""
        full_path = Path(repo_root) / file_path
        if not full_path.exists():
            return None

        try:
            content = full_path.read_text()
            lines = content.split("\n")

            # Truncate if too long
            if len(lines) > self._max_context_lines:
                half = self._max_context_lines // 2
                truncated = (
                    lines[:half]
                    + [
                        "...",
                        f"# ({len(lines) - self._max_context_lines} lines omitted)",
                        "...",
                    ]
                    + lines[-half:]
                )
                return "\n".join(truncated)

            return content
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None

    def _create_unified_diff(
        self,
        edits: list[SearchReplaceEdit],
        repo_root: str,
    ) -> str:
        """Convert search-replace edits to unified diff format."""
        patches = []

        for edit in edits:
            file_path = Path(repo_root) / edit.file
            if not file_path.exists():
                logger.warning(f"File not found: {edit.file}")
                continue

            try:
                original = file_path.read_text()

                # Apply the search-replace
                if edit.search not in original:
                    preview = edit.search[:200].replace('\n', '\\n')
                    logger.warning(
                        "Search string not found in %s: %r", edit.file, preview
                    )
                    continue

                modified = original.replace(edit.search, edit.replace, 1)

                # Generate unified diff
                diff_lines = list(
                    difflib.unified_diff(
                        original.splitlines(keepends=True),
                        modified.splitlines(keepends=True),
                        fromfile=f"a/{edit.file}",
                        tofile=f"b/{edit.file}",
                    )
                )

                if diff_lines:
                    patches.append("".join(diff_lines))

            except Exception as e:
                logger.warning(f"Error processing {edit.file}: {e}")
                continue

        return "\n".join(patches)
