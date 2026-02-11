"""PatchGenerator: Generates patches using search-replace format.

Uses PydanticAI structured output to generate code changes as
search-replace blocks, which are then converted to unified diff
format programmatically. This approach is more reliable than asking
LLMs to generate diff format directly.
"""

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Artifact, Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import Analysis, Localization, Patch, SearchReplaceEdit

logger = logging.getLogger("swe_bench_ablation.generators")


class PatchGenerator(PydanticAIGenerator[Patch]):
    """Generator that produces patches via search-replace blocks.

    The LLM specifies exact code blocks to find and replace.
    This is converted to unified diff format programmatically using difflib.
    """

    output_type = Patch

    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file that should not be modified.

        Test files are excluded from the patch context because:
        1. They are not meant to be modified by bug fixes
        2. Including them can mislead the LLM into patching tests instead of source
        """
        path_lower = file_path.lower()
        filename = path_lower.split("/")[-1]
        return (
            "/test/" in path_lower
            or "/tests/" in path_lower
            or path_lower.startswith("test/")
            or path_lower.startswith("tests/")
            or "_test.py" in filename
            or filename.startswith("test_")  # test_*.py files
        )

    def __init__(
        self,
        *,
        include_file_content: bool = True,
        max_file_lines: int | None = None,
        repo_root: str | None = None,
        code_block_tag: str = "python",
        **kwargs: Any,
    ):
        kwargs.setdefault("temperature", 0.3)
        super().__init__(**kwargs)
        self._include_file_content = include_file_content
        self._max_file_lines = max_file_lines  # None = no limit (optional safeguard)
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
    ) -> str | tuple[str, list[str]]:
        """Build the prompt for patch generation.

        Returns:
            Either a prompt string (for base class compatibility) or a tuple
            of (prompt, file_errors) when called via _build_prompt_with_errors().
        """
        prompt, errors = self._build_prompt_with_errors(context, template)
        # For base class compatibility, return just the prompt
        # The generate() override handles errors separately
        return prompt

    def _build_prompt_with_errors(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> tuple[str, list[str]]:
        """Build the prompt for patch generation with error tracking.

        Returns:
            Tuple of (prompt_string, file_errors_list).
        """
        repo_root = self._resolve_repo_root(context)
        parts: list[str] = []
        file_errors: list[str] = []

        # Task from template
        if template:
            parts.append(template.task)

        # Problem statement
        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Track test files to exclude and warn about
        excluded_test_files: list[str] = []

        # Get analysis from dependencies (S1-direct and S1-TDD arms)
        analysis = self._get_analysis(context)
        if analysis:
            parts.append("\n\n## Bug Analysis")
            parts.append(f"Bug type: {analysis.bug_type.value}")
            parts.append(f"Root cause: {analysis.root_cause_hypothesis}")
            # Show all files in analysis, but mark test files
            source_files = [f for f in analysis.files if not self._is_test_file(f)]
            test_files = [f for f in analysis.files if self._is_test_file(f)]
            parts.append(f"Files: {', '.join(source_files)}")
            if test_files:
                excluded_test_files.extend(test_files)
            parts.append(f"Fix approach: {analysis.fix_approach}")

            # Include file content from analysis (excluding test files)
            if self._include_file_content and repo_root:
                parts.append("\n\n## Current File Content")
                tag = self._code_block_tag
                files_to_include = [
                    f for f in analysis.files if not self._is_test_file(f)
                ]
                for file_path in files_to_include[:3]:
                    content, error = self._read_file(repo_root, file_path)
                    if error:
                        file_errors.append(error)
                    elif content:
                        parts.append(f"\n### {file_path}\n```{tag}\n{content}\n```")
                # Add file content anchoring instruction
                parts.append(self._get_search_string_rules())

        # Get localization from dependencies (baseline arm)
        localization = self._get_localization(context)
        if localization and not analysis:
            # Filter out test files from localization
            source_files = [f for f in localization.files if not self._is_test_file(f)]
            test_files = [f for f in localization.files if self._is_test_file(f)]
            if test_files:
                excluded_test_files.extend(test_files)

            parts.append("\n\n## Files to Modify")
            parts.append(f"Files: {', '.join(source_files)}")
            if localization.functions:
                # Filter functions to only those in source files
                funcs = [
                    f"{f.name} in {f.file}"
                    for f in localization.functions
                    if not self._is_test_file(f.file)
                ]
                if funcs:
                    parts.append(f"Functions: {', '.join(funcs)}")
            if localization.reasoning:
                parts.append(f"Reasoning: {localization.reasoning}")

            # Include file content (excluding test files)
            if self._include_file_content and repo_root:
                parts.append("\n\n## Current File Content")
                tag = self._code_block_tag
                files_to_include = [
                    f for f in localization.files if not self._is_test_file(f)
                ]
                for file_path in files_to_include[:3]:  # Limit to 3 files
                    content, error = self._read_file(repo_root, file_path)
                    if error:
                        file_errors.append(error)
                    elif content:
                        parts.append(f"\n### {file_path}\n```{tag}\n{content}\n```")
                # Add file content anchoring instruction
                parts.append(self._get_search_string_rules())

        # Singleshot fallback: include content of files referenced in the problem
        if (
            not analysis
            and not localization
            and self._include_file_content
            and repo_root
        ):
            repo_files = self._list_repo_files(repo_root)
            referenced = [f for f in repo_files if f in context.specification]
            # Filter out test files
            source_referenced = [f for f in referenced if not self._is_test_file(f)]
            test_referenced = [f for f in referenced if self._is_test_file(f)]
            if test_referenced:
                excluded_test_files.extend(test_referenced)
            if source_referenced:
                parts.append("\n\n## Current File Content")
                tag = self._code_block_tag
                for file_path in source_referenced[:5]:
                    content, error = self._read_file(repo_root, file_path)
                    if error:
                        file_errors.append(error)
                    elif content:
                        parts.append(f"\n### {file_path}\n```{tag}\n{content}\n```")
                # Add file content anchoring instruction
                parts.append(self._get_search_string_rules())

        # Add "Do Not Modify" warning for excluded test files
        if excluded_test_files:
            unique_test_files = sorted(set(excluded_test_files))
            parts.append("\n\n## Files NOT to Modify")
            parts.append(
                "The following test files were identified but should NOT be modified. "
                "Bug fixes should only modify source files, not tests:"
            )
            parts.append(f"- {', '.join(unique_test_files)}")

        # Discover and include utility functions when analysis is present
        if (analysis or localization) and repo_root:
            utility_files = self._discover_utility_files(repo_root, max_files=5)
            if utility_files:
                utility_parts: list[str] = []
                for util_file in utility_files:
                    signatures = self._extract_function_signatures(repo_root, util_file)
                    if signatures:
                        utility_parts.append(f"\n### {util_file}")
                        for sig in signatures[:10]:  # Limit signatures per file
                            utility_parts.append(f"- `{sig}`")
                if utility_parts:
                    parts.append("\n\n## Available Utilities")
                    parts.append(
                        "The following utility functions are available in the codebase. "
                        "Consider using these instead of reimplementing similar logic:"
                    )
                    parts.extend(utility_parts)

        # Get test code from dependencies (S1-TDD arm)
        test_code = self._get_test_code(context)
        if test_code:
            tag = self._code_block_tag
            parts.append(
                f"\n\n## Generated Test (REFERENCE ONLY — DO NOT MODIFY)\n"
                f"This test demonstrates the expected behavior after the bug is fixed. "
                f"Use it to understand what the correct behavior should be.\n\n"
                f"**IMPORTANT:** Your patch should fix the SOURCE files so this test passes. "
                f"Do NOT create or modify any test files. Do NOT include test code in your patch.\n"
                f"```{tag}\n{test_code}\n```"
            )

        # Constraints from template
        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        # Additional constraints from context (includes stagnation warnings)
        if context.ambient.constraints:
            parts.append(f"\n\n{context.ambient.constraints}")

        # Add feedback history with guard chain
        if context.feedback_history and template:
            # Show the most recent 3 attempts (most recent first)
            history_to_show = list(context.feedback_history)[-3:]
            num_attempts = len(context.feedback_history)

            parts.append(
                f"\n\n## Previous Attempts ({num_attempts} total, showing last {len(history_to_show)})"
            )
            parts.append(
                "Review what went wrong and fix the issues. "
                "Each rejection shows which guard failed and why."
            )

            for i, (prev_content, feedback) in enumerate(reversed(history_to_show)):
                attempt_num = num_attempts - i
                guard_name = self._extract_guard_name(feedback)
                parts.append(f"\n### Attempt {attempt_num} - Rejected by {guard_name}")
                parts.append(f"{feedback}")

                # Show the edits from the most recent attempt only
                if i == 0 and prev_content:
                    prev_edits = self._extract_edits_from_content(prev_content)
                    if prev_edits:
                        parts.append("\n**Your Edits (for reference):**")
                        parts.append(
                            "Ensure search strings match EXACTLY (including whitespace):"
                        )
                        for edit in prev_edits[:3]:  # Limit to 3 edits
                            parts.append(f"\nFile: `{edit.get('file', 'unknown')}`")
                            search = edit.get("search", "")[:200]
                            parts.append(f"Search (first 200 chars): `{search}`")

        return "\n".join(parts), file_errors

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

    def _extract_edits_from_content(self, content: str) -> list[dict]:
        """Extract edits from a previous patch artifact's content.

        The content is JSON with an 'edits' key containing search/replace blocks.
        Returns a list of edit dicts, or empty list if parsing fails.
        """
        try:
            data = json.loads(content)
            edits = data.get("edits", [])
            if isinstance(edits, list):
                return edits
        except (json.JSONDecodeError, TypeError):
            pass
        return []

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
        skip_dirs = {
            "__pycache__",
            ".git",
            "node_modules",
            "vendor",
            "venv",
            ".venv",
            ".tox",
        }
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

    def _discover_utility_files(
        self,
        repo_root: str,
        max_files: int = 10,
    ) -> list[str]:
        """Discover utility files that may contain helper functions.

        Looks for files with 'utils', 'helpers', 'common', or 'utility' in
        their path. These often contain reusable functions the LLM should
        know about when generating patches.
        """
        utility_patterns = ("utils", "helpers", "common", "utility", "utilities")
        skip_dirs = {
            "__pycache__",
            ".git",
            "node_modules",
            "vendor",
            "venv",
            ".venv",
            ".tox",
        }
        root = Path(repo_root)
        found: list[str] = []

        for dirpath, dirnames, filenames in root.walk():
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in sorted(filenames):
                if not fname.endswith(".py"):
                    continue
                rel_path = str((dirpath / fname).relative_to(root))
                path_lower = rel_path.lower()
                # Check if any utility pattern is in the path
                if any(pattern in path_lower for pattern in utility_patterns):
                    # Exclude test files
                    if not self._is_test_file(rel_path):
                        found.append(rel_path)
                        if len(found) >= max_files:
                            return found
        return found

    def _extract_function_signatures(
        self,
        repo_root: str,
        file_path: str,
        max_signatures: int = 20,
    ) -> list[str]:
        """Extract function signatures from a Python file.

        Returns a list of function signatures (def lines) without the
        implementation. This provides a concise overview of available
        functions without including the full file content.
        """
        full_path = Path(repo_root) / file_path
        if not full_path.exists():
            return []

        try:
            content = full_path.read_text()
        except Exception:
            return []

        signatures: list[str] = []
        # Match function definitions (handles multiline signatures)
        # Pattern: def name(...): with optional type hints and return type
        pattern = re.compile(
            r"^(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:)",
            re.MULTILINE,
        )

        for match in pattern.finditer(content):
            sig = match.group(1).strip()
            # Clean up multiline signatures
            sig = " ".join(sig.split())
            signatures.append(sig)
            if len(signatures) >= max_signatures:
                break

        return signatures

    def _read_file(
        self, repo_root: str, file_path: str
    ) -> tuple[str | None, str | None]:
        """Read file content from repository with line numbers.

        Returns (content, error_message). If error_message is set, content is None.
        Explicit failure is better than silent truncation.
        Line numbers are added to help the model reference specific lines.
        """
        full_path = Path(repo_root) / file_path
        if not full_path.exists():
            return None, None

        try:
            content = full_path.read_text()
            lines = content.split("\n")

            # Optional safeguard: fail if file exceeds limit
            if self._max_file_lines and len(lines) > self._max_file_lines:
                return None, (
                    f"File {file_path} has {len(lines)} lines, exceeding limit of "
                    f"{self._max_file_lines}. Cannot proceed without full file context."
                )

            # Add line numbers for easier reference
            # Format: right-aligned number, pipe, then code (no space after pipe)
            # This prevents models from including "| " in search strings
            numbered_lines = [f"{i + 1:>4d}|{line}" for i, line in enumerate(lines)]
            return "\n".join(numbered_lines), None
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None, None

    def _extract_guard_name(self, feedback: str) -> str:
        """Extract guard name from feedback text.

        Guards often include their name in the feedback (e.g., "guard_name=PatchGuard").
        Falls back to detecting common guard names from patterns in the text.
        """
        # Common guard name patterns in feedback
        patterns = [
            r"guard_name[=:]?\s*['\"]?(\w+Guard)['\"]?",
            r"Rejected by (\w+Guard)",
            r"(\w+Guard)\s*:\s*",
            r"^(PatchGuard|TestGreenGuard|TestRedGuard|GeneratorValidation|DiffReviewGuard)",
        ]

        for pattern in patterns:
            match = re.search(pattern, feedback, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1)

        # Infer from feedback content
        feedback_lower = feedback.lower()
        if "search string" in feedback_lower or "not found" in feedback_lower:
            return "PatchGuard"
        if "test" in feedback_lower and (
            "pass" in feedback_lower or "fail" in feedback_lower
        ):
            return "TestGuard"
        if "json" in feedback_lower or "schema" in feedback_lower:
            return "GeneratorValidation"
        if "review" in feedback_lower or "verdict" in feedback_lower:
            return "DiffReviewGuard"

        return "Guard"

    def _get_search_string_rules(self) -> str:
        """Return the search string rules instruction.

        This is a critical instruction that helps prevent hallucinated search strings.
        The model must copy-paste exact code from the file content shown above.
        """
        return """

## CRITICAL: Search String Rules

Your SEARCH strings MUST be copied EXACTLY from the file contents shown above.

**DO NOT:**
- Write search strings from memory or training data
- Modify whitespace, indentation, or line breaks
- Guess what the code looks like

**DO:**
- Copy-paste exact lines from the numbered file content above
- Include the EXACT indentation (spaces/tabs) as shown
- Use line numbers to verify you have the right code

**NOTE:** The line numbers (e.g., "  42|") are for reference only.
Do NOT include line numbers in your search strings - only the code after the "|".

If the code you need to modify isn't shown above, the patch cannot succeed.
"""

    def _create_unified_diff(
        self,
        edits: list[SearchReplaceEdit],
        repo_root: str,
    ) -> str:
        """Convert search-replace edits to unified diff format.

        Groups edits by file and applies them cumulatively to avoid
        conflicting hunks when multiple edits target the same file.
        """
        patches = []

        # Group edits by file to apply cumulatively
        edits_by_file: dict[str, list[SearchReplaceEdit]] = {}
        for edit in edits:
            edits_by_file.setdefault(edit.file, []).append(edit)

        for file_name, file_edits in edits_by_file.items():
            file_path = Path(repo_root) / file_name
            if not file_path.exists():
                logger.warning(f"File not found: {file_name}")
                continue

            try:
                original = file_path.read_text()
                modified = original

                # Apply all edits to this file cumulatively
                for edit in file_edits:
                    if edit.search not in modified:
                        preview = edit.search[:200].replace("\n", "\\n")
                        logger.warning(
                            "Search string not found in %s: %r", edit.file, preview
                        )
                        continue
                    modified = modified.replace(edit.search, edit.replace, 1)

                # Generate unified diff from original to fully-modified
                diff_lines = list(
                    difflib.unified_diff(
                        original.splitlines(keepends=True),
                        modified.splitlines(keepends=True),
                        fromfile=f"a/{file_name}",
                        tofile=f"b/{file_name}",
                    )
                )

                if diff_lines:
                    patches.append("".join(diff_lines))

            except Exception as e:
                logger.warning(f"Error processing {file_name}: {e}")
                continue

        return "\n".join(patches)

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
        workflow_ref: str | None = None,
    ) -> Artifact:
        """Generate with fatal error on file size issues.

        Overrides base to check for file errors after prompt building.
        If file errors occurred (e.g., file exceeds size limit), returns
        an artifact with fatal error marker in metadata.
        """
        # Build prompt and collect any file errors (no instance state mutation)
        _, file_errors = self._build_prompt_with_errors(context, template)

        # Check for file size errors before calling LLM - these should be fatal
        if file_errors:
            from datetime import UTC, datetime
            from types import MappingProxyType
            from uuid import uuid4

            from atomicguard.domain.models import (
                ArtifactStatus,
                ContextSnapshot,
                FeedbackEntry,
            )

            error_msg = "\n".join(file_errors)
            context_snapshot = ContextSnapshot(
                workflow_id=workflow_id,
                specification=context.specification,
                constraints=context.ambient.constraints,
                feedback_history=tuple(
                    FeedbackEntry(artifact_id=aid, feedback=fb)
                    for aid, fb in context.feedback_history
                ),
                dependency_artifacts=context.dependency_artifacts,
            )
            return Artifact(
                artifact_id=str(uuid4()),
                workflow_id=workflow_id,
                content=json.dumps({"error": error_msg}),
                previous_attempt_id=None,
                parent_action_pair_id=None,
                action_pair_id=action_pair_id,
                created_at=datetime.now(UTC).isoformat(),
                attempt_number=0,
                status=ArtifactStatus.PENDING,
                guard_result=None,
                context=context_snapshot,
                workflow_ref=workflow_ref,
                metadata=MappingProxyType(
                    {
                        "generator_error": error_msg,
                        "generator_error_kind": "fatal_file_size",
                    }
                ),
            )

        # No file errors - proceed with normal generation
        return super().generate(
            context, template, action_pair_id, workflow_id, workflow_ref
        )
