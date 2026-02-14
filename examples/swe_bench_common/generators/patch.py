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
from examples.swe_bench_common.models import Patch, SearchReplaceEdit

from atomicguard.domain.models import Artifact, Context
from atomicguard.domain.prompts import PromptTemplate

logger = logging.getLogger("swe_bench_ablation.generators")


class PatchGenerator(PydanticAIGenerator[Patch]):
    """Generator that produces patches via search-replace blocks.

    The LLM specifies exact code blocks to find and replace.
    This is converted to unified diff format programmatically using difflib.

    Context comes from prompt templates via placeholders like {ap_analysis},
    {ap_fix_approach}, {ap_gen_test}. File content is injected by _build_prompt
    when repo_root is available.
    """

    output_type = Patch

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
        self._max_file_lines = max_file_lines
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
        template: PromptTemplate,
    ) -> str:
        """Build the prompt for patch generation.

        Uses template.render() for base prompt and dependency placeholders,
        then injects additional file content if repo_root is available.

        For singleshot mode (no dependencies), scans the specification for
        file paths and includes their content directly.
        """
        # Get base prompt from template (includes dependency substitution)
        base_prompt = template.render(context)

        # If we have repo_root, try to inject file content
        repo_root = self._resolve_repo_root(context)
        if not repo_root or not self._include_file_content:
            return base_prompt

        # Extract file paths from the rendered prompt (or spec if no deps)
        file_paths = self._extract_file_paths(base_prompt)

        # Singleshot fallback: scan specification for files in repo
        if not file_paths and not context.dependency_artifacts:
            # Use language-specific extensions if available
            extensions = getattr(self, "_lang", None)
            if extensions and hasattr(extensions, "file_extensions"):
                repo_files = self._list_repo_files(
                    repo_root, extensions=extensions.file_extensions
                )
            else:
                repo_files = self._list_repo_files(repo_root)
            file_paths = [f for f in repo_files if f in context.specification]

        if not file_paths:
            return base_prompt

        # Filter out test files
        source_files = [f for f in file_paths if not self._is_test_file(f)]
        if not source_files:
            return base_prompt

        # Build file content section
        parts = [base_prompt]
        parts.append("\n\n## Current File Content")
        tag = self._code_block_tag

        file_errors: list[str] = []
        # Limit to 5 files for singleshot, 3 for normal mode
        max_files = 5 if not context.dependency_artifacts else 3
        for file_path in source_files[:max_files]:
            content, error = self._read_file(repo_root, file_path)
            if error:
                file_errors.append(error)
            elif content:
                parts.append(f"\n### {file_path}\n```{tag}\n{content}\n```")

        if file_errors:
            parts.append("\n\n## File Errors\n" + "\n".join(file_errors))

        # Add search string rules
        parts.append(self._get_search_string_rules())

        return "\n".join(parts)

    def _extract_file_paths(self, prompt: str) -> list[str]:
        """Extract file paths from the rendered prompt.

        Looks for patterns like 'Files: file1.py, file2.py' or
        'files_to_modify: ["file1.py"]' in the prompt text.
        """
        paths: list[str] = []

        # Pattern for "Files: file1.py, file2.py"
        files_match = re.search(r"Files?:\s*([^\n]+)", prompt)
        if files_match:
            files_str = files_match.group(1)
            # Split on comma or whitespace
            for part in re.split(r"[,\s]+", files_str):
                part = part.strip().strip("'\"[]")
                if part.endswith(".py"):
                    paths.append(part)

        # Pattern for JSON-style lists
        json_match = re.search(r'"files":\s*\[([^\]]+)\]', prompt)
        if json_match:
            files_str = json_match.group(1)
            for part in files_str.split(","):
                part = part.strip().strip("\"'")
                if part.endswith(".py"):
                    paths.append(part)

        return list(dict.fromkeys(paths))  # Remove duplicates, preserve order

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
                if any(
                    pattern in path_lower for pattern in utility_patterns
                ) and not self._is_test_file(rel_path):
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

    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file that should not be modified."""
        path_lower = file_path.lower()
        filename = path_lower.split("/")[-1]
        return (
            "/test/" in path_lower
            or "/tests/" in path_lower
            or path_lower.startswith("test/")
            or path_lower.startswith("tests/")
            or "_test.py" in filename
            or filename.startswith("test_")
        )

    def _read_file(
        self, repo_root: str, file_path: str
    ) -> tuple[str | None, str | None]:
        """Read file content from repository with line numbers.

        Returns (content, error_message). If error_message is set, content is None.
        """
        full_path = Path(repo_root) / file_path
        if not full_path.exists():
            return None, None

        try:
            content = full_path.read_text()
            lines = content.split("\n")

            if self._max_file_lines and len(lines) > self._max_file_lines:
                return None, (
                    f"File {file_path} has {len(lines)} lines, exceeding limit of "
                    f"{self._max_file_lines}. Cannot proceed without full file context."
                )

            # Add line numbers for easier reference
            numbered_lines = [f"{i + 1:>4d}|{line}" for i, line in enumerate(lines)]
            return "\n".join(numbered_lines), None
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None, None

    def _get_search_string_rules(self) -> str:
        """Return the search string rules instruction."""
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

    def _create_unified_diff(
        self,
        edits: list[SearchReplaceEdit],
        repo_root: str,
    ) -> str:
        """Convert search-replace edits to unified diff format."""
        patches = []

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

                for edit in file_edits:
                    if edit.search not in modified:
                        preview = edit.search[:200].replace("\n", "\\n")
                        logger.warning(
                            "Search string not found in %s: %r", edit.file, preview
                        )
                        continue
                    modified = modified.replace(edit.search, edit.replace, 1)

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
        template: PromptTemplate,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
        workflow_ref: str | None = None,
    ) -> Artifact:
        """Generate with fatal error on file size issues."""
        # Build prompt and check for fatal file errors
        prompt = self._build_prompt(context, template)

        # Check if prompt contains file size errors
        if "exceeding limit of" in prompt and "Cannot proceed" in prompt:
            from datetime import UTC, datetime
            from types import MappingProxyType
            from uuid import uuid4

            from atomicguard.domain.models import (
                ArtifactStatus,
                ContextSnapshot,
                FeedbackEntry,
            )

            # Extract error message
            error_match = re.search(
                r"File .+ exceeding limit of .+ Cannot proceed[^.]*\.", prompt
            )
            error_msg = (
                error_match.group(0) if error_match else "File size limit exceeded"
            )

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

        return super().generate(
            context, template, action_pair_id, workflow_id, workflow_ref
        )
