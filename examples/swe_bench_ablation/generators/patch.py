"""PatchGenerator: Generates patches using search-replace format.

Uses an LLM to generate code changes as search-replace blocks,
which are then converted to unified diff format programmatically.
This approach is more reliable than asking LLMs to generate diff format directly.
"""

import difflib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    FeedbackEntry,
)
from atomicguard.domain.prompts import PromptTemplate

from ..models import Localization, Patch, SearchReplaceEdit

logger = logging.getLogger("swe_bench_ablation.generators")


class PatchGenerator(GeneratorInterface):
    """Generator that produces patches via search-replace blocks.

    The LLM specifies exact code blocks to find and replace.
    This is converted to unified diff format programmatically using difflib.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434/v1",
        timeout: float = 180.0,
        include_file_content: bool = True,
        max_context_lines: int = 100,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the patch generator.

        Args:
            model: LLM model to use
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            include_file_content: Whether to include file content in prompt
            max_context_lines: Maximum lines of context per file
        """
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("openai library required: pip install openai") from err

        self._model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=timeout,
        )
        self._include_file_content = include_file_content
        self._max_context_lines = max_context_lines
        self._attempt_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,
        action_pair_id: str = "ap_patch",
        workflow_id: str = "unknown",
        workflow_ref: str | None = None,
    ) -> Artifact:
        """Generate a patch to fix the identified bug.

        Args:
            context: Execution context with localization and specification
            template: Prompt template from prompts.json
            action_pair_id: Identifier for this action pair
            workflow_id: UUID of the workflow execution
            workflow_ref: Content-addressed workflow hash

        Returns:
            Artifact containing unified diff patch
        """
        logger.info("[PatchGenerator] Generating patch...")

        # Get repo_root from context metadata if available
        repo_root = (
            context.ambient.repository.metadata.get("repo_root")
            if hasattr(context.ambient.repository, "metadata")
            else None
        )

        # Build prompt
        prompt = self._build_prompt(context, template, repo_root)

        # Get system prompt from template or use default
        system_prompt = (
            template.role
            if template
            else "You are a senior software engineer writing minimal, correct bug fixes."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.3,
            )
            content = response.choices[0].message.content or ""
            logger.debug(f"[PatchGenerator] Response: {content[:200]}...")

            # Parse search-replace blocks and convert to patch
            result = self._process_response(content, repo_root)

        except Exception as e:
            logger.warning(f"[PatchGenerator] LLM call failed: {e}")
            result = json.dumps({"error": str(e)})

        self._attempt_counter += 1

        # Build context snapshot
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
            content=result,
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now(UTC).isoformat(),
            attempt_number=self._attempt_counter,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=context_snapshot,
            workflow_ref=workflow_ref,
        )

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
        repo_root: str | None,
    ) -> str:
        """Build the prompt for patch generation."""
        parts = []

        # Task from template
        if template:
            parts.append(template.task)

        # Problem statement
        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Get localization from dependencies
        localization = self._get_localization(context)
        if localization:
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
                for file_path in localization.files[:3]:  # Limit to 3 files
                    content = self._read_file(repo_root, file_path)
                    if content:
                        parts.append(f"\n### {file_path}\n```python\n{content}\n```")

        # Constraints from template
        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        # Output format instruction
        parts.append(
            """

## Output Format
Return a JSON object with search-replace edits:
```json
{
  "edits": [
    {
      "file": "path/to/file.py",
      "search": "exact code to find\\nincluding multiple lines",
      "replace": "new code to replace with\\nincluding multiple lines"
    }
  ],
  "reasoning": "Brief explanation of the fix"
}
```

CRITICAL REQUIREMENTS:
1. EXACT MATCH: The 'search' string must match the file content EXACTLY (including whitespace/indentation)
2. VALID PYTHON: The 'replace' code must be syntactically valid
3. MINIMAL CHANGE: Only change what's necessary to fix the bug
4. PRESERVE STYLE: Match existing code style

TIPS:
- Include enough surrounding context in 'search' to ensure uniqueness
- Copy the exact indentation from the file
- Include 1-2 lines before/after your target if the match is ambiguous
"""
        )

        # Add feedback if retry
        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

    def _get_localization(self, context: Context) -> Localization | None:
        """Extract localization from dependency artifacts."""
        # Look for localization in dependencies
        for dep_id, artifact_id in context.dependency_artifacts:
            if "localize" in dep_id.lower():
                # Get the artifact content
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return Localization.model_validate(data)
                    except Exception:
                        pass
        return None

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

    def _process_response(self, content: str, repo_root: str | None) -> str:
        """Parse response and convert to unified diff."""
        # Extract JSON from response
        json_match = re.search(r"```json?\s*([\s\S]*?)```", content)
        if json_match:
            json_str = json_match.group(1)
        else:
            obj_match = re.search(r"\{[\s\S]*\}", content)
            if obj_match:
                json_str = obj_match.group(0)
            else:
                return json.dumps({"error": "No JSON found in response"})

        try:
            data = json.loads(json_str)

            # Validate with Pydantic
            patch = Patch.model_validate(data)

            # Convert to unified diff if repo_root available
            if repo_root and patch.edits:
                unified_diff = self._create_unified_diff(patch.edits, repo_root)
                return json.dumps(
                    {
                        "patch": unified_diff,
                        "edits": [e.model_dump() for e in patch.edits],
                        "reasoning": patch.reasoning,
                    }
                )

            # Return raw edits if can't create diff
            return patch.model_dump_json(indent=2)

        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})
        except Exception as e:
            return json.dumps({"error": f"Processing failed: {e}"})

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
                    logger.warning(f"Search string not found in {edit.file}")
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
