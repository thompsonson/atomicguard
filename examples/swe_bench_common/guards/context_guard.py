"""ContextGuard: Validates code context reading output.

Ensures the context summary has valid JSON matching the ContextSummary
schema with non-empty required fields.
"""

import json
import logging
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from examples.swe_bench_common.models import ContextSummary

logger = logging.getLogger("swe_bench_ablation.guards")


class ContextGuard(GuardInterface):
    """Validates code context reading output.

    Checks:
    - Valid JSON matching ContextSummary schema
    - Non-empty file_path, code_snippet, and summary
    - file_path exists in repository (if repo_root provided)
    """

    def __init__(
        self,
        repo_root: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the guard.

        Args:
            repo_root: Repository root for file existence validation
        """
        self._repo_root = repo_root

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the context artifact.

        Args:
            artifact: The context artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info(
            "[ContextGuard] Validating artifact %s...", artifact.artifact_id[:8]
        )

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="ContextGuard",
            )

        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {data['error']}",
                guard_name="ContextGuard",
            )

        try:
            context = ContextSummary.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="ContextGuard",
            )

        errors: list[str] = []

        if not context.file_path.strip():
            errors.append("file_path is empty")

        if not context.code_snippet.strip():
            errors.append("code_snippet is empty")

        if not context.summary.strip():
            errors.append("summary is empty")

        # Validate file exists in repository and snippet matches real content
        if context.file_path and self._repo_root:
            full_path = Path(self._repo_root) / context.file_path
            if not full_path.exists():
                errors.append(f"File not found in repository: {context.file_path}")
            elif context.code_snippet.strip():
                try:
                    real_content = full_path.read_text(errors="replace")
                    # For multi-file snippets with "--- file: ..." headers,
                    # extract the section for the primary file.
                    snippet_to_check = context.code_snippet
                    header = f"--- file: {context.file_path} ---"
                    if header in snippet_to_check:
                        # Extract content after this header until next header or end
                        start = snippet_to_check.index(header) + len(header)
                        next_header = snippet_to_check.find("\n--- file: ", start)
                        if next_header != -1:
                            snippet_to_check = snippet_to_check[start:next_header]
                        else:
                            snippet_to_check = snippet_to_check[start:]
                        snippet_to_check = snippet_to_check.strip()

                    if snippet_to_check and snippet_to_check not in real_content:
                        preview_lines = real_content.splitlines()[:30]
                        preview = "\n".join(preview_lines)
                        errors.append(
                            f"code_snippet does not match actual file content for "
                            f"{context.file_path}. First 30 lines of real file:\n"
                            f"{preview}"
                        )
                except Exception as e:
                    logger.warning(
                        "[ContextGuard] Could not read %s for verification: %s",
                        full_path,
                        e,
                    )

        if errors:
            feedback = "Context validation failed:\n- " + "\n- ".join(errors)
            logger.info("[ContextGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="ContextGuard",
            )

        feedback = (
            f"Context valid: file={context.file_path}, "
            f"{len(context.relevant_functions)} functions, "
            f"{len(context.code_snippet)} chars snippet"
        )
        logger.info("[ContextGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="ContextGuard",
        )
