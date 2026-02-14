"""ContextReadGenerator: Reads code context directly from disk.

Reads the actual files identified by localization and builds a
ContextSummary from real file content — no LLM involved.
"""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from examples.swe_bench_common.models import ContextSummary, Localization

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    FeedbackEntry,
)
from atomicguard.domain.prompts import PromptTemplate

logger = logging.getLogger("swe_bench_common.generators.context_read")


class ContextReadGenerator(GeneratorInterface):
    """Generator that reads code context directly from disk.

    Reads files identified by ``ap_localise_issue`` and builds a
    :class:`ContextSummary` from their real content.  No LLM is used —
    this eliminates hallucinated file content.

    Requires ``repo_root`` at construction time.
    """

    def __init__(self, *, repo_root: str, **kwargs: Any):  # noqa: ARG002
        """Initialize the disk-based context reader.

        Args:
            repo_root: Absolute path to the cloned repository.
            **kwargs: Ignored (accepts LLM params like model/base_url harmlessly).

        Raises:
            ValueError: If *repo_root* is not provided or is empty.
        """
        if not repo_root:
            raise ValueError(
                "ContextReadGenerator requires repo_root for disk-based "
                "file reading, but none was provided."
            )
        self._repo_root = Path(repo_root)
        self._attempt_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate,  # noqa: ARG002
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
        workflow_ref: str | None = None,
    ) -> Artifact:
        """Read files from disk and build a ContextSummary artifact.

        Steps:
        1. Fetch the localization artifact from dependency_artifacts
        2. Parse it as a :class:`Localization` model
        3. Read each file from disk
        4. Build a :class:`ContextSummary` with real code content

        Args:
            context: Execution context with dependency artifacts.
            template: Prompt template (ignored — no LLM).
            action_pair_id: Identifier for this action pair.
            workflow_id: UUID of the workflow execution.
            workflow_ref: Content-addressed workflow hash.

        Returns:
            Artifact containing the serialised ContextSummary.
        """
        self._attempt_counter += 1
        metadata: dict[str, Any] = {"generator": "ContextReadGenerator"}

        try:
            content = self._read_context(context)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            content = json.dumps({"error": error_msg})
            metadata["generator_error"] = error_msg
            metadata["generator_error_kind"] = "infrastructure"
            logger.warning("[ContextReadGenerator] Failed: %s", e)

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
            content=content,
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now(UTC).isoformat(),
            attempt_number=self._attempt_counter,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=context_snapshot,
            workflow_ref=workflow_ref,
            metadata=MappingProxyType(metadata),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_context(self, context: Context) -> str:
        """Fetch localization, read files, return serialised ContextSummary."""
        localization = self._get_localization(context)

        file_contents: dict[str, str] = {}
        for file_path in localization.files:
            full_path = self._repo_root / file_path
            if full_path.is_file():
                try:
                    file_contents[file_path] = full_path.read_text(errors="replace")
                except Exception as e:
                    logger.warning(
                        "[ContextReadGenerator] Could not read %s: %s", file_path, e
                    )

        if not file_contents:
            raise RuntimeError(
                f"No readable files found in repository. "
                f"Localization files: {localization.files}"
            )

        # Build code snippet with file headers
        snippet_parts: list[str] = []
        for path, content in file_contents.items():
            snippet_parts.append(f"--- file: {path} ---")
            snippet_parts.append(content)
        code_snippet = "\n".join(snippet_parts)

        # Extract function/class names from localization
        relevant_functions = [f.name for f in localization.functions]
        relevant_classes = self._extract_classes(file_contents)

        # Parse imports from file content
        imports_used = self._extract_imports(file_contents)

        # Build summary
        paths = list(file_contents.keys())
        summary = f"Code context from {len(paths)} file(s): {', '.join(paths)}"

        context_summary = ContextSummary(
            file_path=paths[0],
            relevant_functions=relevant_functions,
            relevant_classes=relevant_classes,
            imports_used=imports_used,
            code_snippet=code_snippet,
            summary=summary,
        )

        return context_summary.model_dump_json()

    def _get_localization(self, context: Context) -> Localization:
        """Fetch and parse the localization artifact from dependencies."""
        loc_artifact_id = context.get_dependency("ap_localise_issue")
        if not loc_artifact_id:
            raise RuntimeError(
                "ContextReadGenerator requires 'ap_localise_issue' dependency "
                "but it was not found in context.dependency_artifacts"
            )

        artifact = context.ambient.repository.get_artifact(loc_artifact_id)
        data = json.loads(artifact.content)
        return Localization.model_validate(data)

    @staticmethod
    def _extract_imports(file_contents: dict[str, str]) -> list[str]:
        """Extract import statements from file contents."""
        imports: list[str] = []
        seen: set[str] = set()
        pattern = re.compile(
            r"^(?:from\s+\S+\s+import\s+.+|import\s+.+)$", re.MULTILINE
        )
        for content in file_contents.values():
            for match in pattern.findall(content):
                line = match.strip()
                if line not in seen:
                    seen.add(line)
                    imports.append(line)
        return imports

    @staticmethod
    def _extract_classes(file_contents: dict[str, str]) -> list[str]:
        """Extract class names from file contents."""
        classes: list[str] = []
        seen: set[str] = set()
        pattern = re.compile(r"^class\s+(\w+)", re.MULTILINE)
        for content in file_contents.values():
            for match in pattern.findall(content):
                if match not in seen:
                    seen.add(match)
                    classes.append(match)
        return classes
