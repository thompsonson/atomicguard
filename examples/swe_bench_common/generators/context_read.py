"""ContextReadGenerator: Reads and summarizes relevant code context.

Builds on localization to read the actual code files and provide
a summary of the relevant context around the bug location.
"""

import json
import logging
from pathlib import Path
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import ContextSummary, Localization

logger = logging.getLogger("swe_bench_ablation.generators")


class ContextReadGenerator(PydanticAIGenerator[ContextSummary]):
    """Generator that reads and summarizes code context.

    Uses the localization from ap_localise_issue to read the actual
    file contents and provide a contextual summary.
    """

    output_type = ContextSummary

    def __init__(
        self,
        *,
        repo_root: str | None = None,
        max_file_size: int = 5000,
        **kwargs: Any,
    ):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
        self._repo_root = repo_root
        self._max_file_size = max_file_size

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for context reading."""
        parts = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Get localization from dependencies
        localization = self._get_localization(context)
        if localization:
            parts.append("\n\n## Localized Files")
            parts.append(f"Files: {', '.join(localization.files)}")
            if localization.functions:
                func_names = [f"{f.file}:{f.name}" for f in localization.functions]
                parts.append(f"Functions: {', '.join(func_names)}")

            # Include file contents if repo_root is available
            if self._repo_root:
                parts.append("\n\n## File Contents")
                for file_path in localization.files[:3]:  # Limit to 3 files
                    content = self._read_file(file_path)
                    if content:
                        if len(content) > self._max_file_size:
                            content = content[: self._max_file_size] + "\n... (truncated)"
                        parts.append(f"\n### {file_path}\n```python\n{content}\n```")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

    def _get_localization(self, context: Context) -> Localization | None:
        """Extract localization from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "localise_issue" in dep_id.lower() or "localize" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return Localization.model_validate(data)
                    except Exception:
                        pass
        return None

    def _read_file(self, file_path: str) -> str | None:
        """Read file content from repository."""
        if not self._repo_root:
            return None
        full_path = Path(self._repo_root) / file_path
        if not full_path.exists():
            return None
        try:
            return full_path.read_text()
        except Exception:
            return None
