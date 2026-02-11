"""AnalysisGenerator: Analyzes bug type, root cause, and fix approach.

Uses PydanticAI structured output to classify the bug and produce a
structured analysis that downstream generators (patch or test) consume
via dependency artifacts.
"""

import logging
from pathlib import Path
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import Analysis

logger = logging.getLogger("swe_bench_ablation.generators")


class AnalysisGenerator(PydanticAIGenerator[Analysis]):
    """Generator that produces structured bug analysis.

    Outputs an Analysis JSON classifying the bug type, identifying
    root cause, affected components, likely files, and fix approach.

    When ``repo_root`` is provided, reads candidate files mentioned in
    the problem statement and includes their content in the prompt for
    more accurate file selection.
    """

    output_type = Analysis

    def __init__(
        self,
        *,
        repo_root: str | None = None,
        max_candidate_files: int = 5,
        **kwargs: Any,
    ):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
        self._repo_root = repo_root
        self._max_candidate_files = max_candidate_files

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for analysis."""
        parts = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Include candidate file contents for code-aware analysis
        if self._repo_root:
            candidates = self._find_candidate_files(context.specification)
            if candidates:
                parts.append("\n\n## Candidate File Contents")
                parts.append("Review these files to identify which contain the bug:")
                for fpath in candidates[: self._max_candidate_files]:
                    content = self._read_file(fpath)
                    if content:
                        # Truncate large files
                        if len(content) > 3000:
                            content = content[:3000] + "\n... (truncated)"
                        parts.append(f"\n### {fpath}\n```python\n{content}\n```")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

    def _find_candidate_files(self, specification: str) -> list[str]:
        """Find files mentioned in problem statement or likely relevant."""
        # Extract file listing from specification
        if "## Repository Structure" not in specification:
            return []

        listing_section = specification.split("## Repository Structure")[1]
        if "```" in listing_section:
            listing = listing_section.split("```")[1]
            repo_files = [f.strip() for f in listing.split("\n") if f.strip()]
        else:
            repo_files = []

        # Find files mentioned in problem text (before Repository Structure)
        problem_text = specification.split("## Repository Structure")[0].lower()

        mentioned = []
        for fpath in repo_files:
            filename = fpath.split("/")[-1].lower()
            # Check if filename or path fragment appears in problem
            if filename in problem_text or any(
                part in problem_text
                for part in fpath.lower().split("/")
                if len(part) > 3
            ):
                mentioned.append(fpath)

        return mentioned

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
