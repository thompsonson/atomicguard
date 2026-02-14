"""FormatGuard: Detects common LLM output formatting issues.

A detect-and-reject guard that catches JSON-escaping artefacts and other
formatting problems before downstream guards run.  Designed to sit first
in a CompositeGuard chain so that clear, actionable feedback reaches the
LLM without wasting a more expensive guard evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from .escape_utils import detect_escape_issues

logger = logging.getLogger("swe_bench_common.guards.format")


class FormatGuard(GuardInterface):
    """Rejects artifacts whose content contains formatting/escaping issues.

    Checks:
    - Over-escaped quotes (``\\\"``, ``\\'``)
    - Over-escaped whitespace (``\\\\n``, ``\\\\t``)
    - Single-line content that should be multi-line
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ARG002
        """Initialize the guard."""

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate artifact content for formatting issues.

        Args:
            artifact: The artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        content = artifact.content
        if not content or not content.strip():
            # Let downstream guards handle empty content
            return GuardResult(
                passed=True,
                feedback="No content to check for formatting issues",
                guard_name="FormatGuard",
            )

        feedback = detect_escape_issues(content)
        if feedback:
            logger.info(
                "[FormatGuard] REJECTED artifact %s: escape issues detected",
                artifact.artifact_id[:8],
            )
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="FormatGuard",
            )

        logger.debug("[FormatGuard] PASSED artifact %s", artifact.artifact_id[:8])
        return GuardResult(
            passed=True,
            feedback="No formatting issues detected",
            guard_name="FormatGuard",
        )
