"""AnalysisGuard: Validates structured bug analysis output.

Ensures the analysis has valid JSON matching the Analysis schema,
with non-empty required fields and at least one likely file.
"""

import json
import logging
from pathlib import Path
from typing import Any

from examples.swe_bench_common.models import Analysis

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

logger = logging.getLogger("swe_bench_ablation.guards")


class AnalysisGuard(GuardInterface):
    """Validates analysis output.

    Checks:
    - Valid JSON matching Analysis schema
    - Non-empty root_cause_hypothesis and fix_approach
    - At least one files entry
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
        """Validate the analysis artifact.

        Args:
            artifact: The analysis artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info(
            "[AnalysisGuard] Validating artifact %s...", artifact.artifact_id[:8]
        )

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="AnalysisGuard",
            )

        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {data['error']}",
                guard_name="AnalysisGuard",
            )

        try:
            analysis = Analysis.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="AnalysisGuard",
            )

        errors: list[str] = []

        if not analysis.root_cause_hypothesis.strip():
            errors.append("root_cause_hypothesis is empty")

        if not analysis.fix_approach.strip():
            errors.append("fix_approach is empty")

        if not analysis.files:
            errors.append("No files identified. Must identify at least one file.")

        # Validate files exist in repository
        if analysis.files and self._repo_root:
            repo_path = Path(self._repo_root)
            missing_files = []
            for f in analysis.files:
                if not (repo_path / f).exists():
                    missing_files.append(f)
            if missing_files:
                errors.append(
                    f"Files not found in repository: {', '.join(missing_files[:3])}"
                    + (
                        f" (and {len(missing_files) - 3} more)"
                        if len(missing_files) > 3
                        else ""
                    )
                )

        if errors:
            feedback = "Analysis validation failed:\n- " + "\n- ".join(errors)
            logger.info("[AnalysisGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="AnalysisGuard",
            )

        feedback = (
            f"Analysis valid: bug_type={analysis.bug_type.value}, "
            f"{len(analysis.files)} files, "
            f"confidence={analysis.confidence}"
        )
        logger.info("[AnalysisGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="AnalysisGuard",
        )
