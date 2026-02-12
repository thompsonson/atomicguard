"""EditPlanGuard: Validates edit plan output.

Ensures the edit plan has valid JSON matching the EditPlan schema,
non-empty rationale, and that all referenced files exist in the repository.
"""

import json
import logging
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from examples.swe_bench_common.models import EditPlan

logger = logging.getLogger("swe_bench_ablation.guards")


class EditPlanGuard(GuardInterface):
    """Validates edit plan output.

    Checks:
    - Valid JSON matching EditPlan schema
    - Non-empty rationale
    - At least one file to edit
    - All referenced files exist in repository (if repo_root provided)
    """

    def __init__(
        self,
        repo_root: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the guard.

        Args:
            repo_root: Repository root for file validation
        """
        self._repo_root = repo_root

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the edit plan artifact.

        Args:
            artifact: The edit plan artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info(
            "[EditPlanGuard] Validating artifact %s...", artifact.artifact_id[:8]
        )

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="EditPlanGuard",
            )

        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {data['error']}",
                guard_name="EditPlanGuard",
            )

        try:
            plan = EditPlan.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="EditPlanGuard",
            )

        errors: list[str] = []

        if not plan.rationale.strip():
            errors.append("rationale is empty")

        # Validate files exist (if repo_root provided)
        if self._repo_root:
            repo_path = Path(self._repo_root)
            missing_files = [
                fe.file
                for fe in plan.files_to_edit
                if not (repo_path / fe.file).exists()
            ]
            if missing_files:
                truncated = missing_files[:3]
                suffix = (
                    f" (and {len(missing_files) - 3} more)"
                    if len(missing_files) > 3
                    else ""
                )
                errors.append(
                    f"These files do not exist in the repository: "
                    f"{', '.join(truncated)}{suffix}. "
                    f"Only reference files that actually exist in the codebase."
                )

        if errors:
            feedback = "Edit plan validation failed:\n- " + "\n- ".join(errors)
            logger.info("[EditPlanGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="EditPlanGuard",
            )

        feedback = (
            f"Edit plan valid: {len(plan.files_to_edit)} files, "
            f"{len(plan.import_changes)} import changes"
        )
        logger.info("[EditPlanGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="EditPlanGuard",
        )
