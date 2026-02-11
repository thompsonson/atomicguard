"""FixApproachGuard: Validates fix approach design output.

Ensures the fix approach has valid JSON matching the FixApproach
schema with non-empty required fields.
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from examples.swe_bench_common.models import FixApproach

logger = logging.getLogger("swe_bench_ablation.guards")


class FixApproachGuard(GuardInterface):
    """Validates fix approach design output.

    Checks:
    - Valid JSON matching FixApproach schema
    - Non-empty approach_summary and reasoning
    - At least one step defined
    - At least one file to modify
    """

    def __init__(
        self,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the guard."""
        pass

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the fix approach artifact.

        Args:
            artifact: The fix approach artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info(
            "[FixApproachGuard] Validating artifact %s...", artifact.artifact_id[:8]
        )

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="FixApproachGuard",
            )

        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {data['error']}",
                guard_name="FixApproachGuard",
            )

        try:
            fix_approach = FixApproach.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="FixApproachGuard",
            )

        errors: list[str] = []

        if not fix_approach.approach_summary.strip():
            errors.append("approach_summary is empty")

        if not fix_approach.reasoning.strip():
            errors.append("reasoning is empty")

        if not fix_approach.steps:
            errors.append(
                "No fix steps defined. Must provide at least one step."
            )

        if not fix_approach.files_to_modify:
            errors.append(
                "No files_to_modify identified. Must specify at least one file."
            )

        if errors:
            feedback = "Fix approach validation failed:\n- " + "\n- ".join(errors)
            logger.info("[FixApproachGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="FixApproachGuard",
            )

        feedback = (
            f"Fix approach valid: {len(fix_approach.steps)} steps, "
            f"{len(fix_approach.files_to_modify)} files, "
            f"{len(fix_approach.edge_cases)} edge cases"
        )
        logger.info("[FixApproachGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="FixApproachGuard",
        )
