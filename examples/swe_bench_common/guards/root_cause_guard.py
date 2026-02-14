"""RootCauseGuard: Validates root cause analysis output.

Ensures the root cause analysis has valid JSON matching the RootCause
schema with non-empty required fields.
"""

import json
import logging
from typing import Any

from examples.swe_bench_common.models import RootCause

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

logger = logging.getLogger("swe_bench_ablation.guards")


class RootCauseGuard(GuardInterface):
    """Validates root cause analysis output.

    Checks:
    - Valid JSON matching RootCause schema
    - Non-empty cause_type and cause_description
    - At least one triggering condition
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
        """Validate the root cause artifact.

        Args:
            artifact: The root cause artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info(
            "[RootCauseGuard] Validating artifact %s...", artifact.artifact_id[:8]
        )

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="RootCauseGuard",
            )

        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {data['error']}",
                guard_name="RootCauseGuard",
            )

        try:
            root_cause = RootCause.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="RootCauseGuard",
            )

        errors: list[str] = []

        if not root_cause.cause_type.strip():
            errors.append("cause_type is empty")

        if not root_cause.cause_description.strip():
            errors.append("cause_description is empty")

        if not root_cause.triggering_conditions:
            errors.append(
                "No triggering conditions identified. "
                "Must identify at least one condition that triggers the bug."
            )

        if errors:
            feedback = "Root cause validation failed:\n- " + "\n- ".join(errors)
            logger.info("[RootCauseGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="RootCauseGuard",
            )

        feedback = (
            f"Root cause valid: type={root_cause.cause_type}, "
            f"{len(root_cause.triggering_conditions)} triggers, "
            f"confidence={root_cause.confidence}"
        )
        logger.info("[RootCauseGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="RootCauseGuard",
        )
