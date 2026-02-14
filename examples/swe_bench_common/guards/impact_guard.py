"""ImpactGuard: Validates impact analysis output.

Ensures the impact analysis has valid JSON matching the ImpactAnalysis
schema with non-empty required fields.
"""

import json
import logging
from typing import Any

from examples.swe_bench_common.models import ImpactAnalysis

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

logger = logging.getLogger("swe_bench_ablation.guards")


class ImpactGuard(GuardInterface):
    """Validates impact analysis output.

    Checks:
    - Valid JSON matching ImpactAnalysis schema
    - Non-empty reasoning
    - Valid risk_level
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
        """Validate the impact analysis artifact.

        Args:
            artifact: The impact analysis artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info("[ImpactGuard] Validating artifact %s...", artifact.artifact_id[:8])

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="ImpactGuard",
            )

        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {data['error']}",
                guard_name="ImpactGuard",
            )

        try:
            impact = ImpactAnalysis.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="ImpactGuard",
            )

        errors: list[str] = []

        if not impact.reasoning.strip():
            errors.append("reasoning is empty")

        if errors:
            feedback = "Impact analysis validation failed:\n- " + "\n- ".join(errors)
            logger.info("[ImpactGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="ImpactGuard",
            )

        feedback = (
            f"Impact analysis valid: risk={impact.risk_level}, "
            f"{len(impact.affected_tests)} tests, "
            f"{len(impact.potential_regressions)} regressions"
        )
        logger.info("[ImpactGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="ImpactGuard",
        )
