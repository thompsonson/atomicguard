"""ClassificationGuard: Validates problem classification output schema.

G_val guard for ap_classify_problem. Checks that the classification output
is valid JSON matching the ProblemClassification schema with a valid category
and complexity estimate.

Used by: Arms 20, 21
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import ProblemClassification

logger = logging.getLogger("swe_bench_ablation.guards")


class ClassificationGuard(GuardInterface):
    """Validates problem classification output.

    Checks:
    - Valid JSON matching ProblemClassification schema
    - Non-empty reasoning
    - Category is a valid ProblemCategory enum value
    - Complexity is in range 1-5
    """

    def __init__(self, **kwargs: Any):  # noqa: ARG002
        pass

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the classification artifact."""
        logger.info(
            "[ClassificationGuard] Validating artifact %s...",
            artifact.artifact_id[:8],
        )

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="ClassificationGuard",
            )

        try:
            classification = ProblemClassification.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="ClassificationGuard",
            )

        errors: list[str] = []

        if not classification.reasoning.strip():
            errors.append("reasoning is empty")

        if errors:
            feedback = "Classification validation failed:\n- " + "\n- ".join(errors)
            logger.info("[ClassificationGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="ClassificationGuard",
            )

        feedback = (
            f"Classification valid: category={classification.category.value}, "
            f"complexity={classification.estimated_complexity}/5"
        )
        logger.info("[ClassificationGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="ClassificationGuard",
        )
