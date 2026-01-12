"""
EnvironmentPropertiesGuard: Validates environment classification.

Validates that all 6 dimensions are classified with justifications.
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import EnvironmentPropertiesAnalysis

logger = logging.getLogger("agent_design")

REQUIRED_DIMENSIONS = {
    "observable",
    "deterministic",
    "static",
    "discrete",
    "agents",
    "known",
}


class EnvironmentPropertiesGuard(GuardInterface):
    """
    Validates environment properties classification (Step 2).

    Checks:
    - Valid JSON structure
    - Parses as EnvironmentPropertiesAnalysis schema
    - All 6 dimensions classified
    - Each classification has justification
    - Overall complexity assessed
    """

    def __init__(self, required_dimensions: int = 6, **_kwargs: Any):
        self.required_dimensions = required_dimensions

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate environment classification."""
        logger.debug(
            "[EnvironmentPropertiesGuard] Validating environment properties..."
        )

        try:
            data = json.loads(artifact.content)
            logger.debug("[EnvironmentPropertiesGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[EnvironmentPropertiesGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        if "error" in data:
            logger.debug(
                f"[EnvironmentPropertiesGuard] Generator error: {data.get('error')}"
            )
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('error')}",
            )

        # Parse as EnvironmentPropertiesAnalysis
        try:
            env = EnvironmentPropertiesAnalysis.model_validate(data)
            logger.debug("[EnvironmentPropertiesGuard] Schema validation passed")
        except Exception as e:
            logger.debug(f"[EnvironmentPropertiesGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Check all 6 dimensions are covered
        found_dimensions = {prop.dimension for prop in env.properties}
        missing = REQUIRED_DIMENSIONS - found_dimensions

        issues = []

        if missing:
            issues.append(
                f"Missing dimension classifications: {', '.join(sorted(missing))}"
            )

        if len(env.properties) != 6:
            issues.append(f"Expected exactly 6 properties, got {len(env.properties)}")

        # Check each has justification
        for prop in env.properties:
            if not prop.justification or len(prop.justification.strip()) < 10:
                issues.append(
                    f"Dimension '{prop.dimension}' needs better justification (at least 10 chars)"
                )

        # Check overall complexity is set
        if not env.overall_complexity:
            issues.append("Overall complexity assessment missing")

        if issues:
            logger.debug(f"[EnvironmentPropertiesGuard] Validation failed: {issues}")
            return GuardResult(
                passed=False,
                feedback="Environment classification issues:\n- " + "\n- ".join(issues),
            )

        logger.debug("[EnvironmentPropertiesGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"All 6 dimensions classified. Overall complexity: {env.overall_complexity}",
        )
