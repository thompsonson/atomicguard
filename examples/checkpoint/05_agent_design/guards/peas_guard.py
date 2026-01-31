"""
PEASGuard: Validates PEAS analysis completeness.

Validates that the PEASGenerator produced valid PEASAnalysis.
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import PEASAnalysis

logger = logging.getLogger("agent_design")


class PEASGuard(GuardInterface):
    """
    Validates PEAS analysis completeness (Step 1).

    Checks:
    - Valid JSON structure
    - Parses as PEASAnalysis schema
    - At least 1 performance measure
    - At least 1 environment element
    - At least 1 actuator
    - At least 1 sensor
    - Summary is non-empty
    """

    def __init__(
        self,
        min_performance_measures: int = 1,
        min_environment_elements: int = 1,
        min_actuators: int = 1,
        min_sensors: int = 1,
        **_kwargs: Any,
    ):
        self.min_performance = min_performance_measures
        self.min_environment = min_environment_elements
        self.min_actuators = min_actuators
        self.min_sensors = min_sensors

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate PEAS analysis completeness."""
        logger.debug("[PEASGuard] Validating PEAS analysis...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[PEASGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[PEASGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        # Check for errors from generator
        if "error" in data:
            logger.debug(f"[PEASGuard] Generator returned error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('error')}",
            )

        # Parse as PEASAnalysis
        try:
            peas = PEASAnalysis.model_validate(data)
            logger.debug("[PEASGuard] Schema validation passed")
        except Exception as e:
            logger.debug(f"[PEASGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Check completeness
        issues = []

        if len(peas.performance_measures) < self.min_performance:
            issues.append(
                f"Need at least {self.min_performance} performance measure(s), "
                f"got {len(peas.performance_measures)}"
            )

        if len(peas.environment_elements) < self.min_environment:
            issues.append(
                f"Need at least {self.min_environment} environment element(s), "
                f"got {len(peas.environment_elements)}"
            )

        if len(peas.actuators) < self.min_actuators:
            issues.append(
                f"Need at least {self.min_actuators} actuator(s), "
                f"got {len(peas.actuators)}"
            )

        if len(peas.sensors) < self.min_sensors:
            issues.append(
                f"Need at least {self.min_sensors} sensor(s), "
                f"got {len(peas.sensors)}"
            )

        if not peas.summary or len(peas.summary.strip()) < 10:
            issues.append(
                "Summary must be non-empty and meaningful (at least 10 chars)"
            )

        if issues:
            logger.debug(f"[PEASGuard] Validation failed: {issues}")
            return GuardResult(
                passed=False,
                feedback="PEAS analysis incomplete:\n- " + "\n- ".join(issues),
            )

        logger.debug("[PEASGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"PEAS analysis complete: {len(peas.performance_measures)} performance measures, "
            f"{len(peas.environment_elements)} environment elements, "
            f"{len(peas.actuators)} actuators, {len(peas.sensors)} sensors",
        )
