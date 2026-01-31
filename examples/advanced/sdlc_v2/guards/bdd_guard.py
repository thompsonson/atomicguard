"""
BDDGuard: Validates generated BDD scenarios.

Validates that BDDGenerator produced valid Gherkin scenarios.
"""

import json
import logging
import re
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import BDDScenariosResult

logger = logging.getLogger("sdlc_checkpoint")


class BDDGuard(GuardInterface):
    """
    Validates generated BDD scenarios.

    Checks:
    - Valid JSON structure matching BDDScenariosResult schema
    - At least min_scenarios scenarios generated
    - Each scenario has valid Gherkin syntax (Given/When/Then)
    - Feature name is non-empty
    """

    def __init__(self, min_scenarios: int = 1):
        self._min_scenarios = min_scenarios

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate generated BDD scenarios."""
        logger.debug("[BDDGuard] Validating scenarios...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[BDDGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[BDDGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        # Check for errors
        if "error" in data:
            logger.debug(f"[BDDGuard] Generator error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('details', data['error'])}",
            )

        # Parse as BDDScenariosResult
        try:
            result = BDDScenariosResult.model_validate(data)
            logger.debug(f"[BDDGuard] Schema valid, {len(result.scenarios)} scenarios")
        except Exception as e:
            logger.debug(f"[BDDGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Check feature name
        if not result.feature_name:
            logger.debug("[BDDGuard] Empty feature name")
            return GuardResult(
                passed=False,
                feedback="Feature name is required",
            )

        # Check minimum scenario count
        if len(result.scenarios) < self._min_scenarios:
            logger.debug(
                f"[BDDGuard] Insufficient scenarios: {len(result.scenarios)} < {self._min_scenarios}"
            )
            return GuardResult(
                passed=False,
                feedback=f"Expected at least {self._min_scenarios} scenarios, got {len(result.scenarios)}",
            )

        # Validate Gherkin syntax for each scenario
        for scenario in result.scenarios:
            gherkin_result = self._validate_gherkin(scenario.name, scenario.gherkin)
            if gherkin_result:
                return gherkin_result

        logger.debug("[BDDGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"Valid BDD scenarios: {len(result.scenarios)} scenarios for '{result.feature_name}'",
        )

    def _validate_gherkin(self, scenario_name: str, gherkin: str) -> GuardResult | None:
        """Validate Gherkin syntax of a scenario."""
        # Check for required keywords
        has_given = bool(re.search(r"\bGiven\b", gherkin, re.IGNORECASE))
        has_when = bool(re.search(r"\bWhen\b", gherkin, re.IGNORECASE))
        has_then = bool(re.search(r"\bThen\b", gherkin, re.IGNORECASE))

        if not has_given:
            logger.debug(f"[BDDGuard] Scenario '{scenario_name}' missing Given")
            return GuardResult(
                passed=False,
                feedback=f"Scenario '{scenario_name}' must have a 'Given' step",
            )

        if not has_when:
            logger.debug(f"[BDDGuard] Scenario '{scenario_name}' missing When")
            return GuardResult(
                passed=False,
                feedback=f"Scenario '{scenario_name}' must have a 'When' step",
            )

        if not has_then:
            logger.debug(f"[BDDGuard] Scenario '{scenario_name}' missing Then")
            return GuardResult(
                passed=False,
                feedback=f"Scenario '{scenario_name}' must have a 'Then' step",
            )

        return None
