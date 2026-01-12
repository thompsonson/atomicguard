"""
ATDDGuard: Validates acceptance criteria.

Validates Given-When-Then scenarios and 10 Principles compliance.
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import AcceptanceCriteria

logger = logging.getLogger("agent_design")


class ATDDGuard(GuardInterface):
    """
    Validates acceptance criteria (Step 5).

    Checks:
    - Valid JSON structure
    - Parses as AcceptanceCriteria schema
    - Minimum scenario count
    - Given-When-Then structure
    - References to valid percepts/actions
    - 10 Principles compliance (optional)
    """

    def __init__(
        self,
        min_scenarios: int = 3,
        validate_10_principles: bool = True,
        **_kwargs: Any,
    ):
        self.min_scenarios = min_scenarios
        self.validate_10_principles = validate_10_principles

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate acceptance criteria."""
        logger.debug("[ATDDGuard] Validating acceptance criteria...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[ATDDGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[ATDDGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        if "error" in data:
            logger.debug(f"[ATDDGuard] Generator error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('error')}",
            )

        # Parse as AcceptanceCriteria
        try:
            criteria = AcceptanceCriteria.model_validate(data)
            logger.debug("[ATDDGuard] Schema validation passed")
        except Exception as e:
            logger.debug(f"[ATDDGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        issues = []

        # Check minimum scenarios
        if len(criteria.scenarios) < self.min_scenarios:
            issues.append(
                f"Need at least {self.min_scenarios} scenario(s), got {len(criteria.scenarios)}"
            )

        # Validate each scenario
        for scenario in criteria.scenarios:
            scenario_issues = self._validate_scenario(scenario)
            issues.extend(scenario_issues)

        # Validate 10 Principles if enabled
        if self.validate_10_principles:
            principle_issues = self._validate_10_principles(criteria)
            issues.extend(principle_issues)

        if issues:
            logger.debug(f"[ATDDGuard] Validation failed: {issues}")
            return GuardResult(
                passed=False,
                feedback="Acceptance criteria issues:\n- " + "\n- ".join(issues),
            )

        logger.debug("[ATDDGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"Acceptance criteria valid: {len(criteria.scenarios)} scenarios defined",
        )

    def _validate_scenario(self, scenario: Any) -> list[str]:
        """Validate a single scenario."""
        issues = []

        if not scenario.given:
            issues.append(
                f"Scenario '{scenario.scenario_id}': 'Given' section is empty"
            )

        if not scenario.when:
            issues.append(f"Scenario '{scenario.scenario_id}': 'When' section is empty")

        if not scenario.then:
            issues.append(f"Scenario '{scenario.scenario_id}': 'Then' section is empty")

        if not scenario.percept_refs:
            issues.append(
                f"Scenario '{scenario.scenario_id}': No percept references. "
                "Link to percepts from agent function."
            )

        if not scenario.action_refs:
            issues.append(
                f"Scenario '{scenario.scenario_id}': No action references. "
                "Link to actions from agent function."
            )

        return issues

    def _validate_10_principles(self, criteria: AcceptanceCriteria) -> list[str]:
        """Validate compliance with 10 Principles for Acceptance Criteria."""
        issues = []

        # Check that scenarios test observable behaviors (Principle 2)
        for scenario in criteria.scenarios:
            has_observable_then = any(
                "observable" in step.lower() or "should" in step.lower()
                for step in scenario.then
            )
            if not has_observable_then:
                issues.append(
                    f"Scenario '{scenario.scenario_id}': 'Then' steps should describe "
                    "observable behaviors (Principle 2: Test observable behaviors)"
                )

        # Check full percept-action cycles covered (Principle 8)
        if not criteria.coverage_summary:
            issues.append(
                "Missing coverage_summary - describe which percept-action cycles "
                "are covered (Principle 8: Cover full percept-action cycles)"
            )

        # Principle compliance metadata (soft check)
        scenarios_with_compliance = sum(
            1 for s in criteria.scenarios if s.principle_compliance
        )
        if scenarios_with_compliance < len(criteria.scenarios) // 2:
            issues.append(
                "Most scenarios should include principle_compliance metadata "
                "to confirm adherence to 10 Principles"
            )

        return issues
