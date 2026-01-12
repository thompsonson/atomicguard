"""
AgentTypeGuard: Validates agent type selection.

Validates that the selected agent type is justified.
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import AgentTypeAnalysis

logger = logging.getLogger("agent_design")

VALID_AGENT_TYPES = {
    "simple_reflex",
    "model_based_reflex",
    "goal_based",
    "utility_based",
    "learning",
}


class AgentTypeGuard(GuardInterface):
    """
    Validates agent type selection (Step 4).

    Checks:
    - Valid JSON structure
    - Parses as AgentTypeAnalysis schema
    - Selected type is valid
    - Justification is provided
    - At least 1 alternative considered
    - Required capabilities are identified
    """

    def __init__(self, min_alternatives: int = 1, **_kwargs: Any):
        self.min_alternatives = min_alternatives

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate agent type selection."""
        logger.debug("[AgentTypeGuard] Validating agent type...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[AgentTypeGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[AgentTypeGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        if "error" in data:
            logger.debug(f"[AgentTypeGuard] Generator error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('error')}",
            )

        # Parse as AgentTypeAnalysis
        try:
            type_analysis = AgentTypeAnalysis.model_validate(data)
            logger.debug("[AgentTypeGuard] Schema validation passed")
        except Exception as e:
            logger.debug(f"[AgentTypeGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        issues = []

        # Check valid type
        if type_analysis.selected_type not in VALID_AGENT_TYPES:
            issues.append(
                f"Invalid agent type '{type_analysis.selected_type}'. "
                f"Valid types: {', '.join(sorted(VALID_AGENT_TYPES))}"
            )

        # Check justification
        if (
            not type_analysis.justification
            or len(type_analysis.justification.strip()) < 20
        ):
            issues.append(
                "Justification too short (need at least 20 chars explaining the choice)"
            )

        # Check alternatives considered
        if len(type_analysis.alternatives_considered) < self.min_alternatives:
            issues.append(
                f"Need at least {self.min_alternatives} alternative(s) considered, "
                f"got {len(type_analysis.alternatives_considered)}"
            )

        # Check rejection reasons exist for alternatives
        for alt in type_analysis.alternatives_considered:
            if alt not in type_analysis.rejection_reasons:
                issues.append(f"Missing rejection reason for alternative '{alt}'")

        # Check required capabilities
        if not type_analysis.required_capabilities:
            issues.append("Required capabilities list is empty")

        if issues:
            logger.debug(f"[AgentTypeGuard] Validation failed: {issues}")
            return GuardResult(
                passed=False,
                feedback="Agent type selection issues:\n- " + "\n- ".join(issues),
            )

        logger.debug("[AgentTypeGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"Agent type selected: {type_analysis.selected_type}. "
            f"Considered {len(type_analysis.alternatives_considered)} alternatives.",
        )
