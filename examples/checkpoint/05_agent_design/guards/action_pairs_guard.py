"""
ActionPairsGuard: Validates Action Pair specifications.

Validates that each action pair has complete ρ, a_gen, and G.
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import ActionPairsDesign

logger = logging.getLogger("agent_design")


class ActionPairsGuard(GuardInterface):
    """
    Validates Action Pair specifications (Step 6).

    Checks:
    - Valid JSON structure
    - Parses as ActionPairsDesign schema
    - Minimum action pair count
    - Each action pair has complete ρ, a_gen, G
    - Traceability to acceptance criteria
    - Valid workflow order and dependencies
    """

    def __init__(
        self,
        min_action_pairs: int = 1,
        require_traceability: bool = True,
        **_kwargs: Any,
    ):
        self.min_action_pairs = min_action_pairs
        self.require_traceability = require_traceability

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate Action Pair specifications."""
        logger.debug("[ActionPairsGuard] Validating action pairs...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[ActionPairsGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[ActionPairsGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        if "error" in data:
            logger.debug(f"[ActionPairsGuard] Generator error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('error')}",
            )

        # Parse as ActionPairsDesign
        try:
            design = ActionPairsDesign.model_validate(data)
            logger.debug("[ActionPairsGuard] Schema validation passed")
        except Exception as e:
            logger.debug(f"[ActionPairsGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        issues = []

        # Check minimum action pairs
        if len(design.action_pairs) < self.min_action_pairs:
            issues.append(
                f"Need at least {self.min_action_pairs} action pair(s), "
                f"got {len(design.action_pairs)}"
            )

        # Validate each action pair
        action_pair_ids = set()
        for ap in design.action_pairs:
            ap_issues = self._validate_action_pair(ap)
            issues.extend(ap_issues)
            action_pair_ids.add(ap.action_pair_id)

        # Validate workflow order
        if design.workflow_order:
            for step_id in design.workflow_order:
                if step_id not in action_pair_ids:
                    issues.append(
                        f"Workflow order references unknown action pair '{step_id}'"
                    )

        # Validate dependencies
        if design.dependencies:
            for ap_id, deps_list in design.dependencies.items():
                if ap_id not in action_pair_ids:
                    issues.append(
                        f"Dependencies reference unknown action pair '{ap_id}'"
                    )
                for dep_id in deps_list:
                    if dep_id not in action_pair_ids:
                        issues.append(
                            f"Action pair '{ap_id}' depends on unknown '{dep_id}'"
                        )

        if issues:
            logger.debug(f"[ActionPairsGuard] Validation failed: {issues}")
            return GuardResult(
                passed=False,
                feedback="Action pair design issues:\n- " + "\n- ".join(issues),
            )

        logger.debug("[ActionPairsGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"Action pairs valid: {len(design.action_pairs)} action pairs defined",
        )

    def _validate_action_pair(self, ap: Any) -> list[str]:
        """Validate a single action pair has complete ρ, a_gen, G."""
        issues = []
        prefix = f"Action pair '{ap.action_pair_id}'"

        # ρ (rho) - Precondition
        if not ap.precondition:
            issues.append(f"{prefix}: Missing precondition (ρ)")
        if not ap.precondition_percepts:
            issues.append(f"{prefix}: No precondition percepts specified")

        # a_gen - Generator
        if not ap.generator_name:
            issues.append(f"{prefix}: Missing generator name (a_gen)")
        if not ap.generator_description:
            issues.append(f"{prefix}: Missing generator description")
        if not ap.generator_inputs:
            issues.append(f"{prefix}: No generator inputs specified")
        if not ap.generator_output_schema:
            issues.append(f"{prefix}: Missing generator output schema")

        # G - Guard
        if not ap.guard_name:
            issues.append(f"{prefix}: Missing guard name (G)")
        if not ap.guard_description:
            issues.append(f"{prefix}: Missing guard description")
        if not ap.guard_checks:
            issues.append(f"{prefix}: No guard checks specified")

        # Traceability
        if self.require_traceability and not ap.acceptance_criteria_refs:
            issues.append(
                f"{prefix}: No acceptance criteria references. "
                "Link to AC scenarios for traceability."
            )

        return issues
