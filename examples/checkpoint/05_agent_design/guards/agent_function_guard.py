"""
AgentFunctionGuard: Validates agent function specification.

Validates percepts, actions, and percept-action sequences.
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import AgentFunctionSpec

logger = logging.getLogger("agent_design")


class AgentFunctionGuard(GuardInterface):
    """
    Validates agent function specification (Step 3).

    Checks:
    - Valid JSON structure
    - Parses as AgentFunctionSpec schema
    - At least 1 percept
    - At least 1 action
    - At least 1 percept-action sequence
    - Sequences reference valid percepts/actions
    """

    def __init__(
        self,
        min_percepts: int = 1,
        min_actions: int = 1,
        min_sequences: int = 1,
        **_kwargs: Any,
    ):
        self.min_percepts = min_percepts
        self.min_actions = min_actions
        self.min_sequences = min_sequences

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate agent function specification."""
        logger.debug("[AgentFunctionGuard] Validating agent function...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[AgentFunctionGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[AgentFunctionGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        if "error" in data:
            logger.debug(f"[AgentFunctionGuard] Generator error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('error')}",
            )

        # Parse as AgentFunctionSpec
        try:
            func = AgentFunctionSpec.model_validate(data)
            logger.debug("[AgentFunctionGuard] Schema validation passed")
        except Exception as e:
            logger.debug(f"[AgentFunctionGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        issues = []

        # Check minimums
        if len(func.percepts) < self.min_percepts:
            issues.append(
                f"Need at least {self.min_percepts} percept(s), got {len(func.percepts)}"
            )

        if len(func.actions) < self.min_actions:
            issues.append(
                f"Need at least {self.min_actions} action(s), got {len(func.actions)}"
            )

        if len(func.percept_action_sequences) < self.min_sequences:
            issues.append(
                f"Need at least {self.min_sequences} percept-action sequence(s), "
                f"got {len(func.percept_action_sequences)}"
            )

        # Validate references in sequences
        percept_names = {p.name for p in func.percepts}
        action_names = {a.name for a in func.actions}

        for seq in func.percept_action_sequences:
            for percept_ref in seq.percepts:
                if percept_ref not in percept_names:
                    issues.append(
                        f"Sequence '{seq.scenario_name}' references unknown percept '{percept_ref}'"
                    )
            if seq.action not in action_names:
                issues.append(
                    f"Sequence '{seq.scenario_name}' references unknown action '{seq.action}'"
                )

        # Check state representation if model-based
        if not func.state_representation:
            issues.append("State representation is missing (describe internal state)")

        if issues:
            logger.debug(f"[AgentFunctionGuard] Validation failed: {issues}")
            return GuardResult(
                passed=False,
                feedback="Agent function issues:\n- " + "\n- ".join(issues),
            )

        logger.debug("[AgentFunctionGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"Agent function defined: {len(func.percepts)} percepts, "
            f"{len(func.actions)} actions, {len(func.percept_action_sequences)} sequences",
        )
