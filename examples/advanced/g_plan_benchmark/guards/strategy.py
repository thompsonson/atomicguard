"""
StrategyGuard: Validates strategy selection output from g_strategy.

Predicates validated:
    G_strategy(s) = parseable_json(s) ^ valid_strategy_id(s)
                     ^ name_present(s) ^ rationale_present(s)
                     ^ key_steps_nonempty(s) ^ expected_guards_nonempty(s)

Complexity: O(1) — fixed number of field checks.
"""

from __future__ import annotations

import json
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

# Strategy vocabulary — maps problem types to resolution approaches.
VALID_STRATEGY_IDS = {
    "S1_locate_and_fix",       # Bug fix: locate → characterization test → fix → regression
    "S2_tdd_feature",          # Feature: design → test first → implement → validate
    "S3_refactor_safely",      # Refactoring: ensure coverage → extract → verify
    "S4_profile_and_optimize", # Performance: profile → benchmark → optimize → verify
    "S5_investigate_first",    # Unknown/complex: investigate → classify → act
}


class StrategyGuard(GuardInterface):
    """
    Validates strategy selection JSON produced by LLMJsonGenerator.

    Expected schema:
    {
        "strategy_id": "S1_locate_and_fix" | "S2_tdd_feature" | ... ,
        "strategy_name": "human-readable name",
        "rationale": "why this strategy fits the problem",
        "key_steps": ["ordered list of high-level steps"],
        "expected_guards": ["guards that should appear in the plan"],
        "risk_factors": ["potential issues with this approach"]
    }

    All checks are O(1).
    """

    def __init__(self, **_kwargs: Any):
        pass

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        """Validate strategy selection artifact."""
        errors: list[str] = []

        # 1. Parseable JSON
        try:
            data = json.loads(artifact.content)
        except (json.JSONDecodeError, TypeError) as e:
            return GuardResult(
                passed=False,
                feedback=f"Strategy not parseable as JSON: {e}",
                guard_name="StrategyGuard",
            )

        if not isinstance(data, dict):
            return GuardResult(
                passed=False,
                feedback=f"Strategy must be a JSON object, got {type(data).__name__}",
                guard_name="StrategyGuard",
            )

        # 2. strategy_id is valid
        strategy_id = data.get("strategy_id")
        if strategy_id is None:
            errors.append("Missing required field: strategy_id")
        elif strategy_id not in VALID_STRATEGY_IDS:
            errors.append(
                f"Invalid strategy_id '{strategy_id}'. "
                f"Must be one of: {', '.join(sorted(VALID_STRATEGY_IDS))}"
            )

        # 3. strategy_name is present and non-empty
        strategy_name = data.get("strategy_name")
        if strategy_name is None:
            errors.append("Missing required field: strategy_name")
        elif not isinstance(strategy_name, str) or not strategy_name.strip():
            errors.append("strategy_name must be a non-empty string")

        # 4. rationale is present and non-empty
        rationale = data.get("rationale")
        if rationale is None:
            errors.append("Missing required field: rationale")
        elif not isinstance(rationale, str) or not rationale.strip():
            errors.append("rationale must be a non-empty string")

        # 5. key_steps is a non-empty list
        key_steps = data.get("key_steps")
        if key_steps is None:
            errors.append("Missing required field: key_steps")
        elif not isinstance(key_steps, list):
            errors.append(
                f"key_steps must be a list, got {type(key_steps).__name__}"
            )
        elif len(key_steps) == 0:
            errors.append("key_steps must be non-empty")

        # 6. expected_guards is a non-empty list
        expected_guards = data.get("expected_guards")
        if expected_guards is None:
            errors.append("Missing required field: expected_guards")
        elif not isinstance(expected_guards, list):
            errors.append(
                f"expected_guards must be a list, got {type(expected_guards).__name__}"
            )
        elif len(expected_guards) == 0:
            errors.append("expected_guards must be non-empty")

        # 7. risk_factors is present and is a list (may be empty)
        risk_factors = data.get("risk_factors")
        if risk_factors is None:
            errors.append("Missing required field: risk_factors")
        elif not isinstance(risk_factors, list):
            errors.append(
                f"risk_factors must be a list, got {type(risk_factors).__name__}"
            )

        if errors:
            return GuardResult(
                passed=False,
                feedback="Strategy validation failed:\n- " + "\n- ".join(errors),
                guard_name="StrategyGuard",
            )

        return GuardResult(
            passed=True,
            feedback=(
                f"Strategy valid: {strategy_id} — {strategy_name}, "
                f"{len(key_steps)} steps, {len(expected_guards)} guards"
            ),
            guard_name="StrategyGuard",
        )
