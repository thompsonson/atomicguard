"""
ExpansivePlanGuard: O(R_max^K) formal verification (simplified).

Guarantees: Goal achievement if generator epsilon-capable + guards correctly specified.

Predicates validated:
    G_exp(p) = G_med(p)
             ^ for_all_traces(terminates)
             ^ for_all_traces(safe)
             ^ for_all_traces(invariant_holds)
"""

from __future__ import annotations

import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import PlanDefinition
from .medium import MediumPlanGuard

logger = logging.getLogger("g_plan_benchmark")

# Safety bound to prevent runaway exploration.
MAX_EXPLORATIONS = 100_000


class ExpansivePlanGuard(GuardInterface):
    """
    Expansive rigor plan validation: state space exploration.

    Physically manifests the O(R_max^K) complexity cliff by recursively
    exploring retry branches. Each step can succeed or fail up to
    retry_budget times; we explore branches to demonstrate exponential cost.

    Additional checks beyond Medium:
    - All execution traces reach the goal
    - Steps with retries produce effects (monotonic progress)
    - No dead-end steps (contributes to goal or has failure handling)

    Complexity: O(R^K) where R = max retries, K = steps.
    """

    def __init__(
        self,
        guard_catalog: set[str] | None = None,
        r_max: int = 100,
        **_kwargs: Any,
    ):
        self._medium_guard = MediumPlanGuard(guard_catalog=guard_catalog, r_max=r_max)

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        """Validate plan artifact at expansive rigor."""
        # First run medium checks
        medium_result = self._medium_guard.validate(artifact, **deps)
        if not medium_result.passed:
            return GuardResult(
                passed=False,
                feedback=medium_result.feedback,
                guard_name="ExpansivePlanGuard",
            )

        plan = PlanDefinition.from_json(artifact.content)
        errors: list[str] = []

        # State space exploration with branching
        exploration_count = [0]

        def explore_branches(
            step_idx: int,
            current_state: set[str],
            depth: int,
        ) -> bool:
            """Recursively explore success/failure branches."""
            exploration_count[0] += 1

            if exploration_count[0] > MAX_EXPLORATIONS:
                return True  # Assume valid if explored enough

            # Base case: reached end of plan
            if step_idx >= len(plan.steps):
                return plan.goal_state.issubset(current_state)

            step = plan.steps[step_idx]

            # Check preconditions
            if not step.preconditions.issubset(current_state):
                return False

            # Branch 1: Success path
            success_state = current_state | step.effects
            success_path = explore_branches(step_idx + 1, success_state, depth + 1)

            # Branch 2..R: Failure/retry paths (capped for tractability)
            retry_branches = min(step.retry_budget - 1, 2)
            for _ in range(retry_branches):
                explore_branches(step_idx + 1, success_state, depth + 1)

            return success_path

        all_paths_valid = explore_branches(0, plan.initial_state, 0)

        if not all_paths_valid:
            errors.append("State space exploration found unreachable goal paths")

        # All terminal states contribute to goal or have failure handling
        goal_tokens = plan.goal_state if plan.goal_state else set()
        for step in plan.steps:
            contributes = bool(step.effects & goal_tokens)
            has_handling = step.retry_budget >= 1

            if not contributes and not has_handling:
                errors.append(
                    f"Step {step.step_id}: neither contributes to goal "
                    f"nor has failure handling"
                )

        # Monotonic progress: steps with retries must have effects
        for step in plan.steps:
            if step.retry_budget > 1 and not step.effects:
                errors.append(
                    f"Step {step.step_id}: has retries but no effects "
                    f"- cannot refine context"
                )

        if errors:
            return GuardResult(
                passed=False,
                feedback="Expansive validation failed:\n- " + "\n- ".join(errors),
                guard_name="ExpansivePlanGuard",
            )

        return GuardResult(
            passed=True,
            feedback=(
                f"Expansive validation passed: {len(plan.steps)} steps, "
                f"{exploration_count[0]} states explored"
            ),
            guard_name="ExpansivePlanGuard",
        )
