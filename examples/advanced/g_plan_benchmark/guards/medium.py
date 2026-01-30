"""
MediumPlanGuard: O(V x L) convergence preservation.

Guarantees: Termination + progress toward goal (if epsilon-capable generator).

Predicates validated:
    G_med(p) = G_min(p)
             ^ reachable(p)
             ^ precond_satisfiable(p)
             ^ path_exists(init, goal)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import PlanDefinition
from .minimal import MinimalPlanGuard

logger = logging.getLogger("g_plan_benchmark")

# Configurable maximum total retry budget.
DEFAULT_R_MAX = 100


class MediumPlanGuard(GuardInterface):
    """
    Medium rigor plan validation: convergence preservation.

    Checks (in addition to Minimal):
    - initial_state_defined
    - goal_state_defined
    - precondition_satisfiable: each step's preconditions met by
      initial_state + prior effects (symbolic intersection)
    - path_exists: goal state reachable via BFS through effects
    - budget_consistent: sum of step budgets <= total_retry_budget

    Complexity: O(V x L) where L = number of literals.
    """

    def __init__(
        self,
        guard_catalog: set[str] | None = None,
        r_max: int = DEFAULT_R_MAX,
        **_kwargs: Any,
    ):
        self._minimal_guard = MinimalPlanGuard(guard_catalog=guard_catalog)
        self._r_max = r_max

    def validate(self, artifact: Artifact, **deps: Artifact) -> GuardResult:
        """Validate plan artifact at medium rigor."""
        # First run minimal checks
        minimal_result = self._minimal_guard.validate(artifact, **deps)
        if not minimal_result.passed:
            return GuardResult(
                passed=False,
                feedback=minimal_result.feedback,
                guard_name="MediumPlanGuard",
            )

        plan = PlanDefinition.from_json(artifact.content)
        errors: list[str] = []

        # 5. Initial state defined
        if not plan.initial_state:
            errors.append("initial_state not defined")

        # 6. Goal state defined
        if not plan.goal_state:
            errors.append("goal_state not defined")

        # 7. Precondition satisfiability: walk topological order,
        #    accumulate available tokens, check each step's preconditions.
        step_order = _topological_sort(plan)

        if step_order:
            reachable_tokens = set(plan.initial_state) if plan.initial_state else set()
            for step_id in step_order:
                step = next(s for s in plan.steps if s.step_id == step_id)
                unsatisfied = step.preconditions - reachable_tokens
                if unsatisfied:
                    errors.append(
                        f"Step {step_id}: preconditions {sorted(unsatisfied)} "
                        f"not satisfiable (available: {sorted(reachable_tokens)})"
                    )
                reachable_tokens.update(step.effects)

        # 8. Path exists: goal reachable from initial state
        if plan.goal_state and step_order:
            final_tokens = set(plan.initial_state) if plan.initial_state else set()
            for step_id in step_order:
                step = next(s for s in plan.steps if s.step_id == step_id)
                final_tokens.update(step.effects)

            unreachable_goals = plan.goal_state - final_tokens
            if unreachable_goals:
                errors.append(
                    f"Goal tokens unreachable: {sorted(unreachable_goals)}"
                )

        # 9. Total retry budget check
        if plan.total_retry_budget > self._r_max:
            errors.append(
                f"total_retry_budget ({plan.total_retry_budget}) "
                f"exceeds R_max ({self._r_max})"
            )

        step_budget_sum = sum(s.retry_budget for s in plan.steps)
        if step_budget_sum > plan.total_retry_budget:
            errors.append(
                f"Sum of step budgets ({step_budget_sum}) "
                f"exceeds total_retry_budget ({plan.total_retry_budget})"
            )

        if errors:
            return GuardResult(
                passed=False,
                feedback="Medium validation failed:\n- " + "\n- ".join(errors),
                guard_name="MediumPlanGuard",
            )

        return GuardResult(
            passed=True,
            feedback=(
                f"Medium validation passed: {len(plan.steps)} steps, "
                f"goal reachable, preconditions satisfiable"
            ),
            guard_name="MediumPlanGuard",
        )


def _topological_sort(plan: PlanDefinition) -> list[str] | None:
    """Return topologically sorted step IDs, or None if cycle exists."""
    in_degree: dict[str, int] = {s.step_id: 0 for s in plan.steps}
    adjacency: dict[str, list[str]] = defaultdict(list)
    step_ids = {s.step_id for s in plan.steps}

    for step in plan.steps:
        for dep in step.dependencies:
            if dep in step_ids:
                adjacency[dep].append(step.step_id)
                in_degree[step.step_id] += 1

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    result: list[str] = []

    while queue:
        current = queue.pop(0)
        result.append(current)
        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == len(plan.steps) else None
