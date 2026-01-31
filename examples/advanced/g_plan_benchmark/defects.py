"""
Defect injection for G_plan benchmark.

Operates on PlanDefinition dicts to inject known defects,
then serializes back to JSON for use as Artifact content.
Each defect type targets a specific predicate in the G_plan taxonomy.
"""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any


class DefectType(Enum):
    """
    Types of plan defects to inject.

    Each maps to specific predicates in the G_plan taxonomy:
    - Structural defects: caught by Minimal (G_min)
    - Semantic defects: caught by Medium (G_med), missed by Minimal
    """

    # Structural (caught by Minimal)
    CYCLE = "cycle"
    MISSING_GUARD = "missing_guard"
    ZERO_RETRY = "zero_retry"

    # Semantic (caught by Medium, missed by Minimal)
    UNREACHABLE_GOAL = "unreachable_goal"
    UNSATISFIABLE_PRECOND = "unsatisfiable_precond"
    MISSING_INITIAL = "missing_initial"
    BUDGET_OVERFLOW = "budget_overflow"
    ORPHAN_STEP = "orphan_step"


def inject_defect(plan_dict: dict[str, Any], defect_type: DefectType) -> dict[str, Any]:
    """
    Inject a specific defect into a plan dict. Returns a modified copy.

    Args:
        plan_dict: Serialized PlanDefinition (from PlanDefinition.to_dict())
        defect_type: Which defect to inject

    Returns:
        Modified plan dict with the specified defect
    """
    plan = copy.deepcopy(plan_dict)
    steps = plan.get("steps", [])

    if defect_type == DefectType.CYCLE:
        # Add circular dependency: first step depends on last
        if len(steps) >= 2:
            steps[0]["dependencies"].append(steps[-1]["step_id"])

    elif defect_type == DefectType.MISSING_GUARD:
        # Replace guard with unknown identifier
        if steps:
            steps[0]["guard"] = "nonexistent_guard_xyz"

    elif defect_type == DefectType.ZERO_RETRY:
        # Set retry budget to 0
        if steps:
            steps[0]["retry_budget"] = 0

    elif defect_type == DefectType.UNREACHABLE_GOAL:
        # Goal requires a token no step produces
        plan["goal_state"] = ["impossible_token_xyz"]

    elif defect_type == DefectType.UNSATISFIABLE_PRECOND:
        # Last step requires a token nothing produces
        if steps:
            preconditions = steps[-1].get("preconditions", [])
            preconditions.append("never_produced_token")
            steps[-1]["preconditions"] = preconditions

    elif defect_type == DefectType.MISSING_INITIAL:
        # Remove initial state
        plan["initial_state"] = []

    elif defect_type == DefectType.BUDGET_OVERFLOW:
        # Step budgets exceed total
        for step in steps:
            step["retry_budget"] = 100
        plan["total_retry_budget"] = 10

    elif defect_type == DefectType.ORPHAN_STEP:
        # Add a step that produces nothing toward the goal
        steps.append(
            {
                "step_id": "orphan",
                "name": "Orphan Step",
                "generator": "IdentityGenerator",
                "guard": "syntax",
                "retry_budget": 1,
                "preconditions": [],
                "effects": ["orphan_token"],
                "dependencies": [],
            }
        )

    return plan
