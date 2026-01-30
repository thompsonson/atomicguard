"""
MinimalPlanGuard: O(V + E) structural validity.

Guarantees: Plan won't crash on execution start.

Predicates validated:
    G_min(p) = parseable(p) ^ is_dag(p) ^ guard_exists(p) ^ budget_defined(p)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import PlanDefinition

logger = logging.getLogger("g_plan_benchmark")

# Known guard identifiers from the AtomicGuard ecosystem.
# Extensible via guard_catalog constructor arg.
DEFAULT_GUARD_CATALOG: set[str] = {
    # Base guards
    "syntax",
    "import",
    "human",
    "dynamic_test",
    "composite",
    # sdlc_v2 guards
    "config_extracted",
    "architecture_tests_valid",
    "scenarios_valid",
    "rules_valid",
    "all_tests_pass",
    "quality_gates",
    "arch_validation",
    "merge_ready",
    "composite_validation",
    # G_plan guards (self-referential)
    "plan_minimal",
    "plan_medium",
    "plan_expansive",
}


class MinimalPlanGuard(GuardInterface):
    """
    Minimal rigor plan validation: structural validity only.

    Checks:
    - parseable: valid JSON, required fields present
    - is_dag: no cycles in step dependencies (Kahn's algorithm)
    - guard_exists: all guard references resolve to known catalog
    - budget_defined: all steps have retry_budget > 0

    Complexity: O(V + E) - linear scan of plan structure.
    """

    def __init__(
        self,
        guard_catalog: set[str] | None = None,
        **_kwargs: Any,
    ):
        self._guard_catalog = guard_catalog or DEFAULT_GUARD_CATALOG

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        """Validate plan artifact at minimal rigor."""
        errors: list[str] = []

        # 1. Parseable: valid JSON with required structure
        try:
            plan = PlanDefinition.from_json(artifact.content)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return GuardResult(
                passed=False,
                feedback=f"Plan not parseable: {e}",
                guard_name="MinimalPlanGuard",
            )

        if not plan.plan_id:
            errors.append("Missing plan_id")
        if not plan.steps:
            errors.append("No steps defined")

        for step in plan.steps:
            if not step.step_id:
                errors.append(f"Step missing step_id: {step.name}")
            if not step.name:
                errors.append(f"Step {step.step_id} missing name")

        # 2. Is DAG: cycle detection via Kahn's algorithm
        if plan.steps:
            cycle_error = _detect_cycles(plan)
            if cycle_error:
                errors.append(cycle_error)

        # 3. Guard exists: all guard references in catalog
        for step in plan.steps:
            if step.guard and step.guard not in self._guard_catalog:
                errors.append(
                    f"Step {step.step_id}: unknown guard '{step.guard}'"
                )

        # 4. Budget defined: retry_budget > 0 for all steps
        for step in plan.steps:
            if step.retry_budget <= 0:
                errors.append(
                    f"Step {step.step_id}: retry_budget must be > 0, "
                    f"got {step.retry_budget}"
                )

        if errors:
            return GuardResult(
                passed=False,
                feedback="Minimal validation failed:\n- " + "\n- ".join(errors),
                guard_name="MinimalPlanGuard",
            )

        return GuardResult(
            passed=True,
            feedback=f"Minimal validation passed: {len(plan.steps)} steps, DAG valid",
            guard_name="MinimalPlanGuard",
        )


def _detect_cycles(plan: PlanDefinition) -> str | None:
    """Kahn's algorithm for topological sort / cycle detection."""
    in_degree: dict[str, int] = {s.step_id: 0 for s in plan.steps}
    adjacency: dict[str, list[str]] = defaultdict(list)
    step_ids = {s.step_id for s in plan.steps}

    for step in plan.steps:
        for dep in step.dependencies:
            if dep not in step_ids:
                return f"Step {step.step_id}: dependency '{dep}' not found"
            adjacency[dep].append(step.step_id)
            in_degree[step.step_id] += 1

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    processed = 0

    while queue:
        current = queue.pop(0)
        processed += 1
        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if processed != len(plan.steps):
        return "Cycle detected in step dependencies"
    return None
