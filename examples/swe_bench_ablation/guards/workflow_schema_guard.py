"""WorkflowSchemaGuard: Validates generated workflow specifications.

G_val guard for ap_generate_workflow. Checks that the generated workflow
is valid JSON matching the GeneratedWorkflow schema, all referenced
generators/guards exist in the component registry, dependencies form
an acyclic graph, and total budget is within limits.

Used by: Arms 20, 21
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import GeneratedWorkflow

logger = logging.getLogger("swe_bench_ablation.guards")

# Default component registry — must match what's available in demo.py
DEFAULT_GENERATORS = {
    "AnalysisGenerator",
    "LocalizationGenerator",
    "PatchGenerator",
    "TestGenerator",
    "DiffReviewGenerator",
    "ClassificationGenerator",
    "WorkflowGenerator",
}

DEFAULT_GUARDS = {
    "analysis",
    "localization",
    "patch",
    "test_syntax",
    "test_red",
    "test_green",
    "full_eval",
    "review_schema",
    "classification_schema",
    "composite_test_verified",
    "composite_patch_verified",
}

MAX_BUDGET = 50  # Maximum total LLM calls (rmax * num_action_pairs)


class WorkflowSchemaGuard(GuardInterface):
    """Validates generated workflow specifications.

    Checks:
    - Valid JSON matching GeneratedWorkflow schema
    - All generators reference registered components
    - All guards reference registered components
    - Dependencies are acyclic
    - Total budget (rmax * steps) is within B_max
    - At least one action pair is defined
    """

    def __init__(
        self,
        valid_generators: set[str] | None = None,
        valid_guards: set[str] | None = None,
        max_budget: int = MAX_BUDGET,
        **kwargs: Any,  # noqa: ARG002
    ):
        self._valid_generators = valid_generators or DEFAULT_GENERATORS
        self._valid_guards = valid_guards or DEFAULT_GUARDS
        self._max_budget = max_budget

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the generated workflow artifact."""
        logger.info(
            "[WorkflowSchemaGuard] Validating artifact %s...",
            artifact.artifact_id[:8],
        )

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="WorkflowSchemaGuard",
            )

        try:
            workflow = GeneratedWorkflow.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="WorkflowSchemaGuard",
            )

        errors: list[str] = []

        # Check at least one action pair
        if not workflow.action_pairs:
            errors.append("No action pairs defined")

        # Validate generators
        for ap_id, ap_spec in workflow.action_pairs.items():
            if ap_spec.generator not in self._valid_generators:
                errors.append(
                    f"{ap_id}: unknown generator '{ap_spec.generator}'. "
                    f"Available: {', '.join(sorted(self._valid_generators))}"
                )

        # Validate guards
        for ap_id, ap_spec in workflow.action_pairs.items():
            if ap_spec.guard not in self._valid_guards:
                errors.append(
                    f"{ap_id}: unknown guard '{ap_spec.guard}'. "
                    f"Available: {', '.join(sorted(self._valid_guards))}"
                )

        # Validate dependencies exist
        ap_ids = set(workflow.action_pairs.keys())
        for ap_id, ap_spec in workflow.action_pairs.items():
            for dep in ap_spec.requires:
                if dep not in ap_ids:
                    errors.append(
                        f"{ap_id}: dependency '{dep}' not found in action_pairs"
                    )

        # Check for cycles
        cycle = self._detect_cycle(workflow.action_pairs)
        if cycle:
            errors.append(f"Dependency cycle detected: {' -> '.join(cycle)}")

        # Check budget
        total_budget = workflow.rmax * len(workflow.action_pairs)
        if total_budget > self._max_budget:
            errors.append(
                f"Total budget ({total_budget} = rmax {workflow.rmax} x "
                f"{len(workflow.action_pairs)} steps) exceeds maximum ({self._max_budget})"
            )

        if errors:
            feedback = "Workflow validation failed:\n- " + "\n- ".join(errors)
            logger.info("[WorkflowSchemaGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="WorkflowSchemaGuard",
            )

        feedback = (
            f"Workflow valid: '{workflow.name}', "
            f"{len(workflow.action_pairs)} steps, rmax={workflow.rmax}"
        )
        logger.info("[WorkflowSchemaGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="WorkflowSchemaGuard",
        )

    @staticmethod
    def _detect_cycle(
        action_pairs: dict[str, Any],
    ) -> list[str] | None:
        """Detect cycles in action pair dependencies using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {ap_id: WHITE for ap_id in action_pairs}
        path: list[str] = []

        def dfs(node: str) -> list[str] | None:
            color[node] = GRAY
            path.append(node)
            ap_spec = action_pairs[node]
            requires = (
                ap_spec.requires
                if isinstance(ap_spec.requires, list)
                else getattr(ap_spec, "requires", [])
            )
            for dep in requires:
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    cycle_start = path.index(dep)
                    return path[cycle_start:] + [dep]
                if color[dep] == WHITE:
                    result = dfs(dep)
                    if result:
                        return result
            path.pop()
            color[node] = BLACK
            return None

        for ap_id in action_pairs:
            if color[ap_id] == WHITE:
                result = dfs(ap_id)
                if result:
                    return result
        return None
