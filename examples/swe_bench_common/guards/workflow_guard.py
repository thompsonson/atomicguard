"""WorkflowGuard: Validates generated workflow output schema.

G_val guard for ap_generate_workflow. Checks that the generated workflow
is valid JSON matching the GeneratedWorkflow schema with valid action pairs
and proper dependency structure.

Used by: Arm 21
"""

import json
import logging
from typing import Any

from examples.swe_bench_common.models import GeneratedWorkflow

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

logger = logging.getLogger("swe_bench_ablation.guards")

# Valid generators and guards that can be referenced in generated workflows
VALID_GENERATORS = {
    "AnalysisGenerator",
    "LocalizationGenerator",
    "PatchGenerator",
    "TestGenerator",
    "DiffReviewGenerator",
    "ClassificationGenerator",
    "WorkflowGenerator",
}

VALID_GUARDS = {
    "analysis",
    "localization",
    "patch",
    "test_syntax",
    "test_red",
    "test_green",
    "full_eval",
    "review_schema",
    "classification_schema",
    "workflow_schema",
    "composite",
}


class WorkflowGuard(GuardInterface):
    """Validates generated workflow output.

    Checks:
    - Valid JSON matching GeneratedWorkflow schema
    - Non-empty action_pairs
    - All generators reference valid generator names
    - All guards reference valid guard names
    - Dependency graph has no cycles
    - All requires references point to existing action pairs
    """

    def __init__(self, **kwargs: Any):  # noqa: ARG002
        pass

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the generated workflow artifact."""
        logger.info(
            "[WorkflowGuard] Validating artifact %s...",
            artifact.artifact_id[:8],
        )

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="WorkflowGuard",
            )

        try:
            workflow = GeneratedWorkflow.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="WorkflowGuard",
            )

        errors: list[str] = []

        # Check that action_pairs is non-empty
        if not workflow.action_pairs:
            errors.append(
                "action_pairs is empty - workflow must have at least one step"
            )

        # Validate generator and guard references
        ap_ids = set(workflow.action_pairs.keys())
        for ap_id, ap_spec in workflow.action_pairs.items():
            # Check generator
            if ap_spec.generator not in VALID_GENERATORS:
                errors.append(
                    f"Action pair '{ap_id}' references unknown generator "
                    f"'{ap_spec.generator}'. Valid generators: "
                    f"{', '.join(sorted(VALID_GENERATORS))}"
                )

            # Check guard
            if ap_spec.guard not in VALID_GUARDS:
                errors.append(
                    f"Action pair '{ap_id}' references unknown guard "
                    f"'{ap_spec.guard}'. Valid guards: "
                    f"{', '.join(sorted(VALID_GUARDS))}"
                )

            # Check requires references
            for req in ap_spec.requires:
                if req not in ap_ids:
                    errors.append(
                        f"Action pair '{ap_id}' requires unknown step '{req}'. "
                        f"Valid step IDs: {', '.join(sorted(ap_ids))}"
                    )

        # Check for cycles in dependency graph
        cycle = self._detect_cycle(workflow.action_pairs)
        if cycle:
            errors.append(f"Dependency cycle detected: {' -> '.join(cycle)}")

        # Check rmax is reasonable
        if workflow.rmax < 1 or workflow.rmax > 10:
            errors.append(f"rmax must be between 1 and 10, got {workflow.rmax}")

        if errors:
            feedback = "Workflow validation failed:\n- " + "\n- ".join(errors)
            logger.info("[WorkflowGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="WorkflowGuard",
            )

        feedback = (
            f"Workflow valid: {len(workflow.action_pairs)} action pairs, "
            f"rmax={workflow.rmax}"
        )
        if workflow.backtrack_config:
            feedback += (
                f", backtrack_budget={workflow.backtrack_config.backtrack_budget}"
            )
        logger.info("[WorkflowGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="WorkflowGuard",
        )

    def _detect_cycle(self, action_pairs: dict) -> list[str] | None:
        """Detect cycles in the dependency graph using DFS.

        Returns the cycle path if found, None otherwise.
        """
        # Build adjacency list
        graph: dict[str, list[str]] = {ap_id: [] for ap_id in action_pairs}
        for ap_id, ap_spec in action_pairs.items():
            for req in ap_spec.requires:
                if req in graph:
                    graph[ap_id].append(req)

        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle - return path from neighbor to current
                    path.append(neighbor)  # Complete the cycle
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited and dfs(node):
                return path

        return None
