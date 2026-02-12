"""StructureGuard: Validates project structure analysis output.

Ensures the structure analysis has valid JSON matching the ProjectStructure
schema with non-empty required fields.
"""

import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from examples.swe_bench_common.models import ProjectStructure

logger = logging.getLogger("swe_bench_ablation.guards")


class StructureGuard(GuardInterface):
    """Validates project structure analysis output.

    Checks:
    - Valid JSON matching ProjectStructure schema
    - At least one root module or test directory identified
    - Test framework is specified
    """

    def __init__(
        self,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the guard."""
        pass

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the structure artifact.

        Args:
            artifact: The structure artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info(
            "[StructureGuard] Validating artifact %s...", artifact.artifact_id[:8]
        )

        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="StructureGuard",
            )

        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {data['error']}",
                guard_name="StructureGuard",
            )

        try:
            structure = ProjectStructure.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="StructureGuard",
            )

        errors: list[str] = []

        # Must have at least some structure identified
        if not structure.root_modules and not structure.test_directories:
            errors.append(
                "No root modules or test directories identified. "
                "Must identify at least one."
            )

        if not structure.test_framework:
            errors.append("Test framework not specified.")

        if errors:
            feedback = "Structure validation failed:\n- " + "\n- ".join(errors)
            logger.info("[StructureGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="StructureGuard",
            )

        feedback = (
            f"Structure valid: {len(structure.root_modules)} root modules, "
            f"framework={structure.test_framework}, "
            f"{len(structure.test_directories)} test dirs"
        )
        logger.info("[StructureGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="StructureGuard",
        )
