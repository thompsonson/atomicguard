"""LocalizationGuard: Validates localization output.

Ensures the localization identifies valid files and functions
that exist in the repository.
"""

import json
import logging
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from examples.swe_bench_common.models import Localization

logger = logging.getLogger("swe_bench_ablation.guards")


class LocalizationGuard(GuardInterface):
    """Validates localization output.

    Checks:
    - Valid JSON matching Localization schema
    - At least one file identified
    - Maximum 5 files (configurable)
    - Files exist in repository (if repo_root provided)
    """

    def __init__(
        self,
        require_files: bool = True,
        min_files: int = 1,
        max_files: int = 5,
        require_functions: bool = False,
        repo_root: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the guard.

        Args:
            require_files: Whether to require at least one file
            min_files: Minimum number of files required
            max_files: Maximum files allowed
            require_functions: Whether to require functions
            repo_root: Repository root for file validation
        """
        self._require_files = require_files
        self._min_files = min_files
        self._max_files = max_files
        self._require_functions = require_functions
        self._repo_root = repo_root

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the localization artifact.

        Args:
            artifact: The localization artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info(
            "[LocalizationGuard] Validating artifact %s...", artifact.artifact_id[:8]
        )

        # Parse JSON
        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="LocalizationGuard",
            )

        # Check for error field
        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {data['error']}",
                guard_name="LocalizationGuard",
            )

        # Validate against schema
        try:
            localization = Localization.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="LocalizationGuard",
            )

        errors: list[str] = []

        # Check file count
        if self._require_files and not localization.files:
            errors.append("No files identified. Must identify at least one file.")

        if len(localization.files) < self._min_files:
            errors.append(
                f"Only {len(localization.files)} files identified, "
                f"need at least {self._min_files}."
            )

        if len(localization.files) > self._max_files:
            errors.append(
                f"Too many files ({len(localization.files)}), "
                f"maximum is {self._max_files}."
            )

        # Check functions
        if self._require_functions and not localization.functions:
            errors.append("No functions identified.")

        # Validate files exist
        if self._repo_root:
            repo_path = Path(self._repo_root)
            missing_files = []
            for file in localization.files:
                if not (repo_path / file).exists():
                    missing_files.append(file)
            if missing_files:
                errors.append(
                    f"Files not found in repository: {', '.join(missing_files[:3])}"
                    + (
                        f" (and {len(missing_files) - 3} more)"
                        if len(missing_files) > 3
                        else ""
                    )
                )

        if errors:
            feedback = "Localization validation failed:\n- " + "\n- ".join(errors)
            logger.info("[LocalizationGuard] ✗ REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="LocalizationGuard",
            )

        feedback = f"Localization valid: {len(localization.files)} files, {len(localization.functions)} functions"
        logger.info("[LocalizationGuard] ✓ PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="LocalizationGuard",
        )
