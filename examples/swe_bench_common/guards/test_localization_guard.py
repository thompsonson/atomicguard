"""TestLocalizationGuard: Validates test localization output.

Ensures the test localization identifies valid test files and patterns
that exist in the repository.
"""

import json
import logging
import shlex
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from examples.swe_bench_common.models import TestLocalization

logger = logging.getLogger("swe_bench_ablation.guards")


class TestLocalizationGuard(GuardInterface):
    """Validates test localization schema and file existence.

    Checks:
    - Valid JSON matching TestLocalization schema
    - At least one test file identified
    - Test files exist in repository
    - Conftest files exist
    - Test library is valid enum
    - Test invocation paths exist
    """

    VALID_TEST_LIBRARIES = {"pytest", "unittest", "nose", "doctest", "hypothesis"}
    VALID_TEST_STYLES = {"function-based", "class-based", "bdd", "mixed"}

    def __init__(
        self,
        repo_root: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the guard.

        Args:
            repo_root: Repository root for file validation
        """
        self._repo_root = repo_root

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the test localization artifact.

        Args:
            artifact: The test localization artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info(
            "[TestLocalizationGuard] Validating artifact %s...",
            artifact.artifact_id[:8],
        )

        # Parse JSON
        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="TestLocalizationGuard",
            )

        # Check for error field
        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {data['error']}",
                guard_name="TestLocalizationGuard",
            )

        # Validate against schema
        try:
            loc = TestLocalization.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="TestLocalizationGuard",
            )

        errors: list[str] = []

        # Validate test_library
        if loc.test_library.lower() not in self.VALID_TEST_LIBRARIES:
            errors.append(
                f"Unknown test library '{loc.test_library}'. "
                f"Expected one of: {', '.join(sorted(self.VALID_TEST_LIBRARIES))}"
            )

        # Validate test_style
        if loc.test_style not in self.VALID_TEST_STYLES:
            errors.append(
                f"Invalid test_style '{loc.test_style}'. "
                f"Expected one of: {', '.join(sorted(self.VALID_TEST_STYLES))}"
            )

        # Validate files exist (if repo_root provided)
        if self._repo_root:
            repo_path = Path(self._repo_root)

            # Check test files
            missing_test_files = [
                f for f in loc.test_files if not (repo_path / f).exists()
            ]
            if missing_test_files:
                truncated = missing_test_files[:3]
                suffix = (
                    f" (and {len(missing_test_files) - 3} more)"
                    if len(missing_test_files) > 3
                    else ""
                )
                errors.append(
                    f"These test files do not exist in the repository: "
                    f"{', '.join(truncated)}{suffix}. "
                    f"Only reference files that actually exist in the codebase."
                )

            # Check conftest files
            missing_conftest = [
                f for f in loc.conftest_files if not (repo_path / f).exists()
            ]
            if missing_conftest:
                truncated = missing_conftest[:3]
                suffix = (
                    f" (and {len(missing_conftest) - 3} more)"
                    if len(missing_conftest) > 3
                    else ""
                )
                errors.append(
                    f"These conftest files do not exist in the repository: "
                    f"{', '.join(truncated)}{suffix}. "
                    f"Only reference files that actually exist in the codebase."
                )

            # Check test invocation paths exist
            invocation_errors = self._validate_invocation_paths(
                loc.test_invocation, repo_path
            )
            errors.extend(invocation_errors)

        if errors:
            feedback = "Test localization validation failed:\n- " + "\n- ".join(errors)
            logger.info("[TestLocalizationGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="TestLocalizationGuard",
            )

        feedback = (
            f"Test localization valid: {len(loc.test_files)} test files, "
            f"library={loc.test_library}, style={loc.test_style}"
        )
        logger.info("[TestLocalizationGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="TestLocalizationGuard",
        )

    def _validate_invocation_paths(self, invocation: str, repo_path: Path) -> list[str]:
        """Parse test_invocation and verify paths exist.

        Args:
            invocation: The test invocation command string
            repo_path: Path to the repository root

        Returns:
            List of error messages (empty if valid)
        """
        errors: list[str] = []

        try:
            tokens = shlex.split(invocation)
        except ValueError as e:
            return [f"Invalid test_invocation syntax: {e}"]

        if not tokens:
            return ["test_invocation is empty"]

        # First token should be test runner
        runner = tokens[0].lower()
        # Handle "python -m pytest" pattern
        if runner == "python" and len(tokens) >= 3 and tokens[1] == "-m":
            runner = tokens[2].lower()

        valid_runners = {"pytest", "python", "nosetests", "tox", "unittest"}
        if runner not in valid_runners:
            errors.append(
                f"Unknown test runner '{tokens[0]}'. "
                f"Expected one of: {', '.join(sorted(valid_runners))}"
            )

        # Look for file/directory paths in tokens
        for token in tokens[1:]:
            if token.startswith("-"):
                continue  # Skip flags
            if token == "-m":
                continue  # Skip -m flag
            # Check if it looks like a path
            if "/" in token or token.endswith(".py"):
                # Handle pytest path::function syntax
                path_part = token.split("::")[0]
                path = repo_path / path_part
                if not path.exists():
                    errors.append(
                        f"This path in test_invocation does not exist in the "
                        f"repository: {path_part}. "
                        f"Only reference paths that actually exist in the codebase."
                    )

        return errors
