"""SDLC Guards - Validation guards for the SDLC workflow.

Guards validate artifacts at each step:
- config_extracted: Validates ProjectConfig schema
- architecture_tests_valid: Validates ADD output (pytest-arch tests)
- scenarios_valid: Validates BDD Gherkin syntax
- all_tests_pass: Runs pytest and checks all tests pass
"""

import json
import logging
import subprocess
from pathlib import Path

from examples.base import register_guard
from pydantic import ValidationError

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import BDDScenarios, ImplementationManifest, ProjectConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Config Extracted Guard
# =============================================================================


class ConfigExtractedGuard(GuardInterface):
    """Validates that ProjectConfig was correctly extracted."""

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        """Validate ProjectConfig schema."""
        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON in artifact: {e}",
            )

        # Check for error response
        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Generator error: {data.get('details', data['error'])}",
            )

        # Validate schema
        try:
            config = ProjectConfig.model_validate(data)
        except ValidationError as e:
            return GuardResult(
                passed=False,
                feedback=f"ProjectConfig validation failed: {e}",
            )

        # Domain validation
        if not config.source_root:
            return GuardResult(
                passed=False,
                feedback="source_root cannot be empty",
            )

        if not config.package_name:
            return GuardResult(
                passed=False,
                feedback="package_name cannot be empty",
            )

        return GuardResult(passed=True, feedback="ProjectConfig is valid")


# =============================================================================
# Architecture Tests Valid Guard
# =============================================================================


class ArchitectureTestsValidGuard(GuardInterface):
    """Validates ADD output - pytest-arch test files."""

    def __init__(self, min_tests: int = 1):
        self._min_tests = min_tests

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        """Validate architecture tests output."""
        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON in ADD artifact: {e}",
            )

        # Check for error response
        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"ADD Generator error: {data.get('details', data['error'])}",
            )

        # Check for files in manifest
        files = data.get("files", [])
        if not files:
            return GuardResult(
                passed=False,
                feedback="ADD output contains no test files",
            )

        # Count tests (simple heuristic: count 'def test_' occurrences)
        test_count = 0
        for file_info in files:
            content = file_info.get("content", "")
            test_count += content.count("def test_")

        if test_count < self._min_tests:
            return GuardResult(
                passed=False,
                feedback=f"Only {test_count} tests found, minimum is {self._min_tests}",
            )

        return GuardResult(
            passed=True,
            feedback=f"Architecture tests valid: {test_count} tests in {len(files)} files",
        )


# =============================================================================
# Scenarios Valid Guard
# =============================================================================


class ScenariosValidGuard(GuardInterface):
    """Validates BDD scenarios (Gherkin format)."""

    def __init__(self, min_scenarios: int = 1):
        self._min_scenarios = min_scenarios

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        """Validate BDD scenarios."""
        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON in BDD artifact: {e}",
            )

        # Check for error response
        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"BDD Generator error: {data.get('details', data['error'])}",
            )

        # Validate schema
        try:
            bdd = BDDScenarios.model_validate(data)
        except ValidationError as e:
            return GuardResult(
                passed=False,
                feedback=f"BDDScenarios validation failed: {e}",
            )

        # Count total scenarios
        total_scenarios = sum(len(f.scenarios) for f in bdd.features)

        if total_scenarios < self._min_scenarios:
            return GuardResult(
                passed=False,
                feedback=f"Only {total_scenarios} scenarios found, minimum is {self._min_scenarios}",
            )

        # Validate each scenario has required steps
        for feature in bdd.features:
            for scenario in feature.scenarios:
                if not scenario.given:
                    return GuardResult(
                        passed=False,
                        feedback=f"Scenario '{scenario.name}' has no Given steps",
                    )
                if not scenario.when:
                    return GuardResult(
                        passed=False,
                        feedback=f"Scenario '{scenario.name}' has no When steps",
                    )
                if not scenario.then:
                    return GuardResult(
                        passed=False,
                        feedback=f"Scenario '{scenario.name}' has no Then steps",
                    )

        return GuardResult(
            passed=True,
            feedback=f"BDD scenarios valid: {total_scenarios} scenarios in {len(bdd.features)} features",
        )


# =============================================================================
# All Tests Pass Guard
# =============================================================================


class AllTestsPassGuard(GuardInterface):
    """Runs pytest and validates all tests pass.

    This guard runs pytest on the test directory and reports pass/fail.
    Files are expected to already be on disk (written by generators).
    """

    def __init__(self, workdir: Path | str | None = None, timeout: int = 60):
        self._workdir = Path(workdir).resolve() if workdir else Path.cwd()
        self._timeout = timeout

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        """Run tests and check results."""
        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON in implementation artifact: {e}",
            )

        # Check for error response
        if "error" in data:
            return GuardResult(
                passed=False,
                feedback=f"Coder Generator error: {data.get('details', data['error'])}",
            )

        # Validate schema
        try:
            impl = ImplementationManifest.model_validate(data)
        except ValidationError as e:
            return GuardResult(
                passed=False,
                feedback=f"ImplementationManifest validation failed: {e}",
            )

        if not impl.files:
            return GuardResult(
                passed=False,
                feedback="No implementation files generated",
            )

        # Files are already written by CoderGenerator, just run pytest
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-v", str(self._workdir / "tests")],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(self._workdir),
            )

            if result.returncode == 0:
                return GuardResult(
                    passed=True,
                    feedback=f"All tests passed!\n\n{result.stdout}",
                )
            else:
                return GuardResult(
                    passed=False,
                    feedback=f"Tests failed:\n\n{result.stdout}\n\n{result.stderr}",
                )

        except subprocess.TimeoutExpired:
            return GuardResult(
                passed=False,
                feedback=f"Tests timed out after {self._timeout} seconds",
                fatal=True,  # Don't retry on timeout
            )
        except FileNotFoundError:
            return GuardResult(
                passed=False,
                feedback="pytest not found. Ensure pytest is installed.",
                fatal=True,
            )
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Error running tests: {e}",
            )


# =============================================================================
# Guard Registration
# =============================================================================


def register_sdlc_guards() -> None:
    """Register all SDLC guards with the guard registry."""
    register_guard("config_extracted", ConfigExtractedGuard)
    register_guard("architecture_tests_valid", ArchitectureTestsValidGuard)
    register_guard("scenarios_valid", ScenariosValidGuard)
    register_guard("all_tests_pass", AllTestsPassGuard)

    logger.debug(
        "[SDLC] Registered guards: config_extracted, architecture_tests_valid, scenarios_valid, all_tests_pass"
    )
