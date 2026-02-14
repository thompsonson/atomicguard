"""TestSetupVerificationGuard: Verifies test infrastructure is runnable.

Performs runtime checks to verify that the LLM's claims about test
infrastructure are actually actionable.
"""

import json
import logging
import shlex
import subprocess
from typing import Any

from examples.swe_bench_common.models import TestLocalization

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

logger = logging.getLogger("swe_bench_ablation.guards")


class TestSetupVerificationGuard(GuardInterface):
    """Verifies test infrastructure is actually runnable.

    Checks:
    - Test framework is installed (e.g., `pytest --version`)
    - Can collect tests without running them (e.g., `pytest --collect-only`)
    - Plugins are importable

    This is a "smoke test" guard - it verifies the LLM's claims are actionable.
    """

    def __init__(
        self,
        repo_root: str | None = None,
        timeout_seconds: int = 30,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the guard.

        Args:
            repo_root: Repository root for running commands
            timeout_seconds: Timeout for subprocess operations
        """
        self._repo_root = repo_root
        self._timeout = timeout_seconds

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate that test setup is runnable.

        Args:
            artifact: The test localization artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info(
            "[TestSetupVerificationGuard] Validating artifact %s...",
            artifact.artifact_id[:8],
        )

        # Skip if no repo_root provided
        if not self._repo_root:
            return GuardResult(
                passed=True,
                feedback="Test setup verification skipped (no repo_root)",
                guard_name="TestSetupVerificationGuard",
            )

        # Parse JSON
        try:
            data = json.loads(artifact.content)
        except json.JSONDecodeError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid JSON: {e}",
                guard_name="TestSetupVerificationGuard",
            )

        # Validate against schema
        try:
            loc = TestLocalization.model_validate(data)
        except Exception as e:
            return GuardResult(
                passed=False,
                feedback=f"Schema validation failed: {e}",
                guard_name="TestSetupVerificationGuard",
            )

        errors: list[str] = []

        # Check test framework is installed
        framework_result = self._check_framework_installed(loc.test_library)
        if not framework_result.passed:
            errors.append(framework_result.feedback)

        # Check plugins are importable
        for plugin in loc.test_plugins:
            plugin_result = self._check_plugin_importable(plugin)
            if not plugin_result.passed:
                errors.append(plugin_result.feedback)

        # Try to collect tests (don't run them)
        collect_result = self._try_collect_tests(loc.test_invocation)
        if not collect_result.passed:
            errors.append(collect_result.feedback)

        if errors:
            feedback = "Test setup verification failed:\n- " + "\n- ".join(errors)
            logger.info("[TestSetupVerificationGuard] REJECTED: %s", feedback)
            return GuardResult(
                passed=False,
                feedback=feedback,
                guard_name="TestSetupVerificationGuard",
            )

        feedback = (
            f"Test setup verified: {loc.test_library} installed, "
            f"{len(loc.test_plugins)} plugins OK, tests collectible"
        )
        logger.info("[TestSetupVerificationGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="TestSetupVerificationGuard",
        )

    def _check_framework_installed(self, library: str) -> GuardResult:
        """Check if test framework is installed.

        Args:
            library: Name of the test library (pytest, unittest, etc.)

        Returns:
            GuardResult indicating whether the framework is installed
        """
        cmd_map = {
            "pytest": ["pytest", "--version"],
            "unittest": ["python", "-c", "import unittest; print('ok')"],
            "nose": ["nosetests", "--version"],
            "doctest": ["python", "-c", "import doctest; print('ok')"],
            "hypothesis": ["python", "-c", "import hypothesis; print('ok')"],
        }

        cmd = cmd_map.get(library.lower())
        if not cmd:
            return GuardResult(
                passed=True,
                feedback=f"Unknown framework '{library}', skipping check",
                guard_name="TestSetupVerificationGuard",
            )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self._repo_root,
            )
            if result.returncode != 0:
                return GuardResult(
                    passed=False,
                    feedback=f"{library} not installed or broken",
                    guard_name="TestSetupVerificationGuard",
                )
        except FileNotFoundError:
            return GuardResult(
                passed=False,
                feedback=f"{library} command not found",
                guard_name="TestSetupVerificationGuard",
            )
        except subprocess.TimeoutExpired:
            return GuardResult(
                passed=False,
                feedback=f"{library} --version timed out",
                guard_name="TestSetupVerificationGuard",
            )

        return GuardResult(
            passed=True,
            feedback=f"{library} installed",
            guard_name="TestSetupVerificationGuard",
        )

    def _check_plugin_importable(self, plugin: str) -> GuardResult:
        """Check if a pytest plugin is importable.

        Args:
            plugin: Name of the plugin (e.g., 'pytest-qt', 'pytest-asyncio')

        Returns:
            GuardResult indicating whether the plugin is importable
        """
        # Convert pytest-foo to pytest_foo for import
        module_name = plugin.replace("-", "_")

        try:
            result = subprocess.run(
                ["python", "-c", f"import {module_name}"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self._repo_root,
            )
            if result.returncode != 0:
                return GuardResult(
                    passed=False,
                    feedback=f"Plugin '{plugin}' not importable",
                    guard_name="TestSetupVerificationGuard",
                )
        except FileNotFoundError:
            return GuardResult(
                passed=False,
                feedback="Python not found",
                guard_name="TestSetupVerificationGuard",
            )
        except subprocess.TimeoutExpired:
            return GuardResult(
                passed=False,
                feedback=f"Plugin '{plugin}' import check timed out",
                guard_name="TestSetupVerificationGuard",
            )

        return GuardResult(
            passed=True,
            feedback=f"Plugin '{plugin}' OK",
            guard_name="TestSetupVerificationGuard",
        )

    def _try_collect_tests(self, invocation: str) -> GuardResult:
        """Try to collect tests without running them.

        Args:
            invocation: The test invocation command string

        Returns:
            GuardResult indicating whether tests are collectible
        """
        try:
            tokens = shlex.split(invocation)
        except ValueError as e:
            return GuardResult(
                passed=False,
                feedback=f"Invalid invocation syntax: {e}",
                guard_name="TestSetupVerificationGuard",
            )

        if not tokens:
            return GuardResult(
                passed=False,
                feedback="Empty test invocation",
                guard_name="TestSetupVerificationGuard",
            )

        # Determine runner and add --collect-only for pytest
        runner = tokens[0].lower()

        # Handle "python -m pytest" pattern
        if runner == "python" and len(tokens) >= 3 and tokens[1] == "-m":
            actual_runner = tokens[2].lower()
            if actual_runner == "pytest":
                collect_cmd = tokens + ["--collect-only", "-q"]
            else:
                # For other frameworks, just verify the command parses
                return GuardResult(
                    passed=True,
                    feedback="Skipping collect for non-pytest runner",
                    guard_name="TestSetupVerificationGuard",
                )
        elif runner in ("pytest", "py.test"):
            collect_cmd = tokens + ["--collect-only", "-q"]
        else:
            # For other frameworks, just verify the command parses
            return GuardResult(
                passed=True,
                feedback="Skipping collect for non-pytest runner",
                guard_name="TestSetupVerificationGuard",
            )

        try:
            result = subprocess.run(
                collect_cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=self._repo_root,
            )
            if result.returncode != 0:
                # Extract useful error info
                error_lines = result.stderr.strip().split("\n")[-3:]
                error_summary = " ".join(line.strip() for line in error_lines if line)
                return GuardResult(
                    passed=False,
                    feedback=f"Test collection failed: {error_summary}",
                    guard_name="TestSetupVerificationGuard",
                )
        except subprocess.TimeoutExpired:
            return GuardResult(
                passed=False,
                feedback="Test collection timed out",
                guard_name="TestSetupVerificationGuard",
            )
        except FileNotFoundError:
            return GuardResult(
                passed=False,
                feedback=f"Command '{runner}' not found",
                guard_name="TestSetupVerificationGuard",
            )

        return GuardResult(
            passed=True,
            feedback="Tests collectible",
            guard_name="TestSetupVerificationGuard",
        )
