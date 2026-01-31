"""
ArchitectureTestsGuard: Validates generated architecture tests.

Validates that ADDGenerator produced valid pytest-arch tests.
"""

import ast
import json
import logging
import re
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import TestSuite

logger = logging.getLogger("sdlc_checkpoint")


class ArchitectureTestsGuard(GuardInterface):
    """
    Validates generated architecture tests.

    Checks:
    - Valid JSON structure matching TestSuite schema
    - At least min_tests tests generated
    - Test syntax is valid Python
    - Test naming follows conventions
    - pytestarch API usage is valid
    """

    # Whitelisted pytestarch methods
    WHITELISTED_METHODS = frozenset(
        {
            "modules_that",
            "are_sub_modules_of",
            "are_named",
            "have_name_matching",
            "should",
            "should_not",
            "should_only",
            "import_modules_that",
            "be_imported_by_modules_that",
            "import_modules_except_modules_that",
            "be_imported_by_modules_except_modules_that",
            "import_anything",
            "be_imported_by_anything",
            "assert_applies",
        }
    )

    def __init__(self, min_tests: int = 1):
        self._min_tests = min_tests

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate generated architecture tests."""
        logger.debug("[ArchitectureTestsGuard] Validating tests...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[ArchitectureTestsGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[ArchitectureTestsGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        # Check for errors
        if "error" in data:
            logger.debug(
                f"[ArchitectureTestsGuard] Generator error: {data.get('error')}"
            )
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('details', data['error'])}",
            )

        # Parse as TestSuite
        try:
            suite = TestSuite.model_validate(data)
            logger.debug(
                f"[ArchitectureTestsGuard] Schema valid, {len(suite.tests)} tests"
            )
        except Exception as e:
            logger.debug(f"[ArchitectureTestsGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Check minimum test count
        if len(suite.tests) < self._min_tests:
            logger.debug(
                f"[ArchitectureTestsGuard] Insufficient tests: {len(suite.tests)} < {self._min_tests}"
            )
            return GuardResult(
                passed=False,
                feedback=f"Expected at least {self._min_tests} tests, got {len(suite.tests)}",
            )

        # Check test naming
        for test in suite.tests:
            if not test.test_name.startswith("test_"):
                logger.debug(
                    f"[ArchitectureTestsGuard] Invalid test name: {test.test_name}"
                )
                return GuardResult(
                    passed=False,
                    feedback=f"Test '{test.test_name}' must start with 'test_'",
                )

        # Check for unique test names
        test_names = [t.test_name for t in suite.tests]
        if len(set(test_names)) != len(test_names):
            logger.debug("[ArchitectureTestsGuard] Duplicate test names")
            return GuardResult(
                passed=False,
                feedback="Duplicate test names detected",
            )

        # Validate syntax of assembled code
        full_code = self._assemble_code(suite)
        try:
            ast.parse(full_code)
            logger.debug("[ArchitectureTestsGuard] Syntax valid")
        except SyntaxError as e:
            logger.debug(f"[ArchitectureTestsGuard] Syntax error: {e}")
            return GuardResult(
                passed=False,
                feedback=f"Syntax error at line {e.lineno}: {e.msg}",
            )

        # Check pytestarch API whitelist
        api_result = self._check_api_whitelist(full_code)
        if api_result:
            return api_result

        logger.debug("[ArchitectureTestsGuard] ✓ All checks passed")
        return GuardResult(
            passed=True, feedback=f"Valid TestSuite with {len(suite.tests)} tests"
        )

    def _assemble_code(self, suite: TestSuite) -> str:
        """Assemble all code for syntax checking."""
        lines = list(suite.imports)
        lines.extend(suite.fixtures)
        for test in suite.tests:
            lines.append(test.test_code)
        return "\n".join(lines)

    def _check_api_whitelist(self, code: str) -> GuardResult | None:
        """Check that only whitelisted pytestarch methods are used."""
        method_pattern = re.compile(r"\.([a-z_][a-z0-9_]*)\s*\(", re.IGNORECASE)

        # Look for Rule() chains
        rule_sections = re.split(r"\bRule\s*\(\s*\)", code)

        for section in rule_sections[1:]:  # Skip code before first Rule()
            methods = method_pattern.findall(section)
            for method in methods:
                if method not in self.WHITELISTED_METHODS:
                    logger.debug(
                        f"[ArchitectureTestsGuard] Non-whitelisted method: .{method}()"
                    )
                    return GuardResult(
                        passed=False,
                        feedback=f"Invalid pytestarch method '.{method}()'. "
                        f"Only these methods are allowed: {', '.join(sorted(self.WHITELISTED_METHODS))}",
                    )

        return None
