"""TestSyntaxGuard: Validates generated test code.

Ensures the test code is syntactically valid Python and contains
at least one test function or test class.
"""

import ast
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

logger = logging.getLogger("swe_bench_ablation.guards")


class TestSyntaxGuard(GuardInterface):
    """Validates generated test code.

    Checks:
    - Code parses with ast.parse()
    - Contains at least one test_ function or Test class
    """

    def __init__(
        self,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the guard."""

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        """Validate the test code artifact.

        Args:
            artifact: The test code artifact to validate
            **deps: Artifacts from prior workflow steps

        Returns:
            GuardResult with pass/fail and feedback
        """
        logger.info(
            "[TestSyntaxGuard] Validating artifact %s...", artifact.artifact_id[:8]
        )

        code = artifact.content.strip()
        if not code:
            return GuardResult(
                passed=False,
                feedback="Empty test code",
                guard_name="TestSyntaxGuard",
            )

        # Check for error markers
        if code.startswith("# Error:"):
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {code}",
                guard_name="TestSyntaxGuard",
            )

        # Parse with ast
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return GuardResult(
                passed=False,
                feedback=f"Syntax error at line {e.lineno}: {e.msg}",
                guard_name="TestSyntaxGuard",
            )

        # Check for test functions or test classes
        has_test_func = False
        has_test_class = False

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                has_test_func = True
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                has_test_class = True

        if not has_test_func and not has_test_class:
            return GuardResult(
                passed=False,
                feedback=(
                    "No test functions or test classes found. "
                    "Expected at least one function starting with 'test_' "
                    "or a class starting with 'Test'."
                ),
                guard_name="TestSyntaxGuard",
            )

        count = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        )
        feedback = f"Test code valid: {count} test function(s)"
        if has_test_class:
            feedback += ", has Test class(es)"
        logger.info("[TestSyntaxGuard] PASSED: %s", feedback)

        return GuardResult(
            passed=True,
            feedback=feedback,
            guard_name="TestSyntaxGuard",
        )
