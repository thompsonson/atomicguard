"""
Domain-specific guards for ADD workflow.

These guards validate the structured output from each action pair
in the ADD workflow.
"""

import ast
import json
import logging
import re
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from .models import ArtifactManifest, GatesExtractionResult, TestSuite

logger = logging.getLogger("add_workflow")


class GatesExtractedGuard(GuardInterface):
    """
    Validates gate extraction completeness.

    Checks:
    - At least N gates extracted (configurable)
    - No duplicate gate IDs
    - Each gate has valid layer assignment
    - No validation errors in content
    """

    def __init__(
        self,
        min_gates: int = 1,
        required_gates: list[str] | None = None,
    ):
        self._min_gates = min_gates
        self._required_gates = required_gates or []

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate extracted gates."""
        logger.debug("[GatesGuard] Validating extracted gates...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[GatesGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[GatesGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        # Check for validation errors from generator
        if "error" in data:
            logger.debug(f"[GatesGuard] Generator returned error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('details', data['error'])}",
            )

        # Parse as GatesExtractionResult
        try:
            result = GatesExtractionResult.model_validate(data)
            logger.debug(f"[GatesGuard] Schema valid, {len(result.gates)} gates")
        except Exception as e:
            logger.debug(f"[GatesGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Check minimum count
        if len(result.gates) < self._min_gates:
            logger.debug(
                f"[GatesGuard] Insufficient gates: {len(result.gates)} < {self._min_gates}"
            )
            return GuardResult(
                passed=False,
                feedback=f"Expected at least {self._min_gates} gates, got {len(result.gates)}",
            )

        # Check required gates
        gate_ids = {g.gate_id for g in result.gates}
        missing = set(self._required_gates) - gate_ids
        if missing:
            logger.debug(f"[GatesGuard] Missing required gates: {missing}")
            return GuardResult(
                passed=False,
                feedback=f"Missing required gates: {missing}",
            )

        # Check duplicates
        if len(gate_ids) != len(result.gates):
            logger.debug("[GatesGuard] Duplicate gate IDs detected")
            return GuardResult(
                passed=False,
                feedback="Duplicate gate IDs detected",
            )

        logger.debug("[GatesGuard] ✓ All checks passed")
        return GuardResult(passed=True, feedback=f"Extracted {len(result.gates)} gates")


class TestNamingGuard(GuardInterface):
    """
    Validates test naming conventions.

    Checks:
    - All test names start with 'test_'
    - Test names are unique
    """

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate test naming."""
        logger.debug("[NamingGuard] Validating test naming...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[NamingGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[NamingGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        # Check for errors
        if "error" in data:
            logger.debug(f"[NamingGuard] Generator returned error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('details', data['error'])}",
            )

        # Parse as TestSuite
        try:
            suite = TestSuite.model_validate(data)
            logger.debug(f"[NamingGuard] Schema valid, {len(suite.tests)} tests")
        except Exception as e:
            logger.debug(f"[NamingGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Check test naming
        for test in suite.tests:
            if not test.test_name.startswith("test_"):
                logger.debug(f"[NamingGuard] Invalid test name: {test.test_name}")
                return GuardResult(
                    passed=False,
                    feedback=f"Test '{test.test_name}' must start with 'test_'",
                )

        # Check uniqueness
        test_names = [t.test_name for t in suite.tests]
        if len(set(test_names)) != len(test_names):
            logger.debug("[NamingGuard] Duplicate test names detected")
            return GuardResult(
                passed=False,
                feedback="Duplicate test names detected",
            )

        logger.debug("[NamingGuard] ✓ All checks passed")
        return GuardResult(passed=True, feedback=f"Validated {len(suite.tests)} tests")


class TestSyntaxGuard(GuardInterface):
    """
    Validates Python syntax of generated tests.

    Uses ast.parse() to check each test function.
    Note: This is ADD-specific because it parses TestSuite JSON,
    unlike the core SyntaxGuard which validates raw artifact content.
    """

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate test syntax."""
        logger.debug("[SyntaxGuard] Validating test syntax...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[SyntaxGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[SyntaxGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        if "error" in data:
            logger.debug(f"[SyntaxGuard] Generator returned error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('details', data['error'])}",
            )

        try:
            suite = TestSuite.model_validate(data)
            logger.debug(f"[SyntaxGuard] Schema valid, {len(suite.tests)} tests")
        except Exception as e:
            logger.debug(f"[SyntaxGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Assemble full code for syntax check
        full_code = self._assemble_code(suite)
        logger.debug(f"[SyntaxGuard] Assembled {len(full_code)} chars of code")

        try:
            ast.parse(full_code)
        except SyntaxError as e:
            logger.debug(f"[SyntaxGuard] Syntax error at line {e.lineno}: {e.msg}")
            return GuardResult(
                passed=False,
                feedback=f"Syntax error at line {e.lineno}: {e.msg}",
            )

        logger.debug("[SyntaxGuard] ✓ Syntax valid")
        return GuardResult(passed=True, feedback="Syntax valid")

    def _assemble_code(self, suite: TestSuite) -> str:
        """Assemble all code for syntax checking."""
        lines = list(suite.imports)
        lines.extend(suite.fixtures)
        for test in suite.tests:
            lines.append(test.test_code)
        return "\n".join(lines)


class PytestArchAPIGuard(GuardInterface):
    """
    Validates pytestarch API usage with strict whitelist.

    Uses a two-phase approach:
    1. Static whitelist check - reject any method not in the whitelist
    2. Code execution - catch runtime errors from pytestarch
    """

    # ONLY these pytestarch v4.0.1 methods are allowed
    WHITELISTED_METHODS = frozenset(
        {
            # Module selection
            "modules_that",
            "are_sub_modules_of",
            "are_named",
            "have_name_matching",
            # Behavior
            "should",
            "should_not",
            "should_only",
            # Dependency type
            "import_modules_that",
            "be_imported_by_modules_that",
            "import_modules_except_modules_that",
            "be_imported_by_modules_except_modules_that",
            "import_anything",
            "be_imported_by_anything",
            # Assertion
            "assert_applies",
        }
    )

    # Pattern to extract method calls in fluent chains
    _METHOD_CALL_PATTERN = re.compile(r"\.([a-z_][a-z0-9_]*)\s*\(", re.IGNORECASE)

    def _check_whitelist(self, code: str) -> GuardResult | None:
        """Reject any method not in the whitelist."""
        # Find all method calls in Rule chains
        # Look for code after Rule() initialization
        rule_sections = re.split(r"\bRule\s*\(\s*\)", code)

        for section in rule_sections[1:]:  # Skip code before first Rule()
            methods = self._METHOD_CALL_PATTERN.findall(section)

            for method in methods:
                if method not in self.WHITELISTED_METHODS:
                    logger.debug(f"[APIGuard] Non-whitelisted method: .{method}()")
                    return GuardResult(
                        passed=False,
                        feedback=f"Invalid pytestarch method '.{method}()'. "
                        f"Only these methods are allowed: "
                        f"{', '.join(sorted(self.WHITELISTED_METHODS))}",
                    )

        return None  # All methods are whitelisted

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate pytestarch API usage."""
        logger.debug("[APIGuard] Validating pytestarch API usage...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[APIGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[APIGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        if "error" in data:
            logger.debug(f"[APIGuard] Generator returned error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('details', data['error'])}",
            )

        try:
            suite = TestSuite.model_validate(data)
            logger.debug(f"[APIGuard] Schema valid, {len(suite.tests)} tests")
        except Exception as e:
            logger.debug(f"[APIGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Assemble full code
        code = self._assemble_code(suite)
        logger.debug(f"[APIGuard] Assembled {len(code)} chars of code")

        # Check whitelist BEFORE execution (faster feedback)
        whitelist_result = self._check_whitelist(code)
        if whitelist_result:
            return whitelist_result

        logger.debug("[APIGuard] Whitelist check passed")

        try:
            # Compile (catches syntax errors)
            compiled = compile(code, "<generated>", "exec")
            logger.debug("[APIGuard] Code compiled successfully")

            # Execute with pytestarch available
            # This will raise AttributeError for hallucinated methods
            import pytest
            import pytestarch

            namespace = {
                "pytestarch": pytestarch,
                "pytest": pytest,
                "__name__": "__main__",
            }
            exec(compiled, namespace)
            logger.debug("[APIGuard] Code executed successfully")

            return GuardResult(passed=True, feedback="pytestarch API valid")

        except AttributeError as e:
            # e.g., "module 'pytestarch' has no attribute 'modules_in'"
            logger.debug(f"[APIGuard] Invalid API: {e}")
            return GuardResult(
                passed=False,
                feedback=f"Invalid pytestarch API: {e}. Use Rule().modules_that().are_sub_modules_of()... pattern.",
            )
        except ImportError as e:
            logger.debug(f"[APIGuard] Import error: {e}")
            return GuardResult(
                passed=False,
                feedback=f"Import error: {e}. Use 'from pytestarch import Rule, get_evaluable_architecture'",
            )
        except SyntaxError as e:
            logger.debug(f"[APIGuard] Syntax error: {e}")
            return GuardResult(
                passed=False,
                feedback=f"Syntax error at line {e.lineno}: {e.msg}",
            )
        except Exception as e:
            # Catch other execution errors (but not test failures)
            error_msg = str(e)
            logger.debug(f"[APIGuard] Execution error: {type(e).__name__}: {error_msg}")
            return GuardResult(
                passed=False,
                feedback=f"Code execution error: {type(e).__name__}: {error_msg}",
            )

    def _assemble_code(self, suite: TestSuite) -> str:
        """Assemble all code for execution."""
        lines = list(suite.imports)
        lines.extend(suite.fixtures)
        for test in suite.tests:
            lines.append(test.test_code)
        return "\n".join(lines)


class ArtifactStructureGuard(GuardInterface):
    """
    Validates final artifact structure.

    Checks:
    - Valid ArtifactManifest structure
    - At least one test generated
    - Files list is non-empty
    """

    def __init__(self, min_tests: int = 1):
        self._min_tests = min_tests

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate artifact structure."""
        logger.debug("[StructureGuard] Validating artifact structure...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[StructureGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[StructureGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        if "error" in data:
            logger.debug(
                f"[StructureGuard] Generator returned error: {data.get('error')}"
            )
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('details', data['error'])}",
            )

        try:
            manifest = ArtifactManifest.model_validate(data)
            logger.debug(
                f"[StructureGuard] Schema valid, {manifest.test_count} tests, {len(manifest.files)} files"
            )
        except Exception as e:
            logger.debug(f"[StructureGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Check test count
        if manifest.test_count < self._min_tests:
            logger.debug(
                f"[StructureGuard] Insufficient tests: {manifest.test_count} < {self._min_tests}"
            )
            return GuardResult(
                passed=False,
                feedback=f"Expected at least {self._min_tests} tests, got {manifest.test_count}",
            )

        # Check files
        if not manifest.files:
            logger.debug("[StructureGuard] No files in manifest")
            return GuardResult(
                passed=False,
                feedback="No files in manifest",
            )

        # Check non-empty content for test files
        for f in manifest.files:
            is_python_file = f.path.endswith(".py")
            is_not_init = not f.path.endswith("__init__.py")
            if is_python_file and is_not_init and not f.content.strip():
                logger.debug(f"[StructureGuard] Empty content in: {f.path}")
                return GuardResult(
                    passed=False,
                    feedback=f"Empty content in: {f.path}",
                )

        logger.debug("[StructureGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"Valid manifest: {manifest.test_count} tests, {len(manifest.files)} files",
        )
