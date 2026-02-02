"""Multi-language test syntax guard.

For Python, delegates to ``ast.parse()``.  For other languages, applies
heuristic checks: the syntax-check function from :mod:`language` (brace
balancing) plus a regex check for test-function patterns.
"""

import logging
import re
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..language import LanguageConfig

logger = logging.getLogger("swe_bench_pro.guards")


class MultiLangTestSyntaxGuard(GuardInterface):
    """Validates generated test code for any supported language.

    Checks:
    * Code is non-empty and not an error marker.
    * Language-specific syntax check passes (``ast.parse`` for Python,
      brace-balance heuristic for others).
    * At least one test-function pattern is present.
    """

    def __init__(self, language_config: LanguageConfig, **kwargs: Any):  # noqa: ARG002
        self._lang = language_config

    def validate(
        self,
        artifact: Artifact,
        **deps: Artifact,  # noqa: ARG002
    ) -> GuardResult:
        code = artifact.content.strip()

        if not code:
            return GuardResult(
                passed=False,
                feedback="Empty test code",
                guard_name="MultiLangTestSyntaxGuard",
            )

        if code.startswith("# Error:") or code.startswith("// Error:"):
            return GuardResult(
                passed=False,
                feedback=f"Generator returned error: {code}",
                guard_name="MultiLangTestSyntaxGuard",
            )

        # Language-specific syntax check
        if self._lang.syntax_check_fn is not None:
            ok, msg = self._lang.syntax_check_fn(code)
            if not ok:
                return GuardResult(
                    passed=False,
                    feedback=f"Syntax check failed: {msg}",
                    guard_name="MultiLangTestSyntaxGuard",
                )

        # Test-function pattern check
        if not re.search(self._lang.test_function_pattern, code):
            return GuardResult(
                passed=False,
                feedback=(
                    f"No {self._lang.test_framework} test patterns found. "
                    f"Expected pattern: {self._lang.test_function_pattern}"
                ),
                guard_name="MultiLangTestSyntaxGuard",
            )

        return GuardResult(
            passed=True,
            feedback=f"Test code passes {self._lang.name} validation",
            guard_name="MultiLangTestSyntaxGuard",
        )
