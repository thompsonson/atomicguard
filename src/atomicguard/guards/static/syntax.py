"""
Syntax validation guard.

Pure guard with no I/O dependencies - validates Python AST.
"""

import ast
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult


class SyntaxGuard(GuardInterface):
    """
    Validates Python syntax using AST parsing.

    This is a pure guard with no I/O - it only validates
    that the artifact content is syntactically valid Python.
    """

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """
        Parse artifact content as Python AST.

        Returns:
            GuardResult with passed=True if syntax is valid
        """
        try:
            ast.parse(artifact.content)
            return GuardResult(passed=True, feedback="Syntax valid")
        except SyntaxError as e:
            return GuardResult(passed=False, feedback=f"Syntax error: {e}")
