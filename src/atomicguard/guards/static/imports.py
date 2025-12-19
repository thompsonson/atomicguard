"""
Import validation guard.

Pure AST-based guard that validates all used names are imported or defined.
Does NOT execute code - uses static analysis only.
"""

import ast
import builtins
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult


class ImportGuard(GuardInterface):
    """
    Validates that all names used in code are imported or defined.

    Pure, static guard using AST analysis. Does NOT execute code.
    Single responsibility: verify that referenced names exist.

    This catches common LLM mistakes like using `pytest.raises()`
    without `import pytest`.
    """

    # Python builtins that don't need to be imported
    BUILTINS = set(dir(builtins))

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """
        Validate that all used names are defined or imported.

        Args:
            artifact: The code artifact to validate
            **_deps: Ignored dependencies

        Returns:
            GuardResult with pass/fail and feedback
        """
        code = artifact.content
        if not code:
            return GuardResult(passed=False, feedback="No code provided")

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return GuardResult(passed=False, feedback=f"Syntax error: {e}")

        defined = self._collect_defined_names(tree)
        used = self._collect_used_names(tree)

        # Names that are used but not defined, imported, or builtin
        undefined = used - defined - self.BUILTINS

        if undefined:
            sorted_names = ", ".join(sorted(undefined))
            return GuardResult(
                passed=False,
                feedback=f"Undefined names (missing imports?): {sorted_names}",
            )
        return GuardResult(passed=True, feedback="All imports valid")

    def _collect_defined_names(self, tree: ast.AST) -> set[str]:
        """
        Collect all names defined in the code.

        Includes:
        - Import statements (import X, from X import Y)
        - Function definitions (def foo)
        - Class definitions (class Bar)
        - Assignments (x = ...)
        - For loop variables (for x in ...)
        - With statement variables (with ... as x)
        - Exception handlers (except E as e)
        - Comprehension variables
        """
        defined: set[str] = set()

        for node in ast.walk(tree):
            # Import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split(".")[0]
                    defined.add(name)

            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    if name != "*":  # from X import * doesn't define specific names
                        defined.add(name)

            # Function and class definitions
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                defined.add(node.name)
                # Add parameter names
                for arg in node.args.args:
                    defined.add(arg.arg)
                for arg in node.args.posonlyargs:
                    defined.add(arg.arg)
                for arg in node.args.kwonlyargs:
                    defined.add(arg.arg)
                if node.args.vararg:
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined.add(node.args.kwarg.arg)

            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)

            # Assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    defined.update(self._extract_names_from_target(target))

            elif isinstance(node, ast.AnnAssign):
                if node.target:
                    defined.update(self._extract_names_from_target(node.target))

            elif isinstance(node, ast.AugAssign):
                defined.update(self._extract_names_from_target(node.target))

            elif isinstance(node, ast.NamedExpr):  # Walrus operator :=
                defined.add(node.target.id)

            # For loop variables
            elif isinstance(node, ast.For):
                defined.update(self._extract_names_from_target(node.target))

            # With statement variables
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars:
                        defined.update(
                            self._extract_names_from_target(item.optional_vars)
                        )

            # Exception handlers
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    defined.add(node.name)

            # Comprehension variables
            elif isinstance(node, ast.comprehension):
                defined.update(self._extract_names_from_target(node.target))

        return defined

    def _extract_names_from_target(self, target: ast.AST) -> set[str]:
        """Extract variable names from an assignment target."""
        names: set[str] = set()

        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                names.update(self._extract_names_from_target(elt))
        elif isinstance(target, ast.Starred):
            names.update(self._extract_names_from_target(target.value))
        # ast.Attribute and ast.Subscript don't define new names

        return names

    def _collect_used_names(self, tree: ast.AST) -> set[str]:
        """
        Collect all names used (referenced) in the code.

        Only collects top-level names (e.g., `pytest` from `pytest.raises()`).
        """
        used: set[str] = set()

        for node in ast.walk(tree):
            # Check if this is a Name node in Load context (being read, not assigned)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)

        return used
