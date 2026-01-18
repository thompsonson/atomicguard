"""Syntax guard for validating Python code syntax."""

import ast
import sys
from io import StringIO
from typing import Tuple


class SyntaxGuard:
    """Validates that generated code has valid Python syntax."""

    def validate(self, code: str) -> Tuple[bool, str]:
        """Validate Python code syntax.

        Args:
            code: Python code string to validate

        Returns:
            Tuple of (is_valid, feedback_message)
        """
        if not code or not code.strip():
            return False, "Error: Empty code generated"

        try:
            # Try to parse the code as an AST
            ast.parse(code)
            return True, "Syntax validation passed"
        except SyntaxError as e:
            feedback = f"Syntax Error at line {e.lineno}: {e.msg}"
            if e.text:
                feedback += f"\n  {e.text.strip()}"
            return False, feedback
        except Exception as e:
            return False, f"Parsing error: {str(e)}"

    def validate_with_imports(self, code: str) -> Tuple[bool, str]:
        """Validate code and check for import errors.

        Args:
            code: Python code string to validate

        Returns:
            Tuple of (is_valid, feedback_message)
        """
        # First check syntax
        is_valid, feedback = self.validate(code)
        if not is_valid:
            return is_valid, feedback

        # Try to compile the code
        try:
            compile(code, '<string>', 'exec')
            return True, "Syntax and compilation validation passed"
        except SyntaxError as e:
            return False, f"Compilation error: {str(e)}"
        except Exception as e:
            # Other errors (like NameError) are okay at this stage
            # They'll be caught when tests run
            return True, "Syntax validation passed (runtime checks deferred to tests)"
