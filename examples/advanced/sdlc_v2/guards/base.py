"""
Base classes and mixins for guards.

Provides common functionality for guards that need to run validation
tools in isolated temporary environments.
"""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..models import ImplementationResult, TestSuite


class TempDirValidationMixin:
    """
    Mixin providing temp directory setup for validation.

    Guards that need to run external tools (pytest, mypy, ruff) can use
    this mixin to write implementation files to a temporary directory,
    run validation, and have the directory cleaned up automatically.

    This maintains the "guards as sensing-only actions" principle by
    not modifying the actual workdir.
    """

    @contextmanager
    def _temp_implementation(self, content: str) -> Iterator[Path]:
        """
        Write implementation to temp dir, yield path, cleanup.

        Args:
            content: JSON string containing ImplementationResult

        Yields:
            Path to temp directory containing implementation files
        """
        impl = ImplementationResult.model_validate_json(content)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            for file in impl.files:
                path = tmppath / file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(file.content)

            yield tmppath

    @contextmanager
    def _temp_implementation_with_tests(
        self,
        impl_content: str,
        test_content: str,
        test_output_path: str = "tests/test_architecture.py",
    ) -> Iterator[Path]:
        """
        Write implementation and tests to temp dir, yield path, cleanup.

        Args:
            impl_content: JSON string containing ImplementationResult
            test_content: JSON string containing TestSuite
            test_output_path: Relative path for test file

        Yields:
            Path to temp directory containing implementation and test files
        """
        impl = ImplementationResult.model_validate_json(impl_content)
        tests = TestSuite.model_validate_json(test_content)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write implementation files
            for file in impl.files:
                path = tmppath / file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(file.content)

            # Assemble and write test file
            test_lines = self._assemble_test_file(tests)
            test_file = tmppath / test_output_path
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text(test_lines)

            yield tmppath

    def _assemble_test_file(self, tests: TestSuite) -> str:
        """
        Assemble TestSuite into a single runnable test file.

        Args:
            tests: TestSuite model containing test functions

        Returns:
            Complete Python test file content
        """
        lines: list[str] = []

        # Module docstring
        if tests.module_docstring:
            lines.append(f'"""{tests.module_docstring}"""')
            lines.append("")

        # Imports
        for imp in tests.imports:
            lines.append(imp)
        lines.append("")

        # Fixtures
        for fixture in tests.fixtures:
            lines.append(fixture)
            lines.append("")

        # Test functions
        for test in tests.tests:
            lines.append(test.test_code)
            lines.append("")

        return "\n".join(lines)
