"""
File extraction service for materializing artifacts to filesystem.

This service extracts artifact contents to the filesystem AFTER workflow
completion. Guards should be sensing-only (validate without mutating state),
so file writing is handled here as a separate post-workflow step.

Usage:
    extractor = FileExtractor(workdir)
    paths = extractor.extract_implementation(artifact.content)
"""

import logging
from pathlib import Path

from ..models import ImplementationResult, TestSuite

logger = logging.getLogger("sdlc_checkpoint")


class FileExtractor:
    """
    Extracts artifact contents to the filesystem.

    This service is called AFTER workflow completion, not during guard
    validation. This maintains the principle that guards are sensing-only
    actions that do not mutate state.
    """

    def __init__(self, workdir: Path | str):
        """
        Initialize the file extractor.

        Args:
            workdir: Base directory for extracted files
        """
        self._workdir = Path(workdir)

    def extract_implementation(self, content: str) -> list[Path]:
        """
        Extract implementation files from g_coder artifact JSON.

        Args:
            content: JSON string containing ImplementationResult

        Returns:
            List of paths to written files
        """
        result = ImplementationResult.model_validate_json(content)
        written: list[Path] = []

        for file in result.files:
            path = self._workdir / file.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(file.content)
            written.append(path)
            logger.debug(f"[FileExtractor] Wrote: {path}")

        logger.info(f"[FileExtractor] Extracted {len(written)} implementation files")
        return written

    def extract_tests(
        self,
        content: str,
        output_path: str = "tests/test_architecture.py",
    ) -> Path:
        """
        Extract architecture tests from g_add artifact JSON.

        Assembles individual test functions into a single runnable test file.

        Args:
            content: JSON string containing TestSuite
            output_path: Relative path for output test file

        Returns:
            Path to written test file
        """
        tests = TestSuite.model_validate_json(content)

        # Assemble into single test file
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

        path = self._workdir / output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))

        logger.info(f"[FileExtractor] Wrote test file: {path}")
        return path

    def extract_all(
        self,
        coder_content: str,
        add_content: str | None = None,
    ) -> dict[str, list[Path]]:
        """
        Extract all artifacts from a completed workflow.

        Args:
            coder_content: JSON content from g_coder artifact
            add_content: Optional JSON content from g_add artifact

        Returns:
            Dict mapping artifact type to list of extracted paths
        """
        result: dict[str, list[Path]] = {}

        # Extract implementation
        impl_paths = self.extract_implementation(coder_content)
        result["implementation"] = impl_paths

        # Extract tests if provided
        if add_content:
            test_path = self.extract_tests(add_content)
            result["tests"] = [test_path]

        return result
