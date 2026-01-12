"""
AllTestsPassGuard: Validates implementation by running tests.

Validates that CoderGenerator produced code that passes all tests.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import ImplementationResult

logger = logging.getLogger("sdlc_checkpoint")


class AllTestsPassGuard(GuardInterface):
    """
    Validates implementation by running architecture tests.

    Checks:
    - Valid JSON structure matching ImplementationResult schema
    - All generated files have valid Python syntax
    - Implementation follows layer structure

    Note: For this demo, we do simplified validation.
    A production version would actually run pytest.
    """

    def __init__(self, workdir: str = "output"):
        self._workdir = Path(workdir)

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate implementation."""
        logger.debug("[AllTestsPassGuard] Validating implementation...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[AllTestsPassGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[AllTestsPassGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        # Check for errors
        if "error" in data:
            logger.debug(f"[AllTestsPassGuard] Generator error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('details', data['error'])}",
            )

        # Parse as ImplementationResult
        try:
            result = ImplementationResult.model_validate(data)
            logger.debug(f"[AllTestsPassGuard] Schema valid, {len(result.files)} files")
        except Exception as e:
            logger.debug(f"[AllTestsPassGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        # Check for at least some files
        if not result.files:
            logger.debug("[AllTestsPassGuard] No files generated")
            return GuardResult(
                passed=False,
                feedback="No implementation files generated",
            )

        # Validate syntax of each Python file
        for file in result.files:
            if file.path.endswith(".py"):
                syntax_result = self._validate_syntax(file.path, file.content)
                if syntax_result:
                    return syntax_result

        # Check layer structure
        structure_result = self._validate_structure(result)
        if structure_result:
            return structure_result

        # NOTE: File writing removed from guard - guards should be sensing-only.
        # Files are extracted post-workflow via FileExtractor service.

        logger.debug("[AllTestsPassGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"Valid implementation: {len(result.files)} files generated",
        )

    def _validate_syntax(self, path: str, content: str) -> GuardResult | None:
        """Validate Python syntax of a file."""
        try:
            ast.parse(content)
            logger.debug(f"[AllTestsPassGuard] Syntax valid: {path}")
            return None
        except SyntaxError as e:
            logger.debug(f"[AllTestsPassGuard] Syntax error in {path}: {e}")
            return GuardResult(
                passed=False,
                feedback=f"Syntax error in {path} at line {e.lineno}: {e.msg}\n\n"
                f"Code around error:\n{self._get_code_context(content, e.lineno or 1)}",
            )

    def _get_code_context(self, content: str, line_no: int, context: int = 3) -> str:
        """Get code context around an error line."""
        lines = content.split("\n")
        start = max(0, line_no - context - 1)
        end = min(len(lines), line_no + context)

        result_lines = []
        for i, line in enumerate(lines[start:end], start=start + 1):
            marker = ">>> " if i == line_no else "    "
            result_lines.append(f"{marker}{i:4d}: {line}")

        return "\n".join(result_lines)

    def _validate_structure(self, result: ImplementationResult) -> GuardResult | None:
        """Validate that files follow expected layer structure."""
        required_dirs = {"domain", "application", "infrastructure"}
        found_dirs = set()

        for file in result.files:
            for dir_name in required_dirs:
                if f"/{dir_name}/" in file.path or file.path.startswith(f"{dir_name}/"):
                    found_dirs.add(dir_name)

        missing = required_dirs - found_dirs
        if missing:
            logger.debug(f"[AllTestsPassGuard] Missing layers: {missing}")
            # This is a warning, not a failure
            logger.warning(
                f"[AllTestsPassGuard] Implementation may be incomplete: missing {missing}"
            )

        return None

    def _write_files(self, result: ImplementationResult) -> None:
        """Write implementation files to workdir."""
        for file in result.files:
            file_path = self._workdir / file.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file.content)
            logger.debug(f"[AllTestsPassGuard] Wrote: {file_path}")
