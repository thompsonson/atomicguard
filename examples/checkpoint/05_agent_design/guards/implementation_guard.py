"""
ImplementationGuard: Validates generated implementation skeleton.

Validates Python syntax and file structure.
"""

import ast
import json
import logging
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult

from ..models import ImplementationResult

logger = logging.getLogger("agent_design")


class ImplementationGuard(GuardInterface):
    """
    Validates implementation generation (Step 7).

    Checks:
    - Valid JSON structure
    - Parses as ImplementationResult schema
    - workflow_config is valid
    - At least 1 workflow step
    - Required files present (models.py, generators, guards)
    - Python syntax valid for all .py files
    """

    def __init__(
        self,
        validate_syntax: bool = True,
        workdir: str = "output/generated",
        **_kwargs: Any,
    ):
        self.validate_syntax = validate_syntax
        self.workdir = workdir

    def validate(self, artifact: Artifact, **_deps: Any) -> GuardResult:
        """Validate implementation skeleton."""
        logger.debug("[ImplementationGuard] Validating implementation...")

        try:
            data = json.loads(artifact.content)
            logger.debug("[ImplementationGuard] Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.debug(f"[ImplementationGuard] Invalid JSON: {e}")
            return GuardResult(passed=False, feedback=f"Invalid JSON: {e}")

        if "error" in data:
            logger.debug(f"[ImplementationGuard] Generator error: {data.get('error')}")
            return GuardResult(
                passed=False,
                feedback=f"Generation error: {data.get('error')}",
            )

        # Parse as ImplementationResult
        try:
            impl = ImplementationResult.model_validate(data)
            logger.debug("[ImplementationGuard] Schema validation passed")
        except Exception as e:
            logger.debug(f"[ImplementationGuard] Schema invalid: {e}")
            return GuardResult(passed=False, feedback=f"Schema validation failed: {e}")

        issues = []

        # Validate workflow config
        if not impl.workflow_config:
            issues.append("workflow_config is empty")
        elif "action_pairs" not in impl.workflow_config:
            issues.append("workflow_config missing 'action_pairs' key")

        # Validate workflow steps
        if not impl.workflow_steps:
            issues.append("No workflow steps defined")

        # Check required files
        file_paths = {f.path for f in impl.files}
        required_patterns = ["models.py"]  # At minimum need models

        for pattern in required_patterns:
            if not any(pattern in path for path in file_paths):
                issues.append(f"Missing required file: {pattern}")

        # Check for at least one generator and guard
        has_generator = any("generator" in path.lower() for path in file_paths)
        has_guard = any("guard" in path.lower() for path in file_paths)

        if not has_generator:
            issues.append("No generator files found")
        if not has_guard:
            issues.append("No guard files found")

        # Validate Python syntax if enabled
        if self.validate_syntax:
            for file in impl.files:
                if file.file_type == "python" or file.path.endswith(".py"):
                    syntax_issues = self._validate_python_syntax(
                        file.path, file.content
                    )
                    issues.extend(syntax_issues)

        if issues:
            logger.debug(f"[ImplementationGuard] Validation failed: {issues}")
            return GuardResult(
                passed=False,
                feedback="Implementation issues:\n- " + "\n- ".join(issues),
            )

        logger.debug("[ImplementationGuard] ✓ All checks passed")
        return GuardResult(
            passed=True,
            feedback=f"Implementation valid: {len(impl.files)} files, "
            f"{len(impl.workflow_steps)} workflow steps",
        )

    def _validate_python_syntax(self, path: str, content: str) -> list[str]:
        """Validate Python syntax using ast.parse."""
        issues = []

        # Clean up content (remove markdown artifacts if present)
        clean_content = content
        if clean_content.startswith("```"):
            lines = clean_content.split("\n")
            # Remove first and last lines if they're markdown code blocks
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_content = "\n".join(lines)

        try:
            ast.parse(clean_content)
        except SyntaxError as e:
            # Provide context around the error
            lines = clean_content.split("\n")
            error_line = e.lineno or 1
            start = max(0, error_line - 3)
            end = min(len(lines), error_line + 2)
            context_lines = []
            for i in range(start, end):
                marker = ">>> " if i == error_line - 1 else "    "
                context_lines.append(f"{marker}{i + 1}: {lines[i]}")
            context = "\n".join(context_lines)

            issues.append(
                f"Python syntax error in {path} at line {e.lineno}: {e.msg}\n"
                f"Context:\n{context}"
            )

        return issues
