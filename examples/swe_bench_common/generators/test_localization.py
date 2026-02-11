"""TestLocalizationGenerator: Locates existing test files and patterns.

Identifies test files, patterns, and fixtures related to the buggy code
to guide test generation in the appropriate style.
"""

import json
import logging
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import Localization, ProjectStructure, TestLocalization

logger = logging.getLogger("swe_bench_ablation.generators")


class TestLocalizationGenerator(PydanticAIGenerator[TestLocalization]):
    """Generator that locates existing test files and patterns.

    Uses the localization from ap_localise_issue and structure from
    ap_structure to identify relevant test files and patterns.
    """

    output_type = TestLocalization

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.1)
        super().__init__(**kwargs)

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for test localization."""
        parts = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Get localization from dependencies
        localization = self._get_localization(context)
        if localization:
            parts.append("\n\n## Issue Localization")
            parts.append(f"Source files: {', '.join(localization.files)}")
            if localization.functions:
                func_names = [f.name for f in localization.functions]
                parts.append(f"Functions: {', '.join(func_names)}")

        # Get structure from dependencies
        structure = self._get_structure(context)
        if structure:
            parts.append("\n\n## Project Structure")
            parts.append(f"Test framework: {structure.test_framework}")
            if structure.test_directories:
                parts.append(f"Test directories: {', '.join(structure.test_directories)}")
            if structure.root_modules:
                parts.append(f"Root modules: {', '.join(structure.root_modules)}")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

    def _get_localization(self, context: Context) -> Localization | None:
        """Extract localization from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "localise_issue" in dep_id.lower() or "localize" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return Localization.model_validate(data)
                    except Exception:
                        pass
        return None

    def _get_structure(self, context: Context) -> ProjectStructure | None:
        """Extract structure from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "structure" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return ProjectStructure.model_validate(data)
                    except Exception:
                        pass
        return None
