"""FixApproachGenerator: Designs the fix strategy.

Synthesizes information from root cause analysis, context reading,
and localization to design a concrete fix approach.
"""

import json
import logging
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import ContextSummary, FixApproach, Localization, RootCause

logger = logging.getLogger("swe_bench_ablation.generators")


class FixApproachGenerator(PydanticAIGenerator[FixApproach]):
    """Generator that designs the fix strategy.

    Uses root cause, context, and localization to produce a detailed
    fix approach with ordered steps and edge case handling.
    """

    output_type = FixApproach

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for fix approach design."""
        parts = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Get root cause from dependencies
        root_cause = self._get_root_cause(context)
        if root_cause:
            parts.append("\n\n## Root Cause Analysis")
            parts.append(f"Cause type: {root_cause.cause_type}")
            parts.append(f"Description: {root_cause.cause_description}")
            if root_cause.triggering_conditions:
                parts.append(f"Triggers: {', '.join(root_cause.triggering_conditions)}")
            parts.append(f"Confidence: {root_cause.confidence}")

        # Get context from dependencies
        context_summary = self._get_context_summary(context)
        if context_summary:
            parts.append("\n\n## Code Context")
            parts.append(f"File: {context_summary.file_path}")
            if context_summary.relevant_functions:
                parts.append(f"Functions: {', '.join(context_summary.relevant_functions)}")
            parts.append(f"Summary: {context_summary.summary}")
            parts.append(f"\n```python\n{context_summary.code_snippet}\n```")

        # Get localization from dependencies
        localization = self._get_localization(context)
        if localization:
            parts.append("\n\n## Localization")
            parts.append(f"Files: {', '.join(localization.files)}")
            if localization.functions:
                func_names = [f"{f.file}:{f.name}" for f in localization.functions]
                parts.append(f"Functions: {', '.join(func_names)}")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

    def _get_root_cause(self, context: Context) -> RootCause | None:
        """Extract root cause from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "root_cause" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return RootCause.model_validate(data)
                    except Exception:
                        pass
        return None

    def _get_context_summary(self, context: Context) -> ContextSummary | None:
        """Extract context summary from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "context_read" in dep_id.lower() or "context" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return ContextSummary.model_validate(data)
                    except Exception:
                        pass
        return None

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
