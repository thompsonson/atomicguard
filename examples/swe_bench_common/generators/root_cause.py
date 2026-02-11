"""RootCauseGenerator: Identifies the root cause of a bug based on classification.

Builds on the classification step to provide deeper analysis of why
the bug occurs and what conditions trigger it.
"""

import json
import logging
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import ProblemClassification, RootCause

logger = logging.getLogger("swe_bench_ablation.generators")


class RootCauseGenerator(PydanticAIGenerator[RootCause]):
    """Generator that identifies the root cause of a bug.

    Uses the classification from ap_classify to produce a detailed
    root cause analysis including triggering conditions and
    affected code paths.
    """

    output_type = RootCause

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for root cause analysis."""
        parts = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Get classification from dependencies
        classification = self._get_classification(context)
        if classification:
            parts.append("\n\n## Bug Classification")
            parts.append(f"Category: {classification.category.value}")
            parts.append(f"Estimated complexity: {classification.estimated_complexity}/5")
            parts.append(f"Classification reasoning: {classification.reasoning}")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

    def _get_classification(self, context: Context) -> ProblemClassification | None:
        """Extract classification from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "classify" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return ProblemClassification.model_validate(data)
                    except Exception:
                        pass
        return None
