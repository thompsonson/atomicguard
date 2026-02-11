"""WorkflowGenerator: Generates per-instance workflow specifications.

Part of the meta-level pipeline in Arms 20-21. Takes the problem
classification from ap_classify_problem and generates a workflow JSON
specification using the same format as static workflow configs. The
generated workflow is then executed by the object-level orchestrator.

Used by: ap_generate_workflow in Arms 20, 21
"""

import json
import logging
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import GeneratedWorkflow, ProblemClassification

logger = logging.getLogger("swe_bench_ablation.generators")


class WorkflowGenerator(PydanticAIGenerator[GeneratedWorkflow]):
    """Generator that produces workflow specifications from problem classifications.

    Reads the classification from ap_classify_problem and generates a
    workflow JSON matching the static workflow config schema. The component
    registry (available generators and guards) is injected via the prompt
    to constrain the LLM to only reference existing components.
    """

    output_type = GeneratedWorkflow

    def __init__(
        self,
        *,
        component_registry: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)
        self._component_registry = component_registry or {
            "generators": [
                "AnalysisGenerator",
                "LocalizationGenerator",
                "PatchGenerator",
                "TestGenerator",
                "DiffReviewGenerator",
            ],
            "guards": [
                "analysis",
                "localization",
                "patch",
                "test_syntax",
                "test_red",
                "test_green",
                "full_eval",
                "review_schema",
                "composite",
            ],
        }

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for workflow generation."""
        parts = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Get classification from dependencies
        classification = self._get_classification(context)
        if classification:
            parts.append("\n\n## Problem Classification")
            parts.append(f"Category: {classification.category.value}")
            parts.append(
                f"Estimated complexity: {classification.estimated_complexity}/5"
            )
            parts.append(f"Reasoning: {classification.reasoning}")

        # Inject component registry so the LLM knows what's available
        parts.append("\n\n## Available Components (Component Registry)")
        parts.append(
            "You may ONLY reference these generators and guards in your workflow.\n"
        )
        parts.append(
            f"Generators: {', '.join(self._component_registry.get('generators', []))}"
        )
        parts.append(f"Guards: {', '.join(self._component_registry.get('guards', []))}")

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
            if "classify" in dep_id.lower() or "classification" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return ProblemClassification.model_validate(data)
                    except Exception:
                        pass
        return None
