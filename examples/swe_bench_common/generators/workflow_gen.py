"""WorkflowGenerator: Generates per-instance workflow specifications.

Part of the meta-level pipeline in Arms 20-21. Takes the problem
classification from ap_classify_problem and generates a workflow JSON
specification using the same format as static workflow configs. The
generated workflow is then executed by the object-level orchestrator.

Used by: ap_generate_workflow in Arms 20, 21
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import GeneratedWorkflow


class WorkflowGenerator(PydanticAIGenerator[GeneratedWorkflow]):
    """Generator that produces workflow specifications from problem classifications.

    Reads the classification from ap_classify_problem and generates a
    workflow JSON matching the static workflow config schema. The component
    registry (available generators and guards) is injected via the prompt
    to constrain the LLM to only reference existing components.

    Context comes from prompt templates via {ap_classify_problem} placeholder.
    The component registry is injected via _build_prompt for custom logic.
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
        template: PromptTemplate,
    ) -> str:
        """Build the prompt for workflow generation.

        Extends base template rendering by injecting the component registry
        so the LLM knows what generators and guards are available.
        """
        # Get base prompt from template
        base_prompt = template.render(context)

        # Inject component registry
        registry_section = (
            "\n\n## Available Components (Component Registry)\n"
            "You may ONLY reference these generators and guards in your workflow.\n\n"
            f"Generators: {', '.join(self._component_registry.get('generators', []))}\n"
            f"Guards: {', '.join(self._component_registry.get('guards', []))}"
        )

        return base_prompt + registry_section
