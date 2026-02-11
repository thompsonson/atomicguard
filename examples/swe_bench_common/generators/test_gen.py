"""TestGenerator: Generates failing test code that reproduces a bug.

Uses PydanticAI structured output to produce test code via the
GeneratedTest model.  The artifact content is the raw test code
string (not JSON) so that downstream guards and generators can
consume it directly.
"""

import json
import logging
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import Analysis, GeneratedTest

logger = logging.getLogger("swe_bench_ablation.generators")


class TestGenerator(PydanticAIGenerator[GeneratedTest]):
    """Generator that produces failing test code to reproduce a bug.

    Reads analysis from prior step via dependency artifacts.
    Outputs raw test code (not JSON) stored as Artifact.content
    so that TestSyntaxGuard and PatchGenerator can consume it directly.
    """

    output_type = GeneratedTest

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.3)
        super().__init__(**kwargs)

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for test generation."""
        parts = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Get analysis from dependencies
        analysis = self._get_analysis(context)
        if analysis:
            parts.append("\n\n## Bug Analysis")
            parts.append(f"Bug type: {analysis.bug_type.value}")
            parts.append(f"Root cause: {analysis.root_cause_hypothesis}")
            parts.append(
                f"Affected components: {', '.join(analysis.affected_components)}"
            )
            parts.append(f"Files: {', '.join(analysis.files)}")
            parts.append(f"Fix approach: {analysis.fix_approach}")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

    def _process_output(self, output: GeneratedTest, context: Context) -> str:
        """Return raw test code so guards and downstream generators can consume it directly."""
        return output.test_code

    def _get_analysis(self, context: Context) -> Analysis | None:
        """Extract analysis from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "analysis" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return Analysis.model_validate(data)
                    except Exception:
                        pass
        return None
