"""ImpactAnalysisGenerator: Analyzes the impact of a proposed fix.

Evaluates how the fix approach will affect other code and tests,
identifying potential regressions and API changes.
"""

import json
import logging
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import FixApproach, ImpactAnalysis

logger = logging.getLogger("swe_bench_ablation.generators")


class ImpactAnalysisGenerator(PydanticAIGenerator[ImpactAnalysis]):
    """Generator that analyzes the impact of a proposed fix.

    Uses the fix approach to identify affected tests, functions,
    potential regressions, and API changes.
    """

    output_type = ImpactAnalysis

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for impact analysis."""
        parts = []

        if template:
            parts.append(template.task)

        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Get fix approach from dependencies
        fix_approach = self._get_fix_approach(context)
        if fix_approach:
            parts.append("\n\n## Proposed Fix Approach")
            parts.append(f"Summary: {fix_approach.approach_summary}")
            if fix_approach.steps:
                parts.append("Steps:")
                for i, step in enumerate(fix_approach.steps, 1):
                    parts.append(f"  {i}. {step}")
            if fix_approach.files_to_modify:
                parts.append(f"Files to modify: {', '.join(fix_approach.files_to_modify)}")
            if fix_approach.functions_to_modify:
                parts.append(f"Functions to modify: {', '.join(fix_approach.functions_to_modify)}")
            if fix_approach.edge_cases:
                parts.append(f"Edge cases: {', '.join(fix_approach.edge_cases)}")
            parts.append(f"Reasoning: {fix_approach.reasoning}")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

    def _get_fix_approach(self, context: Context) -> FixApproach | None:
        """Extract fix approach from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "fix_approach" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact:
                    try:
                        data = json.loads(artifact.content)
                        return FixApproach.model_validate(data)
                    except Exception:
                        pass
        return None
