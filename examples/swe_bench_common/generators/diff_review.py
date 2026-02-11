"""DiffReviewGenerator: LLM-as-reviewer that critiques generated patches.

Reviews the full context (analysis + test + patch) and produces a structured
critique with verdict, issues, and a backtrack_target heuristic for the
BacktrackOrchestrator (Arm 18).

Used by: ap_diff_review in Arms 17, 18
"""

import json
import logging
from typing import Any

from examples.base.generators import PydanticAIGenerator

from atomicguard.domain.models import Context
from atomicguard.domain.prompts import PromptTemplate

from examples.swe_bench_common.models import Analysis, DiffReview

logger = logging.getLogger("swe_bench_ablation.generators")


class DiffReviewGenerator(PydanticAIGenerator[DiffReview]):
    """Generator that produces structured code review of a patch.

    Reads analysis, test code, and patch from dependency artifacts.
    Outputs a verdict (approve/revise/backtrack) with a backtrack_target
    field that the BacktrackOrchestrator uses as a search heuristic.
    """

    output_type = DiffReview

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.2)
        super().__init__(**kwargs)

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for diff review."""
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
            parts.append(f"Files: {', '.join(analysis.files)}")
            parts.append(f"Fix approach: {analysis.fix_approach}")

        # Get test code from dependencies
        test_code = self._get_test_code(context)
        if test_code:
            parts.append(f"\n\n## Generated Test\n```\n{test_code}\n```")

        # Get patch from dependencies
        patch_content = self._get_patch(context)
        if patch_content:
            parts.append(f"\n\n## Generated Patch\n```diff\n{patch_content}\n```")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

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

    def _get_test_code(self, context: Context) -> str | None:
        """Extract test code from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "test" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact and artifact.content.strip():
                    return artifact.content
        return None

    def _get_patch(self, context: Context) -> str | None:
        """Extract patch content from dependency artifacts."""
        for dep_id, artifact_id in context.dependency_artifacts:
            if "patch" in dep_id.lower():
                artifact = context.ambient.repository.get_artifact(artifact_id)
                if artifact and artifact.content.strip():
                    try:
                        data = json.loads(artifact.content)
                        return data.get("patch", artifact.content)
                    except (json.JSONDecodeError, TypeError):
                        return artifact.content
        return None
