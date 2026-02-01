"""TestGenerator: Generates failing pytest-style tests that reproduce a bug.

Reads analysis from dependency artifacts and produces Python test code
that should fail on the buggy code and pass after the fix.
"""

import json
import logging
import re
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    FeedbackEntry,
)
from atomicguard.domain.prompts import PromptTemplate

from ..models import Analysis

logger = logging.getLogger("swe_bench_ablation.generators")


class TestGenerator(GeneratorInterface):
    """Generator that produces failing test code to reproduce a bug.

    Reads analysis from prior step via dependency artifacts.
    Outputs raw Python test code (not JSON) stored as Artifact.content.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        timeout: float = 120.0,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the test generator.

        Args:
            model: LLM model to use
            base_url: API base URL (Ollama or HuggingFace Inference API)
            api_key: API key (use HF_TOKEN for HuggingFace)
            timeout: Request timeout in seconds
        """
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("openai library required: pip install openai") from err

        self._model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._attempt_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,
        action_pair_id: str = "ap_gen_test",
        workflow_id: str = "unknown",
        workflow_ref: str | None = None,
    ) -> Artifact:
        """Generate failing test code.

        Args:
            context: Execution context with analysis dependency
            template: Prompt template from prompts.json
            action_pair_id: Identifier for this action pair
            workflow_id: UUID of the workflow execution
            workflow_ref: Content-addressed workflow hash

        Returns:
            Artifact containing Python test code
        """
        logger.info("[TestGenerator] Generating failing test...")

        prompt = self._build_prompt(context, template)

        system_prompt = (
            template.role
            if template
            else "You are a test engineer writing pytest-style tests."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        metadata: dict[str, Any] = {}
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.3,
            )
            content = response.choices[0].message.content or ""
            logger.debug("[TestGenerator] Response: %s...", content[:200])

            if hasattr(response, "usage") and response.usage:
                metadata = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            result = self._extract_code(content)

        except Exception as e:
            logger.warning("[TestGenerator] LLM call failed: %s", e)
            result = f"# Error: {e}"

        self._attempt_counter += 1

        context_snapshot = ContextSnapshot(
            workflow_id=workflow_id,
            specification=context.specification,
            constraints=context.ambient.constraints,
            feedback_history=tuple(
                FeedbackEntry(artifact_id=aid, feedback=fb)
                for aid, fb in context.feedback_history
            ),
            dependency_artifacts=context.dependency_artifacts,
        )

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=result,
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now(UTC).isoformat(),
            attempt_number=self._attempt_counter,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=context_snapshot,
            workflow_ref=workflow_ref,
            metadata=MappingProxyType(metadata),
        )

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
            parts.append(f"Likely files: {', '.join(analysis.likely_files)}")
            parts.append(f"Fix approach: {analysis.fix_approach}")

        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        parts.append(
            """

## Output Format
Return Python test code in a markdown code block:
```python
import pytest

def test_bug_reproduction():
    \"\"\"Test that reproduces the bug.\"\"\"
    # Your test code here
    ...
```

REQUIREMENTS:
- Use pytest style (test_ functions or Test classes)
- The test should FAIL on the current buggy code
- The test should PASS after the bug is fixed
- Import only from the project's own modules
- Be specific about the expected behavior
"""
        )

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

    def _extract_code(self, content: str) -> str:
        """Extract Python code from LLM response."""
        # Try python code block first
        match = re.search(r"```python\s*([\s\S]*?)```", content)
        if match:
            return match.group(1).strip()

        # Try generic code block
        match = re.search(r"```\s*([\s\S]*?)```", content)
        if match:
            return match.group(1).strip()

        # If content looks like code, return as-is
        if "def test_" in content or "class Test" in content:
            return content.strip()

        return f"# Could not extract test code from response\n# Raw: {content[:500]}"
