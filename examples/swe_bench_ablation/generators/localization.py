"""LocalizationGenerator: Identifies files and functions to modify.

Uses an LLM to analyze the problem statement and identify which files
and functions need modification to fix the bug.
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

from ..models import Localization

logger = logging.getLogger("swe_bench_ablation.generators")


class LocalizationGenerator(GeneratorInterface):
    """Generator that identifies files and functions to modify.

    Uses an LLM to analyze the problem statement and identify the most
    likely locations that need modification to fix the bug.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        timeout: float = 120.0,
    ):
        """Initialize the localization generator.

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
        action_pair_id: str = "ap_localize",
        workflow_id: str = "unknown",
        workflow_ref: str | None = None,
    ) -> Artifact:
        """Generate localization identifying files and functions.

        Args:
            context: Execution context with specification
            template: Prompt template from prompts.json
            action_pair_id: Identifier for this action pair
            workflow_id: UUID of the workflow execution
            workflow_ref: Content-addressed workflow hash

        Returns:
            Artifact containing Localization JSON
        """
        logger.info("[LocalizationGenerator] Identifying bug locations...")

        # Build prompt
        prompt = self._build_prompt(context, template)

        # Get system prompt from template or use default
        system_prompt = (
            template.role
            if template
            else "You are a code search specialist identifying bug locations."
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
                temperature=0.2,
            )
            content = response.choices[0].message.content or ""
            logger.debug(f"[LocalizationGenerator] Response: {content[:200]}...")

            if hasattr(response, "usage") and response.usage:
                metadata = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            result = self._parse_response(content)

        except Exception as e:
            logger.warning(f"[LocalizationGenerator] LLM call failed: {e}")
            result = json.dumps({"error": str(e)})

        self._attempt_counter += 1

        # Build context snapshot
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
        """Build the prompt for localization."""
        parts = []

        # Task from template
        if template:
            parts.append(template.task)

        # Problem statement
        parts.append(f"\n\n## Problem Statement\n{context.specification}")

        # Constraints from template
        if template and template.constraints:
            parts.append(f"\n\n## Constraints\n{template.constraints}")

        # Output format instruction
        parts.append(
            """

## Output Format
Return a JSON object with this structure:
```json
{
  "files": ["path/to/file1.py", "path/to/file2.py"],
  "functions": [
    {"name": "function_name", "file": "path/to/file.py", "line": null}
  ],
  "reasoning": "Brief explanation of why these locations were identified"
}
```

IMPORTANT:
- Maximum 5 files
- File paths should be relative to repository root
- Function line numbers can be null if unknown
"""
        )

        # Add feedback if retry
        if context.feedback_history and template:
            latest_feedback = context.feedback_history[-1][1]
            parts.append(
                f"\n\n## Previous Attempt Rejected\n{template.feedback_wrapper.format(feedback=latest_feedback)}"
            )

        return "\n".join(parts)

    def _parse_response(self, content: str) -> str:
        """Parse LLM response into Localization JSON."""
        # Extract JSON from response
        json_match = re.search(r"```json?\s*([\s\S]*?)```", content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            obj_match = re.search(r"\{[\s\S]*\}", content)
            if obj_match:
                json_str = obj_match.group(0)
            else:
                return json.dumps({"error": "No JSON found in response"})

        try:
            data = json.loads(json_str)

            # Validate with Pydantic
            localization = Localization.model_validate(data)
            return localization.model_dump_json(indent=2)

        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})
        except Exception as e:
            return json.dumps({"error": f"Validation failed: {e}"})
