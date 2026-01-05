"""
Ollama LLM generator implementation.

Connects to Ollama instances via the OpenAI-compatible API.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
)
from atomicguard.domain.prompts import PromptTemplate

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


@dataclass
class OllamaGeneratorConfig:
    """Configuration for OllamaGenerator.

    This typed config ensures unknown fields are rejected at construction time.
    """

    model: str = "qwen2.5-coder:7b"
    base_url: str = DEFAULT_OLLAMA_URL
    timeout: float = 120.0


class OllamaGenerator(GeneratorInterface):
    """Connects to Ollama instance using OpenAI-compatible API."""

    config_class = OllamaGeneratorConfig

    def __init__(self, config: OllamaGeneratorConfig | None = None, **kwargs: Any):
        """
        Args:
            config: Typed configuration object (preferred)
            **kwargs: Legacy kwargs for backward compatibility (deprecated)
        """
        # Support both config object and legacy kwargs
        if config is None:
            config = OllamaGeneratorConfig(**kwargs)

        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("openai library required: pip install openai") from err

        self._model = config.model
        self._client = OpenAI(
            base_url=config.base_url,
            api_key="ollama",  # required but unused
            timeout=config.timeout,
        )
        self._version_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Generate an artifact based on context."""
        # Build prompt
        if template:
            prompt = template.render(context)
        else:
            prompt = self._build_basic_prompt(context)

        # Call Ollama
        messages = [
            {
                "role": "system",
                "content": "You are a Python programming assistant. Provide complete, runnable code in a markdown block:\n```python\n# code\n```",
            },
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._model, messages=cast(Any, messages), temperature=0.7
        )

        content = response.choices[0].message.content or ""
        code = self._extract_code(content)

        self._version_counter += 1

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            content=code,
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=self._version_counter,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=ContextSnapshot(
                workflow_id=workflow_id,
                specification=context.specification,
                constraints=context.ambient.constraints,
                feedback_history=(),
                dependency_artifacts=context.dependency_artifacts,
            ),
        )

    def _extract_code(self, content: str) -> str:
        """Extract Python code from response."""
        if not content or content.isspace():
            return ""

        # Try python block
        match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        if match:
            return match.group(1)

        # Try generic block
        match = re.search(r"```\n(.*?)\n```", content, re.DOTALL)
        if match:
            return match.group(1)

        # Try first def/import/class
        match = re.search(r"^(def |import |class )", content, re.MULTILINE)
        if match:
            return content[match.start() :]

        # No code block found - return empty to trigger guard validation failure
        return ""

    def _build_basic_prompt(self, context: Context) -> str:
        """Build a basic prompt from context."""
        parts = [context.specification]

        if context.current_artifact:
            parts.append(f"\nPrevious attempt:\n{context.current_artifact}")

        if context.feedback_history:
            feedback_text = "\n".join(
                f"Attempt {i + 1} feedback: {f}"
                for i, (_, f) in enumerate(context.feedback_history)
            )
            parts.append(f"\nFeedback history:\n{feedback_text}")

        return "\n".join(parts)
