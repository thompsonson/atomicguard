"""
HuggingFace Inference API generator implementation.

Connects to HuggingFace Inference Providers via the huggingface_hub
InferenceClient for chat completion.
"""

import os
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


@dataclass
class HuggingFaceGeneratorConfig:
    """Configuration for HuggingFaceGenerator.

    This typed config ensures unknown fields are rejected at construction time.
    """

    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    api_key: str | None = None  # Auto-detects from HF_TOKEN env var
    provider: str | None = None  # e.g. "auto", "hf-inference", "together"
    timeout: float = 120.0
    temperature: float = 0.7
    max_tokens: int = 4096


class HuggingFaceGenerator(GeneratorInterface):
    """Connects to HuggingFace Inference API using huggingface_hub."""

    config_class = HuggingFaceGeneratorConfig

    def __init__(
        self, config: HuggingFaceGeneratorConfig | None = None, **kwargs: Any
    ) -> None:
        """
        Args:
            config: Typed configuration object (preferred)
            **kwargs: Legacy kwargs for backward compatibility (deprecated)
        """
        if config is None:
            config = HuggingFaceGeneratorConfig(**kwargs)

        try:
            from huggingface_hub import InferenceClient
        except ImportError as err:
            raise ImportError(
                "huggingface_hub library required: pip install huggingface_hub"
            ) from err

        api_key = config.api_key
        if api_key is None:
            api_key = os.environ.get("HF_TOKEN")
            if not api_key:
                raise ValueError(
                    "HuggingFace API key required: set HF_TOKEN environment "
                    "variable or pass api_key in config"
                )

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config.timeout,
        }
        if config.provider is not None:
            client_kwargs["provider"] = config.provider

        self._model = config.model
        self._client = InferenceClient(**client_kwargs)
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens
        self._version_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Generate an artifact based on context."""
        if template:
            prompt = template.render(context)
        else:
            prompt = self._build_basic_prompt(context)

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a Python programming assistant. "
                    "Provide complete, runnable code in a markdown block:\n"
                    "```python\n# code\n```"
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat_completion(
            messages=cast(Any, messages),
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
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
