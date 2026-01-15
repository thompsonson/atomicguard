"""
Base generator class for Multi-Agent SDLC workflow.

Provides common functionality for LLM-based generators.
"""

from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from ..interfaces import GeneratorResult, IGenerator


class BaseGenerator(IGenerator):
    """Base class for LLM-based content generation.

    Responsibilities:
    - Initialize LLM client
    - Call LLM with prompts
    - Format responses

    Subclasses must implement:
    - generate() - specific generation logic
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1"):
        """Initialize generator with LLM configuration.

        Args:
            model: Model name (e.g., "qwen2.5-coder:14b")
            base_url: LLM API endpoint
        """
        self.model = model
        self.base_url = base_url
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't use real API keys
        )

    async def _call_llm(
        self, prompt: str, system: str | None = None, temperature: float = 0.7
    ) -> tuple[str, list[dict]]:
        """Call LLM and return response.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature

        Returns:
            Tuple of (content, raw_messages)
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        content = response.choices[0].message.content or ""
        raw_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content},
        ]

        return content, raw_messages

    async def generate(
        self, prompt: str, workspace: Path, context: dict[str, Any]
    ) -> GeneratorResult:
        """Generate content using LLM.

        Args:
            prompt: Instruction for the LLM
            workspace: Working directory
            context: Additional context

        Returns:
            GeneratorResult with content and metadata

        Note:
            This is a default implementation. Subclasses should override
            to implement specific generation logic.
        """
        raise NotImplementedError("Subclasses must implement generate()")
