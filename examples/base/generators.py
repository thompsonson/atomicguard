"""PydanticAI-based generators for structured LLM output.

Provides base classes that use PydanticAI for structured output parsing
with retries=0, allowing AtomicGuard's feedback loop to handle retries.

Design Decision (from docs/design/decisions/decisions.md):
- PydanticAI for structured output: Add as dependency
- PydanticAI retries: Disabled (retries=0) - Merge with AtomicGuard rmax
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    FeedbackEntry,
)
from atomicguard.domain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Type variable for the output model
OutputT = TypeVar("OutputT", bound=BaseModel)


class PydanticAIGenerator(GeneratorInterface, Generic[OutputT]):
    """Base generator using PydanticAI for structured output.

    Uses PydanticAI Agent with retries=0, allowing AtomicGuard's
    Dual-State Action Pair feedback loop to handle retries with
    constructive validation errors.

    Subclasses must implement:
    - output_type: The Pydantic model class for structured output
    - _build_prompt: Construct the prompt from context

    Example:
        class PatchGenerator(PydanticAIGenerator[Patch]):
            output_type = Patch

            def _build_prompt(self, context, template, repo_root):
                return f"Fix this bug: {context.specification}"
    """

    # Subclasses must set this to their Pydantic output model
    output_type: type[OutputT]

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        timeout: float = 180.0,
        temperature: float = 0.2,
        **kwargs: Any,
    ):
        """Initialize the PydanticAI-based generator.

        Args:
            model: Model name (e.g., "qwen2.5-coder:14b" for Ollama,
                "Qwen/Qwen2.5-Coder-32B-Instruct" for HuggingFace)
            base_url: API base URL (Ollama or OpenAI-compatible).
                Ignored for HuggingFace — the provider handles routing.
            api_key: API key (use "ollama" for local Ollama, "hf_..."
                for HuggingFace Inference Providers)
            timeout: Request timeout in seconds
            temperature: Sampling temperature for generation
            **kwargs: Additional config passed to subclasses
        """
        try:
            from pydantic_ai import Agent
        except ImportError as err:
            raise ImportError(
                "pydantic-ai is required: pip install pydantic-ai"
            ) from err

        self._model_name = model
        self._base_url = base_url
        self._timeout = timeout
        self._temperature = temperature

        # Create the PydanticAI model
        # Determine provider: Ollama, HuggingFace, or generic OpenAI-compatible
        if "ollama" in base_url.lower() or api_key == "ollama":
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.ollama import OllamaProvider

            provider = OllamaProvider(base_url=base_url)
            ai_model = OpenAIChatModel(
                model_name=model,
                provider=provider,
            )
        elif api_key.startswith("hf_") or "huggingface" in base_url.lower():
            # HuggingFace Inference Providers — uses native AsyncInferenceClient
            # which properly supports structured output with tool calling.
            from pydantic_ai.models.huggingface import HuggingFaceModel
            from pydantic_ai.providers.huggingface import HuggingFaceProvider

            ai_model = HuggingFaceModel(
                model,
                provider=HuggingFaceProvider(api_key=api_key),
            )
        else:
            # Generic OpenAI-compatible APIs
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            provider = OpenAIProvider(base_url=base_url, api_key=api_key)
            ai_model = OpenAIChatModel(
                model_name=model,
                provider=provider,
            )

        # Create Agent with retries=0 - AtomicGuard handles retries
        self._agent: Agent[None, OutputT] = Agent(
            ai_model,
            output_type=self.output_type,
            retries=0,  # Let AtomicGuard feedback loop handle retries
        )

        self._attempt_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
        workflow_ref: str | None = None,
    ) -> Artifact:
        """Generate an artifact using PydanticAI structured output.

        Args:
            context: Execution context with specification and dependencies
            template: Optional prompt template
            action_pair_id: Identifier for this action pair
            workflow_id: UUID of the workflow execution
            workflow_ref: Content-addressed workflow hash

        Returns:
            Artifact containing either validated output JSON or error
        """
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        logger.info("[%s] Generating with PydanticAI...", self.__class__.__name__)

        # Build the prompt
        prompt = self._build_prompt(context, template)

        # Get system prompt from template or use default
        system_prompt = self._get_system_prompt(template)

        metadata: dict[str, Any] = {}

        try:
            from pydantic_ai.settings import ModelSettings

            # Run PydanticAI agent with structured output
            result = self._agent.run_sync(
                prompt,
                instructions=system_prompt,
                model_settings=ModelSettings(temperature=self._temperature),
            )

            # Success - PydanticAI validated the output
            output_data = result.output
            content = self._process_output(output_data, context)

            # Capture usage if available
            if hasattr(result, "usage") and result.usage:
                usage = result.usage()
                metadata = {
                    "prompt_tokens": usage.request_tokens or 0,
                    "completion_tokens": usage.response_tokens or 0,
                    "total_tokens": usage.total_tokens or 0,
                }

            logger.debug(
                "[%s] Successfully parsed structured output",
                self.__class__.__name__,
            )

        except UnexpectedModelBehavior as e:
            # Validation failed - return constructive error for retry
            error_msg = self._format_validation_error(e)
            content = json.dumps({"error": error_msg})
            logger.warning(
                "[%s] Validation failed: %s",
                self.__class__.__name__,
                error_msg,
            )

        except Exception as e:
            # Other errors (network, timeout, etc.)
            content = json.dumps({"error": f"Generation failed: {e}"})
            logger.warning(
                "[%s] Generation failed: %s",
                self.__class__.__name__,
                e,
            )

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
            content=content,
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

    @abstractmethod
    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate | None,
    ) -> str:
        """Build the prompt for the LLM.

        Subclasses must implement this to construct the prompt
        from the context and template.

        Args:
            context: Execution context
            template: Optional prompt template

        Returns:
            The prompt string to send to the LLM
        """
        pass

    def _get_system_prompt(self, template: PromptTemplate | None) -> str:
        """Get the system prompt from template or default.

        Args:
            template: Optional prompt template

        Returns:
            System prompt string
        """
        if template and template.role:
            return template.role
        return "You are a helpful assistant."

    def _process_output(self, output: OutputT, context: Context) -> str:
        """Process the validated output into artifact content.

        Default implementation serializes to JSON. Subclasses can
        override for custom processing (e.g., generating unified diffs).

        Args:
            output: The validated Pydantic model instance
            context: Execution context

        Returns:
            JSON string for artifact content
        """
        return output.model_dump_json(indent=2)

    def _format_validation_error(self, error: Exception) -> str:
        """Format a validation error into constructive feedback.

        Extracts useful information from PydanticAI validation errors
        to help the LLM correct its output on retry.

        Args:
            error: The validation exception

        Returns:
            Constructive error message for retry feedback
        """
        error_str = str(error)

        # Try to extract the most useful part of the error
        if "validation error" in error_str.lower():
            return f"Output validation failed: {error_str}"

        if "json" in error_str.lower():
            return (
                f"Invalid JSON in response: {error_str}. "
                "Ensure all strings are properly escaped (use \\\" for quotes, "
                "\\n for newlines) and the JSON structure matches the schema."
            )

        return f"Unexpected output format: {error_str}"
