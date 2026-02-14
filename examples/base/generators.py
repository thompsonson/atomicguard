"""PydanticAI-based generators for structured LLM output.

Provides base classes that use PydanticAI for structured output parsing
with retries=0, allowing AtomicGuard's feedback loop to handle retries.

Design Decision (from docs/design/decisions/decisions.md):
- PydanticAI for structured output: Add as dependency
- PydanticAI retries: Disabled (retries=0) - Merge with AtomicGuard rmax
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from examples.swe_bench_common.guards.escape_utils import detect_escape_issues
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


class PydanticAIGenerator[OutputT: BaseModel](GeneratorInterface):
    """Base generator using PydanticAI for structured output.

    Uses PydanticAI Agent with retries=0, allowing AtomicGuard's
    Dual-State Action Pair feedback loop to handle retries with
    constructive validation errors.

    Subclasses must define:
    - output_type: The Pydantic model class for structured output

    Optionally override:
    - _build_prompt: Construct the prompt from context (default uses template.render())

    Design principle: Prompt templates are the single source of truth for
    what context a generator needs. The default _build_prompt() uses
    template.render() which handles {ap_*} placeholder substitution
    automatically. Only override _build_prompt() if you need custom
    logic beyond what templates provide.

    Example (simple generator using template placeholders):
        class PatchGenerator(PydanticAIGenerator[Patch]):
            output_type = Patch
            # No _build_prompt override needed - template handles context

    Example (custom generator with special processing):
        class AnalysisGenerator(PydanticAIGenerator[Analysis]):
            output_type = Analysis

            def _build_prompt(self, context, template):
                # Custom logic that can't be expressed in templates
                return custom_prompt_logic(context)
    """

    # Subclasses must set this to their Pydantic output model
    output_type: type[OutputT]

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        provider: str = "ollama",
        timeout: float = 180.0,
        temperature: float = 0.2,
        output_mode: str = "tool",
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the PydanticAI-based generator.

        Args:
            model: Model name (e.g., "qwen2.5-coder:14b" for Ollama,
                "Qwen/Qwen2.5-Coder-32B-Instruct" for HuggingFace)
            base_url: API base URL (Ollama or OpenAI-compatible).
                Ignored for HuggingFace and OpenRouter — the provider
                handles routing.
            api_key: API key (use "ollama" for local Ollama)
            provider: LLM provider identifier. One of "ollama",
                "huggingface", "openrouter", "openai".
            timeout: Request timeout in seconds
            temperature: Sampling temperature for generation
            output_mode: Structured output mode. One of "tool" (default,
                uses function calling), "prompted" (injects schema into
                prompt, works with all models), or "native" (uses model's
                JSON schema response format).
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
        self._output_mode = output_mode

        # Create the PydanticAI model based on explicit provider selection
        ai_model = self._create_model(provider, model, base_url, api_key)

        # Wrap output_type with the appropriate mode marker
        effective_output_type = self._wrap_output_type(output_mode)

        # Create Agent with retries=0 - AtomicGuard handles retries
        self._agent: Agent[None, OutputT] = Agent(
            ai_model,
            output_type=effective_output_type,
            retries=0,  # Let AtomicGuard feedback loop handle retries
        )

        self._attempt_counter = 0

    def _wrap_output_type(self, output_mode: str) -> Any:
        """Wrap the output_type with the appropriate PydanticAI mode marker.

        Args:
            output_mode: One of "tool", "prompted", or "native".

        Returns:
            The output type, optionally wrapped with PromptedOutput or
            NativeOutput for models that don't support function calling.

        Raises:
            ValueError: If output_mode is not recognised.
        """
        if output_mode == "tool":
            # Default — uses function calling (ToolOutput). No wrapping
            # needed; PydanticAI uses tool output by default.
            return self.output_type

        if output_mode == "prompted":
            from pydantic_ai import PromptedOutput

            logger.info(
                "[%s] Using PromptedOutput mode (text-based JSON)",
                self.__class__.__name__,
            )
            return PromptedOutput(self.output_type)

        if output_mode == "native":
            from pydantic_ai import NativeOutput

            logger.info(
                "[%s] Using NativeOutput mode (model JSON schema)",
                self.__class__.__name__,
            )
            return NativeOutput(self.output_type)

        raise ValueError(
            f"Unknown output_mode: {output_mode!r}. Supported: tool, prompted, native"
        )

    @staticmethod
    def _create_model(
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
    ) -> Any:
        """Create a PydanticAI model for the given provider.

        Args:
            provider: One of "ollama", "huggingface", "openrouter", "openai".
            model: Model name / identifier.
            base_url: API base URL (used by ollama and openai).
            api_key: API key.

        Returns:
            A PydanticAI model instance.

        Raises:
            ValueError: If *provider* is not recognised.
        """
        if provider == "ollama":
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.ollama import OllamaProvider

            return OpenAIChatModel(
                model_name=model,
                provider=OllamaProvider(base_url=base_url),
            )

        if provider == "huggingface":
            from pydantic_ai.models.huggingface import HuggingFaceModel
            from pydantic_ai.providers.huggingface import HuggingFaceProvider

            return HuggingFaceModel(
                model,
                provider=HuggingFaceProvider(api_key=api_key),
            )

        if provider == "openrouter":
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openrouter import OpenRouterProvider

            return OpenAIChatModel(
                model_name=model,
                provider=OpenRouterProvider(api_key=api_key),
            )

        if provider == "openai":
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            # Only pass base_url if non-empty (empty string breaks URL construction)
            provider_kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                provider_kwargs["base_url"] = base_url

            return OpenAIChatModel(
                model_name=model,
                provider=OpenAIProvider(**provider_kwargs),
            )

        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Supported: ollama, huggingface, openrouter, openai"
        )

    def generate(
        self,
        context: Context,
        template: PromptTemplate,
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
            # Validation failed - store raw response and error metadata
            # so ActionPair can skip the domain guard and feed the
            # structural error back to the LLM directly.
            error_msg = self._format_validation_error(e)
            content = e.body if e.body else str(e)
            metadata = {
                "generator_error": error_msg,
                "generator_error_kind": "validation",
            }
            logger.warning(
                "[%s] Validation failed: %s",
                self.__class__.__name__,
                error_msg,
            )

        except Exception as e:
            # Other errors (network, timeout, etc.)
            error_msg = f"Generation failed: {e}"
            content = str(e)
            metadata = {
                "generator_error": error_msg,
                "generator_error_kind": "infrastructure",
            }
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

    def _build_prompt(
        self,
        context: Context,
        template: PromptTemplate,
    ) -> str:
        """Build the prompt for the LLM.

        Default implementation uses template.render() which handles
        {ap_*} placeholder substitution automatically.

        Args:
            context: Execution context
            template: Prompt template

        Returns:
            The prompt string to send to the LLM
        """
        return template.render(context)

    def _get_system_prompt(self, template: PromptTemplate) -> str:
        """Get the system prompt from the template.

        Args:
            template: Prompt template (must have a role defined)

        Returns:
            System prompt string

        Raises:
            ValueError: If template.role is empty
        """
        if not template.role:
            raise ValueError(
                f"{self.__class__.__name__} requires a role in PromptTemplate. "
                "Define 'role' in prompts.json for this action pair."
            )
        return template.role

    def _process_output(self, output: OutputT, context: Context) -> str:  # noqa: ARG002
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

        Accesses ``error.message`` and ``error.body`` (both present on
        PydanticAI's ``UnexpectedModelBehavior``) so the LLM sees exactly
        what structural problem occurred and what it produced.

        When ``body`` is ``None`` (common with ``retries=0``), traverses
        the ``__cause__`` chain to find the underlying validation details
        (e.g. Pydantic ``ValidationError`` or ``ToolRetryError``).

        Args:
            error: The UnexpectedModelBehavior exception

        Returns:
            Constructive error message for retry feedback
        """
        msg = getattr(error, "message", str(error))
        body = getattr(error, "body", None)

        if body:
            # Include a truncated preview of what the model returned
            preview = body[:500]
            base = (
                f"Output validation failed: {msg}\n"
                f"Your response could not be parsed as a tool call. "
                f"You must use the provided tool to structure your output.\n"
                f"Raw response preview: {preview}"
            )
            # Check for JSON-escaping artefacts in the body
            escape_hint = detect_escape_issues(body)
            if escape_hint:
                base += f"\n\nNote: {escape_hint}"
            return base

        # No body — extract details from __cause__ chain
        # (retries=0 → ToolRetryError → ValidationError)
        cause_detail = self._extract_cause_detail(error)
        if cause_detail:
            return (
                f"Output validation failed: {msg}\n"
                f"Your response could not be parsed correctly. "
                f"You must use the provided tool to structure your output.\n"
                f"Validation details: {cause_detail}"
            )

        return f"Output validation failed: {msg}"

    @staticmethod
    def _extract_cause_detail(error: Exception) -> str | None:
        """Walk the ``__cause__`` chain and collect useful details.

        PydanticAI wraps validation errors as:
        ``UnexpectedModelBehavior`` → ``ToolRetryError(.tool_retry)``
        → ``ValidationError``.
        """
        parts: list[str] = []
        seen: set[int] = set()
        current: BaseException | None = getattr(error, "__cause__", None)

        while current is not None and id(current) not in seen:
            seen.add(id(current))
            # ToolRetryError carries a RetryPromptPart with content
            tool_retry = getattr(current, "tool_retry", None)
            if tool_retry is not None:
                content = getattr(tool_retry, "content", None)
                if content:
                    parts.append(str(content)[:500])
            else:
                detail = str(current)[:500]
                if detail:
                    parts.append(detail)
            current = getattr(current, "__cause__", None)

        return " | ".join(parts) if parts else None
