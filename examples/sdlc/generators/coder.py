"""Coder Generator - Generates implementation code from tests and scenarios.

Implements GeneratorInterface to produce implementation code that satisfies
the architecture tests (from ADD) and BDD scenarios.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import PromptedOutput
from pydantic_ai.providers.ollama import OllamaProvider

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import Artifact, ArtifactStatus, Context, ContextSnapshot
from atomicguard.domain.prompts import PromptTemplate

from ..models import ImplementationManifest

logger = logging.getLogger(__name__)

# =============================================================================
# Coder Generator
# =============================================================================

CODER_SYSTEM_PROMPT = """You are a senior Python developer implementing features.

Your task is to generate Python code that:
1. Passes all architecture tests (pytest-arch constraints)
2. Implements all BDD scenarios
3. Follows clean code principles
4. Includes type hints and docstrings

Guidelines:
- Respect the layer boundaries defined in architecture tests
- Import only allowed dependencies for each layer
- Use descriptive names matching the ubiquitous language
- Keep functions focused and testable

Output must be valid JSON matching the ImplementationManifest schema.
"""


def _create_ollama_model(model: str, base_url: str | None) -> OpenAIChatModel:
    """Create Ollama-backed OpenAI-compatible model."""
    # Parse model string (e.g., "ollama:qwen2.5-coder:14b" -> "qwen2.5-coder:14b")
    model_name = model[7:] if model.startswith("ollama:") else model

    effective_base_url = base_url or "http://localhost:11434/v1"

    return OpenAIChatModel(
        model_name,
        provider=OllamaProvider(base_url=effective_base_url),
    )


@dataclass
class CoderGeneratorConfig:
    """Configuration for CoderGenerator.

    This typed config ensures unknown fields are rejected at construction time.
    """

    model: str = "qwen2.5-coder:7b"
    base_url: str | None = None
    prompt_template: PromptTemplate | None = None
    workdir: str | Path | None = None


class CoderGenerator(GeneratorInterface):
    """
    Generates implementation code that satisfies tests and scenarios.

    This is the final step in the SDLC pipeline. It receives:
    - Architecture tests from ADDGenerator (g_add)
    - BDD scenarios from BDDGenerator (g_bdd)

    And produces implementation files that should pass all tests.
    """

    config_class = CoderGeneratorConfig

    def __init__(
        self,
        config: CoderGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize CoderGenerator.

        Args:
            config: Typed configuration object (preferred)
            **kwargs: Legacy kwargs for backward compatibility
        """
        if config is None:
            config = CoderGeneratorConfig(**kwargs)

        self._model = config.model
        self._base_url = config.base_url
        self._prompt_template = config.prompt_template
        self._workdir = Path(config.workdir) if config.workdir else None

        # Build system prompt from template or use default
        if config.prompt_template:
            system_prompt = (
                f"{config.prompt_template.role}\n\n{config.prompt_template.constraints}"
            )
        else:
            system_prompt = CODER_SYSTEM_PROMPT

        pydantic_model = _create_ollama_model(config.model, config.base_url)
        self._agent = Agent(
            pydantic_model,
            output_type=PromptedOutput(ImplementationManifest),
            system_prompt=system_prompt,
            retries=0,  # Retries handled by AtomicGuard layer
        )

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,
        action_pair_id: str = "g_coder",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """
        Generate implementation code from tests and scenarios.

        Args:
            context: Contains specification and dependency artifacts
            template: Optional prompt template override
            action_pair_id: Identifier for this action pair
            workflow_id: UUID of the workflow execution

        Returns:
            Artifact containing ImplementationManifest as JSON
        """
        logger.debug("[CoderGenerator] Building prompt...")

        # Build the prompt
        prompt_parts = [f"Implement the following feature:\n\n{context.specification}"]

        # Add project config from ambient constraints
        if context.ambient and context.ambient.constraints:
            prompt_parts.append(
                f"\n\nProject configuration:\n{context.ambient.constraints}"
            )

        # Add dependency artifacts (architecture tests, BDD scenarios)
        if context.dependency_artifacts:
            prompt_parts.append("\n\n## Dependencies from previous steps:")
            for dep_id, dep_content in context.dependency_artifacts:
                if "add" in dep_id.lower() or "architecture" in dep_id.lower():
                    prompt_parts.append(
                        f"\n\n### Architecture Tests ({dep_id}):\n{dep_content}"
                    )
                elif "bdd" in dep_id.lower() or "scenario" in dep_id.lower():
                    prompt_parts.append(
                        f"\n\n### BDD Scenarios ({dep_id}):\n{dep_content}"
                    )
                else:
                    prompt_parts.append(f"\n\n### {dep_id}:\n{dep_content}")

        # Add retry feedback if present
        if context.feedback_history:
            feedback = context.feedback_history[-1][1]
            effective_template = template or self._prompt_template
            if effective_template and effective_template.feedback_wrapper:
                feedback_prompt = effective_template.feedback_wrapper.format(
                    feedback=feedback
                )
                prompt_parts.append(f"\n\n{feedback_prompt}")

        prompt = "".join(prompt_parts)

        try:
            logger.debug("[CoderGenerator] Calling LLM...")
            result = self._agent.run_sync(prompt)
            implementation: ImplementationManifest = result.output

            # Write implementation files to disk if workdir is configured
            if self._workdir and implementation.files:
                for file_info in implementation.files:
                    file_path = self._workdir / file_info.path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(file_info.content)
                    logger.info(f"[CoderGenerator] Wrote: {file_path}")

            # Serialize to JSON for artifact content
            content = implementation.model_dump_json(indent=2)
            logger.info(f"[CoderGenerator] Generated {len(implementation.files)} files")

        except UnexpectedModelBehavior as e:
            logger.warning(f"[CoderGenerator] LLM output error: {e}")
            content = json.dumps(
                {"error": "schema_validation_failed", "details": str(e)}
            )
        except ValidationError as e:
            logger.warning(f"[CoderGenerator] Pydantic validation error: {e}")
            content = json.dumps({"error": "validation_failed", "details": str(e)})
        except Exception as e:
            logger.error(f"[CoderGenerator] Unexpected error: {e}")
            content = json.dumps({"error": "generation_failed", "details": str(e)})

        # Determine attempt number from feedback history
        attempt_number = len(context.feedback_history) + 1

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=content,
            previous_attempt_id=None,  # Set by caller if retrying
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=attempt_number,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=ContextSnapshot(
                workflow_id=workflow_id,
                specification=context.specification[:500],  # Truncate for storage
                constraints=context.ambient.constraints if context.ambient else "",
                feedback_history=tuple(context.feedback_history),
                dependency_artifacts=context.dependency_artifacts,
            ),
        )
