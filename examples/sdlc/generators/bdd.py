"""BDD Generator - Creates Gherkin scenarios from requirements.

Implements GeneratorInterface to produce BDD scenarios that describe
the expected behavior of the feature being developed.
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

from ..models import BDDScenarios

logger = logging.getLogger(__name__)

# =============================================================================
# BDD Generator
# =============================================================================

BDD_SYSTEM_PROMPT = """You are a BDD specialist creating Gherkin scenarios.

Your task is to create valid Gherkin feature files from requirements documentation.

Guidelines:
1. Each feature should have a clear, descriptive name
2. Scenarios should use Given/When/Then format
3. Steps should be specific and testable
4. Use tags to categorize scenarios (@smoke, @integration, etc.)
5. Keep scenarios focused on a single behavior

Output must be valid JSON matching the BDDScenarios schema.
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
class BDDGeneratorConfig:
    """Configuration for BDDGenerator.

    This typed config ensures unknown fields are rejected at construction time.
    """

    model: str = "qwen2.5-coder:7b"
    base_url: str | None = None
    min_scenarios: int = 3
    prompt_template: PromptTemplate | None = None
    workdir: str | Path | None = None


class BDDGenerator(GeneratorInterface):
    """
    Generates BDD scenarios (Gherkin format) from requirements.

    Uses PydanticAI to produce structured output matching BDDScenarios model.
    """

    config_class = BDDGeneratorConfig

    def __init__(
        self,
        config: BDDGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize BDDGenerator.

        Args:
            config: Typed configuration object (preferred)
            **kwargs: Legacy kwargs for backward compatibility
        """
        if config is None:
            config = BDDGeneratorConfig(**kwargs)

        self._model = config.model
        self._base_url = config.base_url
        self._min_scenarios = config.min_scenarios
        self._prompt_template = config.prompt_template
        self._workdir = Path(config.workdir) if config.workdir else None

        # Build system prompt from template or use default
        if config.prompt_template:
            system_prompt = (
                f"{config.prompt_template.role}\n\n{config.prompt_template.constraints}"
            )
        else:
            system_prompt = BDD_SYSTEM_PROMPT

        pydantic_model = _create_ollama_model(config.model, config.base_url)
        self._agent = Agent(
            pydantic_model,
            output_type=PromptedOutput(BDDScenarios),
            system_prompt=system_prompt,
            retries=0,  # Retries handled by AtomicGuard layer
        )

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,
        action_pair_id: str = "g_bdd",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """
        Generate BDD scenarios from requirements.

        Args:
            context: Contains requirements in specification field
            template: Optional prompt template override
            action_pair_id: Identifier for this action pair
            workflow_id: UUID of the workflow execution

        Returns:
            Artifact containing BDDScenarios as JSON
        """
        logger.debug("[BDDGenerator] Building prompt...")

        # Build the prompt
        prompt_parts = [
            f"Create BDD scenarios for the following requirements:\n\n{context.specification}"
        ]

        # Add project config from ambient constraints if available
        if context.ambient and context.ambient.constraints:
            prompt_parts.append(
                f"\n\nProject configuration:\n{context.ambient.constraints}"
            )

        # Add dependency artifacts context
        if context.dependency_artifacts:
            for dep_id, dep_content in context.dependency_artifacts:
                prompt_parts.append(f"\n\nDependency ({dep_id}):\n{dep_content}")

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
            logger.debug("[BDDGenerator] Calling LLM...")
            result = self._agent.run_sync(prompt)
            bdd_scenarios: BDDScenarios = result.output

            # Write feature file to disk if workdir is configured
            if self._workdir and bdd_scenarios.feature_file_content:
                features_dir = self._workdir / "features"
                features_dir.mkdir(parents=True, exist_ok=True)
                feature_file = features_dir / "scenarios.feature"
                feature_file.write_text(bdd_scenarios.feature_file_content)
                logger.info(f"[BDDGenerator] Wrote feature file: {feature_file}")

            # Serialize to JSON for artifact content
            content = bdd_scenarios.model_dump_json(indent=2)
            logger.info(
                f"[BDDGenerator] Generated {len(bdd_scenarios.features)} features"
            )

        except UnexpectedModelBehavior as e:
            logger.warning(f"[BDDGenerator] LLM output error: {e}")
            content = json.dumps(
                {"error": "schema_validation_failed", "details": str(e)}
            )
        except ValidationError as e:
            logger.warning(f"[BDDGenerator] Pydantic validation error: {e}")
            content = json.dumps({"error": "validation_failed", "details": str(e)})
        except Exception as e:
            logger.error(f"[BDDGenerator] Unexpected error: {e}")
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
