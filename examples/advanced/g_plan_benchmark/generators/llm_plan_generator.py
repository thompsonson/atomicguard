"""
LLMPlanGenerator: LLM-backed plan generation for G_plan benchmark.

Uses an LLM to generate workflow plans from a problem specification,
following the same pattern as ADDGenerator / CoderGenerator in sdlc_v2.
The generated plan artifact is then validated by G_plan guards.

This enables epsilon estimation for plan generation:
    epsilon_hat = (plans passing G_plan) / (total generated)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import Artifact, ArtifactStatus, Context, ContextSnapshot
from atomicguard.domain.prompts import PromptTemplate

logger = logging.getLogger("g_plan_benchmark")


@dataclass
class LLMPlanGeneratorConfig:
    """Configuration for LLMPlanGenerator."""

    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 120.0
    temperature: float = 0.7


class LLMPlanGenerator(GeneratorInterface):
    """
    LLM-backed plan generator.

    Follows the same pattern as ADDGenerator in sdlc_v2:
    - Takes a specification via Context
    - Renders a prompt via PromptTemplate
    - Calls the LLM
    - Extracts JSON from the response
    - Returns an Artifact with the plan as content

    The artifact is then validated by G_plan guards (Minimal/Medium/Expansive).
    """

    config_class = LLMPlanGeneratorConfig

    def __init__(
        self,
        config: LLMPlanGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = LLMPlanGeneratorConfig(**kwargs)

        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("openai library required: pip install openai") from err

        self._model = config.model
        self._client = OpenAI(
            base_url=config.base_url,
            api_key="ollama",
            timeout=config.timeout,
        )
        self._temperature = config.temperature
        self._version_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate,
        action_pair_id: str = "g_plan",
        workflow_id: str = "unknown",
        workflow_ref: str | None = None,  # noqa: ARG002
    ) -> Artifact:
        """
        Generate a workflow plan via LLM.

        The specification in context describes the problem the plan should solve.
        The template provides role/constraints/task framing.
        """
        logger.info("[LLMPlanGenerator] Building prompt...")

        # Build system prompt from template role + constraints
        system_prompt = f"{template.role}\n\n{template.constraints}"
        user_prompt = template.render(context)

        logger.debug(f"[LLMPlanGenerator] System prompt: {len(system_prompt)} chars")
        logger.debug(f"[LLMPlanGenerator] User prompt: {len(user_prompt)} chars")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            logger.info(f"[LLMPlanGenerator] Calling LLM ({self._model})...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self._temperature,
            )
            raw_content = response.choices[0].message.content or ""
            logger.info(f"[LLMPlanGenerator] Got response ({len(raw_content)} chars)")
            content = self._extract_json(raw_content)

        except Exception as e:
            logger.warning(f"[LLMPlanGenerator] LLM call failed: {e}")
            content = json.dumps({"error": str(e)})

        self._version_counter += 1

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=content,
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=self._version_counter,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            context=ContextSnapshot(
                workflow_id=workflow_id,
                specification=context.specification,
                constraints=context.ambient.constraints,
                feedback_history=(),
                dependency_artifacts=context.dependency_artifacts,
            ),
        )

    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response, handling markdown fences."""
        # Try markdown JSON block
        json_match = re.search(r"```json?\s*([\s\S]*?)```", content)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                pass

        # Try entire content as JSON
        try:
            data = json.loads(content)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object
        obj_match = re.search(r"\{[\s\S]*\}", content)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                pass

        return json.dumps({"error": "Could not extract JSON", "raw": content[:500]})
