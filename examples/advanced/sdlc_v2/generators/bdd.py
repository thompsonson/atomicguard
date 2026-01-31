"""
BDDGenerator: Generates BDD scenarios from requirements.

Creates Gherkin scenarios that capture behavior from requirements documentation.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import Artifact, ArtifactStatus, Context, ContextSnapshot
from atomicguard.domain.prompts import PromptTemplate

logger = logging.getLogger("sdlc_checkpoint")

BDD_SYSTEM_PROMPT = """You are a BDD specialist creating Gherkin scenarios.

Generate BDD scenarios from requirements documentation.

## REQUIRED FORMAT

Return a JSON object with this structure:
{
  "feature_name": "Task Management",
  "background": "Given a task repository is initialized",
  "scenarios": [
    {
      "name": "Create a new task",
      "feature": "Task Management",
      "gherkin": "Scenario: Create a new task\\n  Given I have a task management system\\n  When I create a task with title 'Test Task' and priority 'high'\\n  Then the task should be created with status 'pending'\\n  And the task should have a unique ID"
    }
  ]
}

## GHERKIN FORMAT

Use proper Gherkin syntax:
- Feature: (feature name)
- Background: (optional common setup)
- Scenario: (scenario name)
- Given (precondition)
- When (action)
- Then (expected result)
- And (additional step)

Generate at least 3 scenarios covering the main user stories.
"""


@dataclass
class BDDGeneratorConfig:
    """Configuration for BDDGenerator."""

    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 120.0
    min_scenarios: int = 3


class BDDGenerator(GeneratorInterface):
    """
    Generates BDD scenarios from requirements documentation.

    Takes requirements documentation and produces a BDDScenariosResult JSON
    that can be validated by the BDDGuard.
    """

    config_class = BDDGeneratorConfig

    def __init__(
        self,
        config: BDDGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = BDDGeneratorConfig(**kwargs)

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
        self._min_scenarios = config.min_scenarios
        self._version_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate,
        action_pair_id: str = "g_bdd",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Generate BDD scenarios from requirements.

        Args:
            context: Generation context with specification
            template: Required prompt template (no fallback)
            action_pair_id: Identifier for this action pair
            workflow_id: UUID of the workflow execution instance

        Returns:
            Generated artifact with BDD scenarios
        """
        logger.debug("[BDDGenerator] Building prompt...")

        # Use template to render prompt (includes all feedback history)
        prompt = template.render(context)

        logger.debug(f"[BDDGenerator] Prompt length: {len(prompt)} chars")

        # Call LLM
        messages = [
            {"role": "system", "content": BDD_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[BDDGenerator] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.5,
            )
            content = response.choices[0].message.content or ""
            logger.info("[BDDGenerator] Got response")

            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[BDDGenerator] LLM call failed: {e}")
            extracted = json.dumps({"error": str(e)})

        self._version_counter += 1

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=extracted,
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
        """Extract JSON from LLM response."""
        import re

        # Try to find JSON block
        json_match = re.search(r"```json?\s*([\s\S]*?)```", content)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                pass

        # Try to parse entire content as JSON
        try:
            data = json.loads(content)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object pattern
        obj_match = re.search(r"\{[\s\S]*\}", content)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                pass

        return json.dumps({"error": "Could not extract JSON", "raw": content[:500]})
