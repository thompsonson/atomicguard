"""
PEASGenerator: Extracts PEAS analysis from problem statement.

Step 1 in the Agent Design Process workflow.
PEAS = Performance, Environment, Actuators, Sensors (Russell & Norvig)
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

logger = logging.getLogger("agent_design")

PEAS_SYSTEM_PROMPT = """You are an AI systems analyst using the PEAS framework.

PEAS Framework (Russell & Norvig):
- **Performance**: Metrics that define success. Must be measurable.
- **Environment**: Components of the task environment.
- **Actuators**: Actions the agent can perform.
- **Sensors**: Information the agent receives.

## REQUIRED OUTPUT FORMAT

Return a JSON object:
{
  "performance_measures": [
    {
      "name": "accuracy",
      "description": "Percentage of correct predictions",
      "success_criteria": ">= 95% accuracy on test set",
      "measurable": true
    }
  ],
  "environment_elements": [
    {
      "name": "user_input",
      "description": "Text queries from users",
      "observable": true,
      "modifiable": false
    }
  ],
  "actuators": [
    {
      "name": "GENERATE_RESPONSE",
      "description": "Produces text output",
      "effect": "Sends response to user",
      "category": "external"
    }
  ],
  "sensors": [
    {
      "name": "query_sensor",
      "description": "Receives user queries",
      "data_type": "string",
      "source": "user_input"
    }
  ],
  "summary": "Brief description of the agent's task environment"
}

Be specific and comprehensive. Include at least one item in each category.
"""


@dataclass
class PEASGeneratorConfig:
    """Configuration for PEASGenerator."""

    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 120.0


class PEASGenerator(GeneratorInterface):
    """
    Extracts PEAS analysis from problem statement.

    This is Step 1 in the Agent Design Process workflow.
    """

    config_class = PEASGeneratorConfig

    def __init__(
        self,
        config: PEASGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = PEASGeneratorConfig(**kwargs)

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
        self._version_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,
        action_pair_id: str = "g_peas",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Extract PEAS analysis from problem statement."""
        logger.debug("[PEASGenerator] Building prompt...")

        # Build prompt
        if template:
            prompt = template.render(context)
        else:
            prompt = f"Analyze this problem statement and produce a PEAS analysis:\n\n{context.specification}"

        # Add feedback if present
        if context.feedback_history:
            feedback = context.feedback_history[-1][1]
            prompt += (
                f"\n\nPrevious attempt feedback: {feedback}\nFix the issues above."
            )

        logger.debug(f"[PEASGenerator] Prompt length: {len(prompt)} chars")

        # Call LLM
        messages = [
            {"role": "system", "content": PEAS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[PEASGenerator] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.5,
            )
            content = response.choices[0].message.content or ""
            logger.info("[PEASGenerator] Got response")

            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[PEASGenerator] LLM call failed: {e}")
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
