"""
EnvironmentPropertiesGenerator: Classifies environment properties.

Step 2 in the Agent Design Process workflow.
Classifies across 6 dimensions per Russell & Norvig.
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

ENVIRONMENT_SYSTEM_PROMPT = """You are an AI environment analyst.

Classify the task environment across these 6 dimensions (Russell & Norvig):

1. **Observable**:
   - Fully observable: Agent can see complete environment state
   - Partially observable: Agent has limited view (e.g., hidden state, noisy sensors)

2. **Deterministic**:
   - Deterministic: Next state determined entirely by current state + action
   - Stochastic: Uncertainty in outcomes

3. **Static**:
   - Static: Environment doesn't change while agent deliberates
   - Dynamic: Environment can change during agent's decision process

4. **Discrete**:
   - Discrete: Finite number of states, actions, percepts
   - Continuous: Infinite/continuous state/action space

5. **Agents**:
   - Single-agent: Only this agent affects the environment
   - Multi-agent: Other agents (competitive or cooperative)

6. **Known**:
   - Known: Agent knows the rules/dynamics of the environment
   - Unknown: Agent must learn how the environment works

## REQUIRED OUTPUT FORMAT

Return a JSON object:
{
  "properties": [
    {
      "dimension": "observable",
      "classification": "partially observable",
      "justification": "Agent cannot see internal user intent..."
    },
    {
      "dimension": "deterministic",
      "classification": "stochastic",
      "justification": "LLM outputs are probabilistic..."
    },
    {
      "dimension": "static",
      "classification": "static",
      "justification": "Environment doesn't change during processing..."
    },
    {
      "dimension": "discrete",
      "classification": "discrete",
      "justification": "Finite set of actions and states..."
    },
    {
      "dimension": "agents",
      "classification": "single-agent",
      "justification": "No other agents compete..."
    },
    {
      "dimension": "known",
      "classification": "partially known",
      "justification": "Rules known but LLM behavior is opaque..."
    }
  ],
  "overall_complexity": "moderate",
  "key_challenges": ["handling partial observability", "managing stochastic outputs"]
}

You MUST include all 6 dimensions with justifications based on PEAS analysis.
"""


@dataclass
class EnvironmentPropertiesGeneratorConfig:
    """Configuration for EnvironmentPropertiesGenerator."""

    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 120.0


class EnvironmentPropertiesGenerator(GeneratorInterface):
    """
    Classifies environment properties across 6 dimensions.

    This is Step 2 in the Agent Design Process workflow.
    """

    config_class = EnvironmentPropertiesGeneratorConfig

    def __init__(
        self,
        config: EnvironmentPropertiesGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = EnvironmentPropertiesGeneratorConfig(**kwargs)

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
        action_pair_id: str = "g_environment",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Classify environment properties."""
        logger.debug("[EnvironmentPropertiesGenerator] Building prompt...")

        # Get PEAS analysis from dependencies
        peas_info = ""
        if context.dependency_artifacts:
            for dep_name, dep_id in context.dependency_artifacts:
                if dep_name == "g_peas":
                    try:
                        dep_artifact = context.ambient.repository.get_artifact(dep_id)
                        peas_data = json.loads(dep_artifact.content)
                        peas_info = (
                            f"\n\n## PEAS Analysis:\n{json.dumps(peas_data, indent=2)}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[EnvironmentPropertiesGenerator] Could not load PEAS: {e}"
                        )

        # Build prompt
        if template:
            prompt = template.render(context)
        else:
            prompt = f"Classify the environment properties for this agent:\n\n{context.specification}{peas_info}"

        # Add feedback if present
        if context.feedback_history:
            feedback = context.feedback_history[-1][1]
            prompt += (
                f"\n\nPrevious attempt feedback: {feedback}\nFix the issues above."
            )

        logger.debug(
            f"[EnvironmentPropertiesGenerator] Prompt length: {len(prompt)} chars"
        )

        # Call LLM
        messages = [
            {"role": "system", "content": ENVIRONMENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[EnvironmentPropertiesGenerator] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.5,
            )
            content = response.choices[0].message.content or ""
            logger.info("[EnvironmentPropertiesGenerator] Got response")

            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[EnvironmentPropertiesGenerator] LLM call failed: {e}")
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
            feedback="",
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

        json_match = re.search(r"```json?\s*([\s\S]*?)```", content)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                pass

        try:
            data = json.loads(content)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            pass

        obj_match = re.search(r"\{[\s\S]*\}", content)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                pass

        return json.dumps({"error": "Could not extract JSON", "raw": content[:500]})
