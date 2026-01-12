"""
AgentFunctionGenerator: Defines the agent function f: P* → A.

Step 3 in the Agent Design Process workflow.
Maps percept sequences to actions.
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

AGENT_FUNCTION_SYSTEM_PROMPT = """You are an AI agent designer.

Define the agent function f: P* → A, mapping percept sequences to actions.

## Components to Define

1. **Percepts**: What the agent receives from sensors
   - name: Identifier
   - source: Which sensor provides this
   - data_structure: Format/schema
   - example: Sample value

2. **Actions**: What the agent can do
   - name: Action identifier
   - category: external (affects world), sensing (gathers info), internal (updates state)
   - precondition: When this action is applicable
   - effect: What changes
   - actuator: Which actuator performs this

3. **Percept-Action Sequences**: Typical mappings
   - scenario_name: Descriptive name
   - percepts: List of percepts that trigger
   - action: Resulting action
   - rationale: Why this mapping

4. **State Representation**: If the agent is model-based, describe internal state

## REQUIRED OUTPUT FORMAT

Return a JSON object:
{
  "percepts": [
    {
      "name": "user_query",
      "source": "query_sensor",
      "data_structure": "{ text: string, timestamp: datetime }",
      "example": "{ text: 'What is X?', timestamp: '2024-01-01T00:00:00' }"
    }
  ],
  "actions": [
    {
      "name": "GENERATE_RESPONSE",
      "category": "external",
      "precondition": "user_query received and valid",
      "effect": "Response sent to user",
      "actuator": "response_generator"
    }
  ],
  "percept_action_sequences": [
    {
      "scenario_name": "Simple query response",
      "percepts": ["user_query"],
      "action": "GENERATE_RESPONSE",
      "rationale": "Direct query triggers immediate response"
    }
  ],
  "state_representation": "Maintains conversation history and user context",
  "state_variables": ["conversation_history", "user_preferences"]
}

Ensure percepts reference sensors and actions reference actuators from PEAS.
"""


@dataclass
class AgentFunctionGeneratorConfig:
    """Configuration for AgentFunctionGenerator."""

    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 120.0


class AgentFunctionGenerator(GeneratorInterface):
    """
    Defines the agent function mapping percepts to actions.

    This is Step 3 in the Agent Design Process workflow.
    """

    config_class = AgentFunctionGeneratorConfig

    def __init__(
        self,
        config: AgentFunctionGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = AgentFunctionGeneratorConfig(**kwargs)

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
        action_pair_id: str = "g_agent_function",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Define the agent function."""
        logger.debug("[AgentFunctionGenerator] Building prompt...")

        # Get dependencies
        peas_info = ""
        env_info = ""
        if context.dependency_artifacts:
            for dep_name, dep_id in context.dependency_artifacts:
                try:
                    dep_artifact = context.ambient.repository.get_artifact(dep_id)
                    dep_data = json.loads(dep_artifact.content)
                    if dep_name == "g_peas":
                        peas_info = (
                            f"\n\n## PEAS Analysis:\n{json.dumps(dep_data, indent=2)}"
                        )
                    elif dep_name == "g_environment":
                        env_info = f"\n\n## Environment Properties:\n{json.dumps(dep_data, indent=2)}"
                except Exception as e:
                    logger.warning(
                        f"[AgentFunctionGenerator] Could not load {dep_name}: {e}"
                    )

        # Build prompt
        if template:
            prompt = template.render(context)
        else:
            prompt = f"Define the agent function for this agent:\n\n{context.specification}{peas_info}{env_info}"

        if context.feedback_history:
            feedback = context.feedback_history[-1][1]
            prompt += (
                f"\n\nPrevious attempt feedback: {feedback}\nFix the issues above."
            )

        logger.debug(f"[AgentFunctionGenerator] Prompt length: {len(prompt)} chars")

        messages = [
            {"role": "system", "content": AGENT_FUNCTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[AgentFunctionGenerator] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.5,
            )
            content = response.choices[0].message.content or ""
            logger.info("[AgentFunctionGenerator] Got response")

            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[AgentFunctionGenerator] LLM call failed: {e}")
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
