"""
AgentTypeGenerator: Selects and justifies agent type.

Step 4 in the Agent Design Process workflow.
Selects from Russell & Norvig taxonomy.
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

AGENT_TYPE_SYSTEM_PROMPT = """You are an AI architect selecting agent types.

## Agent Types (Russell & Norvig Taxonomy)

1. **simple_reflex**: Condition-action rules based on current percept only
   - Use when: Fully observable, simple mappings, no memory needed
   - Limitation: Cannot handle partial observability

2. **model_based_reflex**: Maintains internal state/model of world
   - Use when: Partially observable, need to track history
   - Adds: Internal state updated by percepts

3. **goal_based**: Explicit goal representation, may use search
   - Use when: Multiple paths to goal, need planning
   - Adds: Goal representation, search capability

4. **utility_based**: Maximizes expected utility function
   - Use when: Multiple goals, trade-offs needed, probabilistic outcomes
   - Adds: Utility function, expected value calculations

5. **learning**: Improves performance over time
   - Use when: Unknown environment dynamics, need adaptation
   - Adds: Learning component, performance feedback

## Selection Criteria

Consider:
1. Environment complexity (from classification)
2. Observability (partial → model-based or better)
3. Need for planning (goal-based or utility-based)
4. Multiple objectives (utility-based)
5. Need to learn/adapt (learning)

## REQUIRED OUTPUT FORMAT

Return a JSON object:
{
  "selected_type": "model_based_reflex",
  "justification": "The agent operates in a partially observable environment and needs to track conversation history. Simple reflex insufficient because...",
  "alternatives_considered": ["simple_reflex", "goal_based"],
  "rejection_reasons": {
    "simple_reflex": "Cannot handle partial observability of user intent",
    "goal_based": "No explicit goal search needed, just response generation"
  },
  "required_capabilities": [
    "Internal state tracking",
    "Percept history maintenance",
    "Model update mechanism"
  ],
  "dual_state_rationale": "Workflow state (S_workflow) tracks guard completion, environment state (S_env) tracks conversation context"
}

Justify based on environment properties and agent function requirements.
"""


@dataclass
class AgentTypeGeneratorConfig:
    """Configuration for AgentTypeGenerator."""

    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 120.0


class AgentTypeGenerator(GeneratorInterface):
    """
    Selects and justifies the agent type.

    This is Step 4 in the Agent Design Process workflow.
    """

    config_class = AgentTypeGeneratorConfig

    def __init__(
        self,
        config: AgentTypeGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = AgentTypeGeneratorConfig(**kwargs)

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
        action_pair_id: str = "g_agent_type",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Select and justify the agent type."""
        logger.debug("[AgentTypeGenerator] Building prompt...")

        # Get dependencies
        env_info = ""
        func_info = ""
        if context.dependency_artifacts:
            for dep_name, dep_id in context.dependency_artifacts:
                try:
                    dep_artifact = context.ambient.repository.get_artifact(dep_id)
                    dep_data = json.loads(dep_artifact.content)
                    if dep_name == "g_environment":
                        env_info = f"\n\n## Environment Properties:\n{json.dumps(dep_data, indent=2)}"
                    elif dep_name == "g_agent_function":
                        func_info = (
                            f"\n\n## Agent Function:\n{json.dumps(dep_data, indent=2)}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[AgentTypeGenerator] Could not load {dep_name}: {e}"
                    )

        # Build prompt
        if template:
            prompt = template.render(context)
        else:
            prompt = f"Select the agent type for this agent:\n\n{context.specification}{env_info}{func_info}"

        if context.feedback_history:
            feedback = context.feedback_history[-1][1]
            prompt += (
                f"\n\nPrevious attempt feedback: {feedback}\nFix the issues above."
            )

        logger.debug(f"[AgentTypeGenerator] Prompt length: {len(prompt)} chars")

        messages = [
            {"role": "system", "content": AGENT_TYPE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[AgentTypeGenerator] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.5,
            )
            content = response.choices[0].message.content or ""
            logger.info("[AgentTypeGenerator] Got response")

            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[AgentTypeGenerator] LLM call failed: {e}")
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
