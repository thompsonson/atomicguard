"""
ATDDGenerator: Generates Given-When-Then acceptance criteria.

Step 5 in the Agent Design Process workflow.
Follows ATDD methodology and 10 Principles for Acceptance Criteria.
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

ATDD_SYSTEM_PROMPT = """You are an ATDD specialist for intelligent agents.

## Given-When-Then Format

For each acceptance scenario:
- **Given**: Preconditions - what percepts/state exist before the trigger
- **When**: The trigger - action or event being tested
- **Then**: Expected outcomes - observable behaviors

## The 10 Principles for Agent Acceptance Criteria

1. Start with clear agent function specification
2. Test observable behaviors, not implementation details
3. Avoid predetermined outcomes that force specific results
4. Match criteria to agent type (reflex vs learning agents differ)
5. Handle probabilistic outputs with statistical properties
6. Translate theory to testable behavior
7. Ignore implementation accidents
8. Cover full percept-action cycles
9. Define clear scope boundaries
10. Make tests stakeholder-meaningful (domain language)

## REQUIRED OUTPUT FORMAT

Return a JSON object:
{
  "scenarios": [
    {
      "scenario_id": "AC-001",
      "name": "Agent responds to valid query",
      "given": [
        "the agent is initialized",
        "the user submits query 'What is X?'"
      ],
      "when": [
        "the agent processes the query percept"
      ],
      "then": [
        "the agent should execute GENERATE_RESPONSE action",
        "the response should be observable within 5 seconds",
        "the response should address the query topic"
      ],
      "percept_refs": ["user_query"],
      "action_refs": ["GENERATE_RESPONSE"],
      "principle_compliance": {
        "observable_behavior": true,
        "full_cycle": true,
        "stakeholder_meaningful": true
      }
    }
  ],
  "coverage_summary": "Covers 3 of 4 defined percept-action pairs",
  "untested_behaviors": ["error handling for malformed input"]
}

Generate at least 3 scenarios covering major agent behaviors.
Reference percepts and actions from the agent function.
"""


@dataclass
class ATDDGeneratorConfig:
    """Configuration for ATDDGenerator."""

    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 120.0
    min_scenarios: int = 3


class ATDDGenerator(GeneratorInterface):
    """
    Generates Given-When-Then acceptance criteria.

    This is Step 5 in the Agent Design Process workflow.
    """

    config_class = ATDDGeneratorConfig

    def __init__(
        self,
        config: ATDDGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = ATDDGeneratorConfig(**kwargs)

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
        template: PromptTemplate | None = None,
        action_pair_id: str = "g_atdd",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Generate acceptance criteria."""
        logger.debug("[ATDDGenerator] Building prompt...")

        # Get dependencies
        func_info = ""
        type_info = ""
        if context.dependency_artifacts:
            for dep_name, dep_id in context.dependency_artifacts:
                try:
                    dep_artifact = context.ambient.repository.get_artifact(dep_id)
                    dep_data = json.loads(dep_artifact.content)
                    if dep_name == "g_agent_function":
                        func_info = (
                            f"\n\n## Agent Function:\n{json.dumps(dep_data, indent=2)}"
                        )
                    elif dep_name == "g_agent_type":
                        type_info = (
                            f"\n\n## Agent Type:\n{json.dumps(dep_data, indent=2)}"
                        )
                except Exception as e:
                    logger.warning(f"[ATDDGenerator] Could not load {dep_name}: {e}")

        # Build prompt
        if template:
            prompt = template.render(context)
        else:
            prompt = f"""Generate acceptance criteria for this agent:

{context.specification}
{func_info}
{type_info}

Generate at least {self._min_scenarios} scenarios covering the main agent behaviors.
"""

        if context.feedback_history:
            feedback = context.feedback_history[-1][1]
            prompt += (
                f"\n\nPrevious attempt feedback: {feedback}\nFix the issues above."
            )

        logger.debug(f"[ATDDGenerator] Prompt length: {len(prompt)} chars")

        messages = [
            {"role": "system", "content": ATDD_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[ATDDGenerator] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.5,
            )
            content = response.choices[0].message.content or ""
            logger.info("[ATDDGenerator] Got response")

            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[ATDDGenerator] LLM call failed: {e}")
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
