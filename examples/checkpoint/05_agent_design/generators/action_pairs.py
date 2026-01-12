"""
ActionPairGenerator: Designs Action Pair specifications.

Step 6 in the Agent Design Process workflow.
Designs A = ⟨ρ, a_gen, G⟩ (precondition, generator, guard).
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

ACTION_PAIRS_SYSTEM_PROMPT = """You are a Dual-State Action Pair architect.

## Action Pair Pattern: A = ⟨ρ, a_gen, G⟩

Each Action Pair couples:
- **ρ (rho) - Precondition**: Boolean condition determining when this action applies
- **a_gen - Generator**: Component that produces the output artifact
- **G - Guard**: Deterministic verification of the generator's output

## Design Principles

1. **Atomicity**: Each action pair is an atomic unit of work
2. **Traceability**: Link to acceptance criteria and agent function
3. **Separation**: Generator is stochastic (may use LLM), Guard is deterministic
4. **Composability**: Action pairs form a DAG workflow

## REQUIRED OUTPUT FORMAT

Return a JSON object:
{
  "action_pairs": [
    {
      "action_pair_id": "ap_process_query",
      "name": "Process User Query",
      "description": "Handles incoming user queries and generates responses",
      "precondition": "percept.type == 'user_query' and percept.text != ''",
      "precondition_percepts": ["user_query"],
      "generator_name": "QueryResponseGenerator",
      "generator_description": "Uses LLM to generate contextual response",
      "generator_inputs": ["user_query", "conversation_history"],
      "generator_output_schema": "{ response: string, confidence: float }",
      "guard_name": "ResponseQualityGuard",
      "guard_description": "Validates response relevance and safety",
      "guard_checks": [
        "response is non-empty",
        "response addresses query topic",
        "response passes safety filter"
      ],
      "acceptance_criteria_refs": ["AC-001", "AC-002"],
      "action_refs": ["GENERATE_RESPONSE"]
    }
  ],
  "workflow_order": ["ap_process_query", "ap_send_response"],
  "dependencies": {
    "ap_send_response": ["ap_process_query"]
  },
  "state_transitions": "Linear progression: receive query → process → respond"
}

Design action pairs that implement the acceptance criteria.
"""


@dataclass
class ActionPairGeneratorConfig:
    """Configuration for ActionPairGenerator."""

    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 120.0


class ActionPairGenerator(GeneratorInterface):
    """
    Designs Action Pair specifications.

    This is Step 6 in the Agent Design Process workflow.
    """

    config_class = ActionPairGeneratorConfig

    def __init__(
        self,
        config: ActionPairGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = ActionPairGeneratorConfig(**kwargs)

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
        action_pair_id: str = "g_action_pairs",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Design Action Pair specifications."""
        logger.debug("[ActionPairGenerator] Building prompt...")

        # Get dependencies
        atdd_info = ""
        peas_info = ""
        if context.dependency_artifacts:
            for dep_name, dep_id in context.dependency_artifacts:
                try:
                    dep_artifact = context.ambient.repository.get_artifact(dep_id)
                    dep_data = json.loads(dep_artifact.content)
                    if dep_name == "g_atdd":
                        atdd_info = f"\n\n## Acceptance Criteria:\n{json.dumps(dep_data, indent=2)}"
                    elif dep_name == "g_peas":
                        peas_info = (
                            f"\n\n## PEAS Analysis:\n{json.dumps(dep_data, indent=2)}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[ActionPairGenerator] Could not load {dep_name}: {e}"
                    )

        # Build prompt
        if template:
            prompt = template.render(context)
        else:
            prompt = f"""Design Action Pairs for this agent:

{context.specification}
{peas_info}
{atdd_info}

Design action pairs that implement the acceptance criteria.
Each action pair must have precondition (ρ), generator (a_gen), and guard (G).
"""

        if context.feedback_history:
            feedback = context.feedback_history[-1][1]
            prompt += (
                f"\n\nPrevious attempt feedback: {feedback}\nFix the issues above."
            )

        logger.debug(f"[ActionPairGenerator] Prompt length: {len(prompt)} chars")

        messages = [
            {"role": "system", "content": ACTION_PAIRS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[ActionPairGenerator] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.5,
            )
            content = response.choices[0].message.content or ""
            logger.info("[ActionPairGenerator] Got response")

            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[ActionPairGenerator] LLM call failed: {e}")
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
