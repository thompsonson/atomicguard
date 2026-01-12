"""
ImplementationGenerator: Generates Dual-State Action Pair agent skeleton.

Step 7 in the Agent Design Process workflow.
Produces workflow.json, models.py, and generator/guard skeletons.
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

IMPLEMENTATION_SYSTEM_PROMPT = """You are a Python developer implementing Dual-State Action Pair agents.

## Output Structure

Generate a complete implementation skeleton:

1. **workflow.json**: DAG configuration with action pairs
2. **models.py**: Pydantic schemas for all generator outputs
3. **generators/**: One file per generator
4. **guards/**: One file per guard

## Code Patterns

### Generator Pattern:
```python
class MyGenerator(GeneratorInterface):
    def generate(self, context: Context, template: PromptTemplate | None,
                 action_pair_id: str, workflow_id: str) -> Artifact:
        # 1. Extract dependencies from context
        # 2. Build prompt
        # 3. Call LLM
        # 4. Return Artifact with JSON content
```

### Guard Pattern:
```python
class MyGuard(GuardInterface):
    def validate(self, artifact: Artifact, **dependencies: Artifact) -> GuardResult:
        # 1. Parse artifact content
        # 2. Validate against schema
        # 3. Check business rules
        # 4. Return GuardResult(passed=bool, feedback=str, fatal=bool)
```

## CRITICAL REQUIREMENTS
- Use 4-space indentation
- NO markdown code blocks in the content field
- Include type hints
- Include docstrings
- Follow atomicguard import patterns

## REQUIRED OUTPUT FORMAT

Return a JSON object:
{
  "workflow_config": {
    "name": "Generated Agent",
    "model": "qwen2.5-coder:14b",
    "rmax": 3,
    "action_pairs": {
      "g_step1": {
        "generator": "Step1Generator",
        "guard": "step1_valid",
        "description": "First step"
      }
    }
  },
  "workflow_steps": [
    {
      "step_id": "g_step1",
      "generator": "Step1Generator",
      "guard": "step1_valid",
      "requires": [],
      "description": "First step"
    }
  ],
  "files": [
    {
      "path": "models.py",
      "content": "from pydantic import BaseModel\\n\\nclass Step1Output(BaseModel):\\n    result: str",
      "description": "Pydantic models for outputs",
      "file_type": "python"
    },
    {
      "path": "generators/step1.py",
      "content": "class Step1Generator(GeneratorInterface):\\n    ...",
      "description": "Generator for step 1",
      "file_type": "python"
    },
    {
      "path": "guards/step1_guard.py",
      "content": "class Step1Guard(GuardInterface):\\n    ...",
      "description": "Guard for step 1",
      "file_type": "python"
    }
  ],
  "setup_instructions": "1. Install dependencies\\n2. Run: python demo.py run",
  "design_summary": "This agent implements X by Y"
}
"""


@dataclass
class ImplementationGeneratorConfig:
    """Configuration for ImplementationGenerator."""

    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 180.0  # Longer timeout for code generation


class ImplementationGenerator(GeneratorInterface):
    """
    Generates Dual-State Action Pair agent implementation skeleton.

    This is Step 7 in the Agent Design Process workflow.
    """

    config_class = ImplementationGeneratorConfig

    def __init__(
        self,
        config: ImplementationGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = ImplementationGeneratorConfig(**kwargs)

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
        action_pair_id: str = "g_implementation",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Generate implementation skeleton."""
        logger.debug("[ImplementationGenerator] Building prompt...")

        # Get dependencies
        action_pairs_info = ""
        peas_info = ""
        if context.dependency_artifacts:
            for dep_name, dep_id in context.dependency_artifacts:
                try:
                    dep_artifact = context.ambient.repository.get_artifact(dep_id)
                    dep_data = json.loads(dep_artifact.content)
                    if dep_name == "g_action_pairs":
                        action_pairs_info = f"\n\n## Action Pairs Design:\n{json.dumps(dep_data, indent=2)}"
                    elif dep_name == "g_peas":
                        peas_info = (
                            f"\n\n## PEAS Analysis:\n{json.dumps(dep_data, indent=2)}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[ImplementationGenerator] Could not load {dep_name}: {e}"
                    )

        # Build prompt
        if template:
            prompt = template.render(context)
        else:
            prompt = f"""Generate the implementation skeleton for this agent:

{context.specification}
{peas_info}
{action_pairs_info}

Generate:
1. workflow.json configuration
2. models.py with Pydantic schemas
3. Generator files for each action pair
4. Guard files for each action pair

Follow the atomicguard patterns exactly.
"""

        if context.feedback_history:
            feedback = context.feedback_history[-1][1]
            prompt += (
                f"\n\nPrevious attempt feedback: {feedback}\nFix the issues above."
            )

        logger.debug(f"[ImplementationGenerator] Prompt length: {len(prompt)} chars")

        messages = [
            {"role": "system", "content": IMPLEMENTATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[ImplementationGenerator] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.5,
            )
            content = response.choices[0].message.content or ""
            logger.info("[ImplementationGenerator] Got response")

            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[ImplementationGenerator] LLM call failed: {e}")
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
