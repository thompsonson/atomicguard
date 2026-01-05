"""
CoderGenerator: Generates implementation code.

Takes architecture tests and BDD scenarios and generates implementation
that satisfies all constraints.
"""

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

logger = logging.getLogger("sdlc_checkpoint")

CODER_SYSTEM_PROMPT = """You are a senior Python developer implementing features.

Generate Python code that satisfies all architecture tests and BDD scenarios.

## CRITICAL REQUIREMENTS

1. **Valid Python Syntax**: Your code MUST parse without errors
2. **Correct Indentation**: Use 4 spaces. Methods inside classes must be indented
3. **No Markdown**: Do NOT wrap code in ```python markers. Return raw Python only
4. **Clean Architecture**: Follow the layer structure:
   - domain/ - Pure business logic, no external imports
   - application/ - Use cases, depends on domain
   - infrastructure/ - Implementations of ports

## COMMON ERRORS TO AVOID

WRONG - method not indented in class:
```
@dataclass
class Task:
    id: str
def create(self):  # WRONG: should be indented
    pass
```

CORRECT:
```
@dataclass
class Task:
    id: str
    def create(self):  # CORRECT: indented inside class
        pass
```

## OUTPUT FORMAT

Return a JSON object with implementation files:
{
  "files": [
    {
      "path": "src/taskmanager/domain/entities.py",
      "content": "from dataclasses import dataclass\\n\\n@dataclass\\nclass Task:\\n    id: str\\n    title: str"
    },
    {
      "path": "src/taskmanager/domain/value_objects.py",
      "content": "from enum import Enum\\n\\nclass Priority(Enum):\\n    LOW = 'low'\\n    HIGH = 'high'"
    }
  ],
  "summary": "Implemented Task entity with value objects"
}

Generate complete implementation for all layers.
"""


@dataclass
class CoderGeneratorConfig:
    """Configuration for CoderGenerator."""

    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 180.0
    workdir: str = "output"


class CoderGenerator(GeneratorInterface):
    """
    Generates implementation code from architecture tests and BDD scenarios.

    Takes the outputs from ADD and BDD generators and produces
    implementation code that satisfies all constraints.
    """

    config_class = CoderGeneratorConfig

    def __init__(
        self,
        config: CoderGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = CoderGeneratorConfig(**kwargs)

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
        self._workdir = config.workdir
        self._version_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,
        action_pair_id: str = "g_coder",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Generate implementation code."""
        logger.debug("[CoderGenerator] Building prompt...")

        # Collect dependency artifacts
        arch_tests = ""
        bdd_scenarios = ""
        config_info = ""

        if context.dependency_artifacts:
            for dep_name, dep_id in context.dependency_artifacts:
                try:
                    dep_artifact = context.ambient.repository.get_artifact(dep_id)
                    if dep_name == "g_add":
                        arch_tests = f"\n\n## Architecture Tests (from ADD)\n{dep_artifact.content}"
                    elif dep_name == "g_bdd":
                        bdd_scenarios = f"\n\n## BDD Scenarios\n{dep_artifact.content}"
                    elif dep_name == "g_config":
                        config_data = json.loads(dep_artifact.content)
                        config_info = f"\n\nProject Config: source_root={config_data.get('source_root', '')}"
                except Exception as e:
                    logger.warning(
                        f"[CoderGenerator] Could not load dep {dep_name}: {e}"
                    )

        # Build prompt
        if template:
            prompt = template.render(context)
        else:
            prompt = f"""Generate implementation for this system:

{context.specification}
{config_info}
{arch_tests}
{bdd_scenarios}

Return a valid JSON object with implementation files.
"""

        # Add feedback if present
        if context.feedback_history:
            feedback = context.feedback_history[-1][1]
            prompt += f"\n\nTEST FAILURE:\n{feedback}\n\nFix the implementation to pass all tests."

        logger.debug(f"[CoderGenerator] Prompt length: {len(prompt)} chars")

        # Call LLM
        messages = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[CoderGenerator] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.5,
            )
            content = response.choices[0].message.content or ""
            logger.info("[CoderGenerator] Got response")

            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[CoderGenerator] LLM call failed: {e}")
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
