"""
ADDGenerator: Architecture-Driven Development generator.

Generates pytest-arch tests from architecture documentation.
This is a simplified version for the SDLC Checkpoint example.
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

ADD_SYSTEM_PROMPT = """You are an architecture test generator using pytestarch.

Generate pytest-arch tests that enforce architectural constraints from the documentation.

## REQUIRED FORMAT

Return a JSON object with this structure:
{
  "module_docstring": "Architecture tests for task manager system.",
  "imports": [
    "from pytestarch import get_evaluable_architecture, Rule",
    "import pytest"
  ],
  "fixtures": [
    "@pytest.fixture(scope='module')\\ndef evaluable():\\n    return get_evaluable_architecture('src/taskmanager', 'src/taskmanager')"
  ],
  "tests": [
    {
      "gate_id": "Gate1",
      "test_name": "test_domain_no_infrastructure_imports",
      "test_code": "def test_domain_no_infrastructure_imports(evaluable):\\n    rule = Rule().modules_that().are_sub_modules_of('taskmanager.domain').should_not().import_modules_that().are_sub_modules_of('taskmanager.infrastructure')\\n    rule.assert_applies(evaluable)",
      "imports_required": [],
      "documentation_reference": "Gate 1: Domain Purity"
    }
  ]
}

## PYTESTARCH API

Only use these methods:
- Rule().modules_that().are_sub_modules_of("package")
- .should_not().import_modules_that().are_sub_modules_of("package")
- .assert_applies(evaluable)

DO NOT use any .or_* methods - they don't exist.
"""


@dataclass
class ADDGeneratorConfig:
    """Configuration for ADDGenerator."""

    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 120.0
    min_gates: int = 1
    min_tests: int = 1


class ADDGenerator(GeneratorInterface):
    """
    Generates pytest-arch tests from architecture documentation.

    Takes architecture documentation and produces a TestSuite JSON
    that can be validated by the ArchitectureTestsGuard.
    """

    config_class = ADDGeneratorConfig

    def __init__(
        self,
        config: ADDGeneratorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = ADDGeneratorConfig(**kwargs)

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
        self._min_gates = config.min_gates
        self._min_tests = config.min_tests
        self._version_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate,
        action_pair_id: str = "g_add",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Generate architecture tests from documentation.

        Args:
            context: Generation context with specification and dependencies
            template: Required prompt template (no fallback)
            action_pair_id: Identifier for this action pair
            workflow_id: UUID of the workflow execution instance

        Returns:
            Generated artifact with architecture tests
        """
        logger.debug("[ADDGenerator] Building prompt...")

        # Get project config from dependencies for context enrichment
        dep_context_parts = []
        if context.dependency_artifacts:
            for dep_name, dep_id in context.dependency_artifacts:
                if dep_name == "g_config":
                    try:
                        dep_artifact = context.ambient.repository.get_artifact(dep_id)
                        config_data = json.loads(dep_artifact.content)
                        source_root = config_data.get("source_root", "")
                        package_name = config_data.get("package_name", "")
                        if source_root:
                            dep_context_parts.append(
                                f"\n\nProject Configuration:\n- Source Root: {source_root}\n- Package Name: {package_name}"
                            )
                    except Exception as e:
                        logger.warning(f"[ADDGenerator] Could not load config: {e}")

        # Use template to render prompt (includes all feedback history)
        prompt = template.render(context)

        # Append dependency context after template content
        if dep_context_parts:
            prompt += "\n" + "\n".join(dep_context_parts)

        logger.debug(f"[ADDGenerator] Prompt length: {len(prompt)} chars")

        # Call LLM
        messages = [
            {"role": "system", "content": ADD_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[ADDGenerator] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.5,
            )
            content = response.choices[0].message.content or ""
            logger.info("[ADDGenerator] Got response")

            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[ADDGenerator] LLM call failed: {e}")
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
