"""
ConfigExtractorGenerator: Extracts project configuration from documentation.

Action Pair 0 in the SDLC workflow - extracts Ω (Global Constraints).
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

CONFIG_EXTRACTOR_SYSTEM_PROMPT = """You are a project configuration extractor.
Extract ACTUAL project metadata from the architecture documentation provided.

CRITICAL: Extract the REAL values from the documentation. Do NOT use placeholders like "your-project-name".

Look for paths like:
- "src/taskmanager/domain/" → source_root is "src/taskmanager", package_name is "taskmanager"
- "src/myapp/application/" → source_root is "src/myapp", package_name is "myapp"

The package name is the directory name directly under src/ that contains domain/, application/, infrastructure/.

Return a JSON object with the ACTUAL values found:
{
  "source_root": "<actual path from docs>",
  "package_name": "<actual package name from docs>"
}

NEVER return placeholder values. Extract exactly what is in the documentation.
"""


@dataclass
class ConfigExtractorConfig:
    """Configuration for ConfigExtractorGenerator."""

    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434/v1"
    timeout: float = 120.0


class ConfigExtractorGenerator(GeneratorInterface):
    """
    Extracts ProjectConfig (Ω) from documentation using LLM.

    This is Action Pair 0 in the SDLC workflow. It extracts global constraints
    that apply to all subsequent action pairs.
    """

    config_class = ConfigExtractorConfig

    def __init__(
        self,
        config: ConfigExtractorConfig | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = ConfigExtractorConfig(**kwargs)

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
        template: PromptTemplate,
        action_pair_id: str = "g_config",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """Extract project configuration from documentation.

        Args:
            context: Generation context with specification
            template: Required prompt template (no fallback)
            action_pair_id: Identifier for this action pair
            workflow_id: UUID of the workflow execution instance

        Returns:
            Generated artifact with project configuration
        """
        logger.debug("[ConfigExtractor] Building prompt...")

        # Use template to render prompt (includes all feedback history)
        prompt = template.render(context)

        logger.debug(f"[ConfigExtractor] Prompt length: {len(prompt)} chars")

        # Call LLM
        messages = [
            {"role": "system", "content": CONFIG_EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info("[ConfigExtractor] Calling LLM...")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore
                temperature=0.3,
            )
            content = response.choices[0].message.content or ""
            logger.info("[ConfigExtractor] Got response")

            # Try to extract JSON from response
            extracted = self._extract_json(content)

        except Exception as e:
            logger.warning(f"[ConfigExtractor] LLM call failed: {e}")
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
