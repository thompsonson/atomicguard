"""
IdentityGenerator: Pass-through generator for validation-only steps.

Used when the action pair only validates (via guard) without generating
new content. Simply passes the content from a dependency artifact through.
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


@dataclass
class IdentityGeneratorConfig:
    """Configuration for IdentityGenerator."""

    source_artifact: str = "g_coder"  # Which dependency to pass through


class IdentityGenerator(GeneratorInterface):
    """
    Pass-through generator that forwards a dependency artifact.

    Used for validation-only steps where the guard does all the work.
    The generator simply copies content from a specified dependency.
    """

    config_class = IdentityGeneratorConfig

    def __init__(
        self,
        config: IdentityGeneratorConfig | None = None,
        source_artifact: str = "g_coder",
        **_kwargs: Any,
    ):
        if config is None:
            config = IdentityGeneratorConfig(source_artifact=source_artifact)

        self._source_artifact = config.source_artifact
        self._version_counter = 0

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,  # noqa: ARG002
        action_pair_id: str = "identity",
        workflow_id: str = "unknown",
    ) -> Artifact:
        """
        Generate artifact by passing through dependency content.

        Args:
            context: Execution context with dependency artifacts
            template: Ignored for identity generator
            action_pair_id: ID of this action pair
            workflow_id: Current workflow ID

        Returns:
            Artifact containing the source artifact's content
        """
        logger.debug(
            f"[IdentityGenerator] Passing through from {self._source_artifact}"
        )

        # Find the source artifact
        content = ""
        if context.dependency_artifacts:
            for dep_name, dep_id in context.dependency_artifacts:
                if dep_name == self._source_artifact:
                    try:
                        dep_artifact = context.ambient.repository.get_artifact(dep_id)
                        content = dep_artifact.content
                        logger.debug(
                            f"[IdentityGenerator] Found {dep_name}: {len(content)} chars"
                        )
                        break
                    except Exception as e:
                        logger.warning(
                            f"[IdentityGenerator] Could not load {dep_name}: {e}"
                        )

        if not content:
            logger.warning(
                f"[IdentityGenerator] Source artifact {self._source_artifact} not found"
            )
            content = json.dumps(
                {"error": f"Source artifact {self._source_artifact} not found"}
            )

        self._version_counter += 1

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=content,
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
