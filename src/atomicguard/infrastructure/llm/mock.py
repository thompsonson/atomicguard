"""
Mock generator for testing without LLM.

Returns predefined responses in sequence.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import Artifact, ArtifactStatus, Context, ContextSnapshot
from atomicguard.domain.prompts import PromptTemplate


@dataclass
class MockGeneratorConfig:
    """Configuration for MockGenerator.

    This typed config ensures unknown fields are rejected at construction time.
    """

    responses: list[str] = field(default_factory=list)


class MockGenerator(GeneratorInterface):
    """Returns predefined responses for testing."""

    config_class = MockGeneratorConfig

    def __init__(self, config: MockGeneratorConfig | None = None, **kwargs: Any):
        """
        Args:
            config: Typed configuration object (preferred)
            **kwargs: Legacy kwargs for backward compatibility (deprecated)
        """
        # Support both config object and legacy kwargs
        if config is None:
            config = MockGeneratorConfig(**kwargs)

        self._responses = config.responses
        self._call_count = 0

    def generate(
        self,
        context: Context,  # noqa: ARG002 - unused but required by interface
        template: PromptTemplate,  # noqa: ARG002
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
        workflow_ref: str | None = None,
    ) -> Artifact:
        """Return the next predefined response."""
        if self._call_count >= len(self._responses):
            raise RuntimeError("MockGenerator exhausted responses")

        content = self._responses[self._call_count]
        self._call_count += 1

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            content=content,
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=self._call_count,
            status=ArtifactStatus.PENDING,
            guard_result=None,  # Guard result set after validation
            context=ContextSnapshot(
                workflow_id=workflow_id,
                specification="",
                constraints="",
                feedback_history=(),
                dependency_artifacts=(),
            ),
            workflow_ref=workflow_ref,
        )

    @property
    def call_count(self) -> int:
        """Number of times generate() has been called."""
        return self._call_count

    def reset(self) -> None:
        """Reset the call counter to reuse responses."""
        self._call_count = 0
