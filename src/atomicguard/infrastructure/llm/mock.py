"""
Mock generator for testing without LLM.

Returns predefined responses in sequence.
"""

import uuid
from datetime import datetime

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import Artifact, ArtifactStatus, Context, ContextSnapshot
from atomicguard.domain.prompts import PromptTemplate


class MockGenerator(GeneratorInterface):
    """Returns predefined responses for testing."""

    def __init__(self, responses: list[str]):
        """
        Args:
            responses: List of response strings to return in sequence
        """
        self._responses = responses
        self._call_count = 0

    def generate(
        self,
        _context: Context,
        _template: PromptTemplate | None = None,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
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
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=self._call_count,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=ContextSnapshot(
                workflow_id=workflow_id,
                specification="",
                constraints="",
                feedback_history=(),
                dependency_artifacts=(),
            ),
        )

    @property
    def call_count(self) -> int:
        """Number of times generate() has been called."""
        return self._call_count

    def reset(self) -> None:
        """Reset the call counter to reuse responses."""
        self._call_count = 0
