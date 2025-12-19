"""
Domain exceptions for the Dual-State Framework.

These represent business rule violations in the domain layer.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomicguard.domain.models import Artifact


class RmaxExhausted(Exception):
    """
    Raised when maximum retry attempts are exhausted.

    This is a domain exception representing the business rule that
    generation must succeed within r_max attempts.
    """

    def __init__(self, message: str, provenance: list[tuple["Artifact", str]]):
        """
        Args:
            message: Human-readable error message
            provenance: List of (artifact, feedback) tuples for each failed attempt
        """
        super().__init__(message)
        self.provenance = provenance


class EscalationRequired(Exception):
    """
    Raised when guard returns ⊥_fatal - human intervention needed.

    This indicates a non-recoverable failure that should not be retried.
    The workflow should surface this to the caller for human review.
    """

    def __init__(self, artifact: "Artifact", feedback: str):
        """
        Args:
            artifact: The artifact that triggered escalation
            feedback: Human-readable feedback explaining the fatal condition
        """
        super().__init__(feedback)
        self.artifact = artifact
        self.feedback = feedback
