"""
Base guard implementations and composition patterns.

CompositeGuard implements the Decorator pattern for guard composition.
"""

from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult


class CompositeGuard(GuardInterface):
    """
    Logical AND of multiple guards. All must pass.

    Evaluates guards in order, short-circuits on first failure.
    This ensures automated checks run before human review.

    Per paper section on Composite Guards:
    G_composite = G_automated ∧ G_human
    """

    def __init__(self, *guards: GuardInterface):
        """
        Args:
            *guards: Guards to compose (evaluated in order)
        """
        self.guards = guards

    def validate(self, artifact: Artifact, **deps: Any) -> GuardResult:
        """
        Validate artifact against all composed guards.

        Short-circuits on first failure.
        """
        for guard in self.guards:
            result = guard.validate(artifact, **deps)
            if not result.passed:
                return result  # Short-circuit on failure
        return GuardResult(passed=True, feedback="All guards passed")
