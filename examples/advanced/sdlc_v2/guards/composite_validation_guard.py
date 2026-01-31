"""
CompositeValidationGuard: Combines multiple validation guards for g_coder.

Extension 08 implementation for SDLC workflow. Composes:
- AllTestsPassGuard (syntax/structure validation)
- QualityGatesGuard (mypy/ruff checks)
- ArchValidationGuard (pytest-arch tests)

All feedback routes to CoderGenerator for effective retry.
"""

import logging
from typing import Any

from atomicguard.domain.models import Artifact, GuardResult
from atomicguard.guards.composite import AggregationPolicy, SequentialGuard

from .arch_validation_guard import ArchValidationGuard
from .quality_guard import QualityGatesGuard
from .tests_guard import AllTestsPassGuard

logger = logging.getLogger("sdlc_checkpoint")


class CompositeValidationGuard(SequentialGuard):
    """
    Composite guard for full code validation (Extension 08).

    Executes in sequence (fail-fast):
    1. AllTestsPassGuard - Syntax and structure validation (fast)
    2. QualityGatesGuard - mypy and ruff checks (medium)
    3. ArchValidationGuard - pytest-arch tests (slow)

    All feedback routes back to CoderGenerator for retry,
    solving the IdentityGenerator feedback routing problem.

    Usage:
        guard = CompositeValidationGuard(
            workdir="output",
            run_mypy=True,
            run_ruff=True,
        )
        result = guard.validate(artifact, g_add=g_add_artifact)
    """

    def __init__(
        self,
        workdir: str = "output",
        run_mypy: bool = True,
        run_ruff: bool = True,
        timeout: float = 60.0,
    ):
        """
        Initialize the composite validation guard.

        Args:
            workdir: Working directory for file extraction
            run_mypy: Whether to run mypy type checking
            run_ruff: Whether to run ruff linting
            timeout: Timeout in seconds for each tool
        """
        guards = [
            AllTestsPassGuard(workdir=workdir),
            QualityGatesGuard(
                run_mypy=run_mypy,
                run_ruff=run_ruff,
                timeout=timeout,
            ),
            ArchValidationGuard(timeout=timeout),
        ]
        super().__init__(guards, AggregationPolicy.ALL_PASS)

        logger.debug(
            "[CompositeValidationGuard] Initialized with %d sub-guards",
            len(guards),
        )

    def validate(self, artifact: Artifact, **deps: Any) -> GuardResult:
        """
        Validate implementation against all composed guards.

        Args:
            artifact: The implementation artifact (from g_coder)
            **deps: Dependency artifacts. Must include 'g_add' for ArchValidationGuard.

        Returns:
            GuardResult with combined feedback from all guards
        """
        logger.debug("[CompositeValidationGuard] Starting composite validation...")
        result = super().validate(artifact, **deps)
        logger.debug(
            "[CompositeValidationGuard] %s",
            "PASS" if result.passed else "FAIL",
        )
        return result
