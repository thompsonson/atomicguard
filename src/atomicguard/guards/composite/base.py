"""
Base guard implementations and composition patterns.

Extension 08: Composite Guards (Definitions 38-43, Theorem 14).

Provides guard composition patterns:
- SequentialGuard: Fail-fast ordered execution (Definition 39)
- ParallelGuard: Concurrent execution with aggregation (Definition 40)
- AggregationPolicy: ALL_PASS, ANY_PASS, MAJORITY_PASS (Definition 41)

CompositeGuard is an alias for SequentialGuard for backwards compatibility.
"""

import concurrent.futures
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from atomicguard.domain.interfaces import GuardInterface
from atomicguard.domain.models import Artifact, GuardResult, SubGuardOutcome


class AggregationPolicy(Enum):
    """
    How to combine multiple guard results (Definition 41).

    - ALL_PASS: All guards must pass for composite to pass
    - ANY_PASS: At least one guard must pass
    - MAJORITY_PASS: More than half must pass
    """

    ALL_PASS = "all_pass"
    ANY_PASS = "any_pass"
    MAJORITY_PASS = "majority_pass"


@dataclass
class SubGuardResult:
    """
    Result from a single sub-guard within a composite (Definition 43).

    Captures outcome with timing for observability.
    """

    guard_name: str
    passed: bool
    feedback: str
    execution_time_ms: float = 0.0


class SequentialGuard(GuardInterface):
    """
    Execute guards in sequence, fail-fast on first failure (Definition 39).

    Properties:
    - Short-circuit evaluation: expensive guards skipped on early failure
    - Feedback attribution: failure feedback comes from specific sub-guard
    - Order matters: place cheap/fast guards first for cost optimization

    Per Extension 08:
    G_seq = ⟨[G₁, ..., Gₙ], SEQUENTIAL, FIRST_FAIL⟩
    """

    def __init__(
        self,
        guards: Sequence[GuardInterface],
        policy: AggregationPolicy = AggregationPolicy.ALL_PASS,
    ):
        """
        Args:
            guards: Guards to compose (evaluated in order)
            policy: Aggregation policy (default: ALL_PASS)
        """
        self._guards = list(guards)
        self._policy = policy

    @property
    def guards(self) -> list[GuardInterface]:
        """Access to sub-guards for inspection."""
        return self._guards

    def validate(self, artifact: Artifact, **deps: Any) -> GuardResult:
        """
        Validate artifact against all composed guards.

        Short-circuits on first failure (fail-fast).
        """
        results: list[SubGuardResult] = []

        for guard in self._guards:
            start = time.time()
            result = guard.validate(artifact, **deps)
            elapsed_ms = (time.time() - start) * 1000

            sub_result = SubGuardResult(
                guard_name=guard.__class__.__name__,
                passed=result.passed,
                feedback=result.feedback,
                execution_time_ms=elapsed_ms,
            )
            results.append(sub_result)

            # Fail-fast: stop on first failure
            if not result.passed:
                break

        return self._aggregate(results)

    def _aggregate(self, results: list[SubGuardResult]) -> GuardResult:
        """Aggregate sub-guard results based on policy."""
        if not results:
            return GuardResult(
                passed=True,
                feedback="No guards to evaluate",
                guard_name=self.__class__.__name__,
            )

        passed_count = sum(1 for r in results if r.passed)
        total = len(results)

        # Determine overall pass/fail based on policy
        if self._policy == AggregationPolicy.ALL_PASS:
            passed = passed_count == total
        elif self._policy == AggregationPolicy.ANY_PASS:
            passed = passed_count > 0
        elif self._policy == AggregationPolicy.MAJORITY_PASS:
            passed = passed_count > total / 2
        else:
            passed = passed_count == total  # Default to ALL_PASS

        # Combine feedback from failed guards
        feedback_parts = []
        for r in results:
            if not r.passed:
                feedback_parts.append(f"### {r.guard_name}: FAIL\n{r.feedback}")

        feedback = (
            "\n\n".join(feedback_parts) if feedback_parts else "All guards passed"
        )

        # Convert SubGuardResult to SubGuardOutcome for immutable storage
        sub_outcomes = tuple(
            SubGuardOutcome(
                guard_name=r.guard_name,
                passed=r.passed,
                feedback=r.feedback,
                execution_time_ms=r.execution_time_ms,
            )
            for r in results
        )

        return GuardResult(
            passed=passed,
            feedback=feedback,
            guard_name=self.__class__.__name__,
            sub_results=sub_outcomes,
        )


class ParallelGuard(GuardInterface):
    """
    Execute guards concurrently, aggregate results (Definition 40).

    Properties:
    - All guards execute: No short-circuit (useful for comprehensive feedback)
    - Concurrent execution: Wall-clock time = max(individual times)
    - Complete feedback: All failures reported, not just first

    Per Extension 08:
    G_par = ⟨{G₁, ..., Gₙ}, PARALLEL, policy⟩
    """

    def __init__(
        self,
        guards: Sequence[GuardInterface],
        policy: AggregationPolicy = AggregationPolicy.ALL_PASS,
    ):
        """
        Args:
            guards: Guards to compose (executed concurrently)
            policy: Aggregation policy (default: ALL_PASS)
        """
        self._guards = list(guards)
        self._policy = policy

    @property
    def guards(self) -> list[GuardInterface]:
        """Access to sub-guards for inspection."""
        return self._guards

    def validate(self, artifact: Artifact, **deps: Any) -> GuardResult:
        """
        Validate artifact against all guards concurrently.

        All guards execute regardless of individual failures.
        """
        results: list[SubGuardResult] = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all guards concurrently
            future_to_guard = {
                executor.submit(self._execute_guard, guard, artifact, deps): guard
                for guard in self._guards
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_guard):
                guard = future_to_guard[future]
                try:
                    sub_result = future.result()
                    results.append(sub_result)
                except Exception as e:
                    # Guard raised an exception - treat as failure
                    results.append(
                        SubGuardResult(
                            guard_name=guard.__class__.__name__,
                            passed=False,
                            feedback=f"Guard raised exception: {e}",
                            execution_time_ms=0.0,
                        )
                    )

        return self._aggregate(results)

    def _execute_guard(
        self, guard: GuardInterface, artifact: Artifact, deps: dict[str, Any]
    ) -> SubGuardResult:
        """Execute a single guard with timing."""
        start = time.time()
        result = guard.validate(artifact, **deps)
        elapsed_ms = (time.time() - start) * 1000

        return SubGuardResult(
            guard_name=guard.__class__.__name__,
            passed=result.passed,
            feedback=result.feedback,
            execution_time_ms=elapsed_ms,
        )

    def _aggregate(self, results: list[SubGuardResult]) -> GuardResult:
        """Aggregate sub-guard results based on policy."""
        if not results:
            return GuardResult(
                passed=True,
                feedback="No guards to evaluate",
                guard_name=self.__class__.__name__,
            )

        passed_count = sum(1 for r in results if r.passed)
        total = len(results)

        # Determine overall pass/fail based on policy
        if self._policy == AggregationPolicy.ALL_PASS:
            passed = passed_count == total
        elif self._policy == AggregationPolicy.ANY_PASS:
            passed = passed_count > 0
        elif self._policy == AggregationPolicy.MAJORITY_PASS:
            passed = passed_count > total / 2
        else:
            passed = passed_count == total  # Default to ALL_PASS

        # Combine feedback from failed guards
        feedback_parts = []
        for r in results:
            if not r.passed:
                feedback_parts.append(f"### {r.guard_name}: FAIL\n{r.feedback}")

        feedback = (
            "\n\n".join(feedback_parts) if feedback_parts else "All guards passed"
        )

        # Convert SubGuardResult to SubGuardOutcome for immutable storage
        sub_outcomes = tuple(
            SubGuardOutcome(
                guard_name=r.guard_name,
                passed=r.passed,
                feedback=r.feedback,
                execution_time_ms=r.execution_time_ms,
            )
            for r in results
        )

        return GuardResult(
            passed=passed,
            feedback=feedback,
            guard_name=self.__class__.__name__,
            sub_results=sub_outcomes,
        )


class CompositeGuard(GuardInterface):
    """
    Logical AND of multiple guards. All must pass.

    BACKWARDS COMPATIBILITY: This class maintains the original varargs API.
    For new code, prefer SequentialGuard or ParallelGuard directly.

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
        self._sequential = SequentialGuard(guards, AggregationPolicy.ALL_PASS)
        # Keep guards attribute for backwards compatibility
        self.guards = guards

    def validate(self, artifact: Artifact, **deps: Any) -> GuardResult:
        """
        Validate artifact against all composed guards.

        Short-circuits on first failure.
        """
        return self._sequential.validate(artifact, **deps)
