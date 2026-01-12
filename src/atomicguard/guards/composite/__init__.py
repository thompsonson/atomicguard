"""
Composite guards - Guard composition patterns (Extension 08).

These guards combine multiple guards using composition strategies:
- SequentialGuard: Fail-fast ordered execution (Definition 39)
- ParallelGuard: Concurrent execution with aggregation (Definition 40)
- CompositeGuard: Backwards-compatible alias for SequentialGuard
"""

from atomicguard.guards.composite.base import (
    AggregationPolicy,
    CompositeGuard,
    ParallelGuard,
    SequentialGuard,
    SubGuardResult,
)

__all__ = [
    "AggregationPolicy",
    "CompositeGuard",
    "ParallelGuard",
    "SequentialGuard",
    "SubGuardResult",
]
