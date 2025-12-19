"""
Guards for the Dual-State Framework.

Guards are deterministic validators that return ⊤ (pass) or ⊥ (fail with feedback).
They can be composed using CompositeGuard for layered validation.

Organization by validation profile:
- static/: Pure AST-based validation (no execution)
- dynamic/: Subprocess-based validation (test execution)
- interactive/: Human-in-loop validation
- composite/: Guard composition patterns
"""

from atomicguard.guards.composite import CompositeGuard
from atomicguard.guards.dynamic import DynamicTestGuard, TestGuard
from atomicguard.guards.interactive import HumanReviewGuard
from atomicguard.guards.static import ImportGuard, SyntaxGuard

__all__ = [
    # Static guards (pure, fast)
    "SyntaxGuard",
    "ImportGuard",
    # Dynamic guards (subprocess-based)
    "TestGuard",
    "DynamicTestGuard",
    # Interactive guards (human-in-loop)
    "HumanReviewGuard",
    # Composition patterns
    "CompositeGuard",
]
