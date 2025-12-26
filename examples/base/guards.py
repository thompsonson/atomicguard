"""Guard building factory for AtomicGuard examples."""

from __future__ import annotations

from typing import Any

from atomicguard import (
    CompositeGuard,
    DynamicTestGuard,
    GuardInterface,
    HumanReviewGuard,
    ImportGuard,
    SyntaxGuard,
)

from .exceptions import ConfigurationError

# Registry of available guard types and their constructors
GUARD_REGISTRY: dict[str, type[GuardInterface]] = {
    "syntax": SyntaxGuard,
    "import": ImportGuard,
    "human": HumanReviewGuard,
    "dynamic_test": DynamicTestGuard,
}


def register_guard(name: str, guard_class: type[GuardInterface]) -> None:
    """
    Register a custom guard type.

    Args:
        name: Guard type identifier (e.g., "my_guard")
        guard_class: GuardInterface implementation class
    """
    GUARD_REGISTRY[name] = guard_class


def build_guard(config: dict[str, Any]) -> GuardInterface:
    """
    Build a guard from configuration.

    Supports:
    - "syntax": SyntaxGuard for Python AST validation
    - "import": ImportGuard for undefined name detection
    - "human": HumanReviewGuard for human approval
    - "dynamic_test": DynamicTestGuard for test execution
    - "composite": CompositeGuard combining multiple guards
    - Custom guards registered via register_guard()

    Args:
        config: Guard configuration dict with "guard" key

    Returns:
        Configured GuardInterface instance

    Raises:
        ConfigurationError: If guard type is unknown or config is invalid
    """
    if "guard" not in config:
        raise ConfigurationError("Guard config missing 'guard' key")

    guard_type = config["guard"]

    if guard_type == "composite":
        return _build_composite_guard(config)

    if guard_type not in GUARD_REGISTRY:
        valid_types = list(GUARD_REGISTRY.keys()) + ["composite"]
        raise ConfigurationError(
            f"Unknown guard type: '{guard_type}'. "
            f"Valid types: {', '.join(valid_types)}"
        )

    guard_class = GUARD_REGISTRY[guard_type]

    # Handle guards with constructor arguments
    if guard_type == "human":
        return HumanReviewGuard(
            prompt_title=config.get("human_prompt_title", "HUMAN REVIEW REQUIRED")
        )

    return guard_class()


def _build_composite_guard(config: dict[str, Any]) -> CompositeGuard:
    """Build a CompositeGuard from configuration."""
    if "guards" not in config:
        raise ConfigurationError("Composite guard missing 'guards' list")
    if not isinstance(config["guards"], list):
        raise ConfigurationError("Composite 'guards' must be a list")
    if not config["guards"]:
        raise ConfigurationError("Composite 'guards' cannot be empty")

    guards = []
    for g in config["guards"]:
        sub_config: dict[str, Any] = {"guard": g}
        # Pass through human_prompt_title if present
        if g == "human" and "human_prompt_title" in config:
            sub_config["human_prompt_title"] = config["human_prompt_title"]
        guards.append(build_guard(sub_config))

    return CompositeGuard(*guards)
