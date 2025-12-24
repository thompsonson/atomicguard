"""
Generator Registry with Entry Points Discovery.

Provides dynamic generator loading via Python entry points (atomicguard.generators group).
External packages can register generators in their pyproject.toml:

    [project.entry-points."atomicguard.generators"]
    MyGenerator = "mypackage.generators:MyGenerator"
"""

from importlib.metadata import entry_points
from typing import Any

from atomicguard.domain.interfaces import GeneratorInterface


class GeneratorRegistry:
    """
    Registry for GeneratorInterface implementations.

    Discovers generators via the 'atomicguard.generators' entry point group.
    Uses lazy loading - entry points are only loaded on first access.

    Example usage:
        registry = GeneratorRegistry()
        generator = registry.create("OllamaGenerator", model="qwen2.5-coder:14b")
    """

    _generators: dict[str, type[GeneratorInterface]] = {}
    _loaded: bool = False

    @classmethod
    def _load_entry_points(cls) -> None:
        """Load generators from entry points (lazy, called once)."""
        if cls._loaded:
            return

        eps = entry_points(group="atomicguard.generators")
        for ep in eps:
            try:
                generator_class = ep.load()
                cls._generators[ep.name] = generator_class
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to load generator '{ep.name}' from entry point: {e}",
                    stacklevel=2,
                )

        cls._loaded = True

    @classmethod
    def register(cls, name: str, generator_class: type[GeneratorInterface]) -> None:
        """
        Manually register a generator class.

        Useful for testing or dynamically-created generators.

        Args:
            name: Generator identifier (e.g., "OllamaGenerator")
            generator_class: Class implementing GeneratorInterface
        """
        cls._generators[name] = generator_class

    @classmethod
    def get(cls, name: str) -> type[GeneratorInterface]:
        """
        Get a generator class by name.

        Args:
            name: Generator identifier

        Returns:
            The generator class

        Raises:
            KeyError: If generator not found
        """
        cls._load_entry_points()
        if name not in cls._generators:
            available = ", ".join(cls._generators.keys()) or "(none)"
            raise KeyError(
                f"Generator '{name}' not found. Available generators: {available}"
            )
        return cls._generators[name]

    @classmethod
    def create(cls, name: str, **config: Any) -> GeneratorInterface:
        """
        Create a generator instance by name.

        Args:
            name: Generator identifier
            **config: Configuration passed to generator constructor

        Returns:
            Instantiated generator

        Raises:
            KeyError: If generator not found
            TypeError: If config doesn't match constructor signature
        """
        generator_class = cls.get(name)
        return generator_class(**config)

    @classmethod
    def available(cls) -> list[str]:
        """
        List available generator names.

        Returns:
            List of registered generator names
        """
        cls._load_entry_points()
        return list(cls._generators.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered generators (useful for testing).

        Also resets the loaded flag so entry points can be reloaded.
        """
        cls._generators.clear()
        cls._loaded = False
