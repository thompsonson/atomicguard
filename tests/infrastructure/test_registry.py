"""Tests for GeneratorRegistry - entry points-based generator discovery."""

import pytest

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.infrastructure import GeneratorRegistry, MockGenerator, OllamaGenerator


class TestEntryPointsLoading:
    """Tests for entry points discovery and loading."""

    def test_available_returns_registered_generators(self) -> None:
        """available() lists generators from entry points."""
        GeneratorRegistry.clear()

        available = GeneratorRegistry.available()

        assert "OllamaGenerator" in available
        assert "MockGenerator" in available

    def test_load_idempotent(self) -> None:
        """Multiple _load_entry_points() calls don't duplicate entries."""
        GeneratorRegistry.clear()

        GeneratorRegistry._load_entry_points()
        count_after_first = len(GeneratorRegistry._generators)

        GeneratorRegistry._load_entry_points()
        count_after_second = len(GeneratorRegistry._generators)

        assert count_after_first == count_after_second

    def test_lazy_loading(self) -> None:
        """Entry points are only loaded on first access."""
        GeneratorRegistry.clear()

        # After clear, _loaded should be False
        assert GeneratorRegistry._loaded is False
        assert len(GeneratorRegistry._generators) == 0

        # Accessing available() triggers load
        _ = GeneratorRegistry.available()

        assert GeneratorRegistry._loaded is True
        assert len(GeneratorRegistry._generators) > 0


class TestRegistryOperations:
    """Tests for registry get/create/available operations."""

    def test_get_returns_generator_class(self) -> None:
        """get() returns the generator class by name."""
        GeneratorRegistry.clear()

        generator_class = GeneratorRegistry.get("OllamaGenerator")

        assert generator_class is OllamaGenerator

    def test_get_mock_generator(self) -> None:
        """get() returns MockGenerator class."""
        GeneratorRegistry.clear()

        generator_class = GeneratorRegistry.get("MockGenerator")

        assert generator_class is MockGenerator

    def test_get_unknown_raises_keyerror(self) -> None:
        """get() raises KeyError with helpful message for unknown names."""
        GeneratorRegistry.clear()

        with pytest.raises(KeyError) as exc_info:
            GeneratorRegistry.get("NonExistentGenerator")

        error_message = str(exc_info.value)
        assert "NonExistentGenerator" in error_message
        assert "Available generators" in error_message

    def test_create_instantiates_generator(self) -> None:
        """create() returns a GeneratorInterface instance."""
        GeneratorRegistry.clear()

        generator = GeneratorRegistry.create(
            "MockGenerator", responses=["test response"]
        )

        assert isinstance(generator, GeneratorInterface)
        assert isinstance(generator, MockGenerator)

    def test_create_passes_config(self) -> None:
        """create() passes config kwargs to constructor."""
        GeneratorRegistry.clear()

        responses = ["first", "second", "third"]
        generator = GeneratorRegistry.create("MockGenerator", responses=responses)

        assert generator._responses == responses

    def test_available_returns_list(self) -> None:
        """available() returns a list of generator names."""
        GeneratorRegistry.clear()

        available = GeneratorRegistry.available()

        assert isinstance(available, list)
        assert all(isinstance(name, str) for name in available)


class TestManualRegistration:
    """Tests for manual generator registration."""

    def test_register_adds_generator(self) -> None:
        """register() adds a generator that can be retrieved."""
        GeneratorRegistry.clear()

        class CustomGenerator(MockGenerator):
            pass

        GeneratorRegistry.register("CustomGenerator", CustomGenerator)

        assert "CustomGenerator" in GeneratorRegistry.available()
        assert GeneratorRegistry.get("CustomGenerator") is CustomGenerator

    def test_register_overrides_entry_point(self) -> None:
        """Manual registration can override entry point registration."""
        GeneratorRegistry.clear()

        # Force load entry points first
        original_class = GeneratorRegistry.get("MockGenerator")
        assert original_class is MockGenerator

        # Create a custom class and register with same name
        class OverrideMockGenerator(MockGenerator):
            pass

        GeneratorRegistry.register("MockGenerator", OverrideMockGenerator)

        # Now get() should return the override
        assert GeneratorRegistry.get("MockGenerator") is OverrideMockGenerator

    def test_clear_resets_registry(self) -> None:
        """clear() resets registry state completely."""
        GeneratorRegistry.clear()

        # Force load
        _ = GeneratorRegistry.available()
        assert GeneratorRegistry._loaded is True
        assert len(GeneratorRegistry._generators) > 0

        # Clear
        GeneratorRegistry.clear()

        assert GeneratorRegistry._loaded is False
        assert len(GeneratorRegistry._generators) == 0

    def test_clear_allows_reload(self) -> None:
        """After clear(), entry points can be reloaded."""
        GeneratorRegistry.clear()

        # Load, clear, reload
        first_available = GeneratorRegistry.available()
        GeneratorRegistry.clear()
        second_available = GeneratorRegistry.available()

        assert first_available == second_available


class TestErrorHandling:
    """Tests for error handling during registry operations."""

    def test_create_with_invalid_config_raises_typeerror(self) -> None:
        """create() raises TypeError for unknown config fields."""
        GeneratorRegistry.clear()

        # MockGenerator's config dataclass rejects unknown fields
        # This tests that config validation happens at construction
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            GeneratorRegistry.create("MockGenerator", unknown_field="bad")

    def test_get_with_empty_registry_after_clear(self) -> None:
        """get() still works after clear() by reloading entry points."""
        GeneratorRegistry.clear()
        GeneratorRegistry._loaded = True  # Pretend we already loaded (but cleared)

        # This should trigger reload since generators dict is empty
        GeneratorRegistry.clear()  # Reset properly
        generator_class = GeneratorRegistry.get("OllamaGenerator")

        assert generator_class is OllamaGenerator
