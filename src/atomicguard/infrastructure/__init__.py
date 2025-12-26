"""
Infrastructure layer for the Dual-State Framework.

Contains adapters for external concerns (persistence, LLMs, registry, etc.).
"""

from atomicguard.infrastructure.llm import (
    MockGenerator,
    OllamaGenerator,
)
from atomicguard.infrastructure.persistence import (
    FilesystemArtifactDAG,
    InMemoryArtifactDAG,
)
from atomicguard.infrastructure.registry import GeneratorRegistry

__all__ = [
    # Persistence
    "InMemoryArtifactDAG",
    "FilesystemArtifactDAG",
    # LLM
    "OllamaGenerator",
    "MockGenerator",
    # Registry
    "GeneratorRegistry",
]
