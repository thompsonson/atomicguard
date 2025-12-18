"""
Infrastructure layer for the Dual-State Framework.

Contains adapters for external concerns (persistence, LLMs, etc.).
"""

from atomicguard.infrastructure.llm import (
    MockGenerator,
    OllamaGenerator,
)
from atomicguard.infrastructure.persistence import (
    FilesystemArtifactDAG,
    InMemoryArtifactDAG,
)

__all__ = [
    # Persistence
    "InMemoryArtifactDAG",
    "FilesystemArtifactDAG",
    # LLM
    "OllamaGenerator",
    "MockGenerator",
]
