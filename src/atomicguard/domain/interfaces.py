"""
Domain interfaces (Ports) for the Dual-State Framework.

These abstract base classes define the contracts that implementations must satisfy.
They have no external dependencies and represent the core domain boundaries.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from atomicguard.domain.models import (
        Artifact,
        Context,
        GuardResult,
    )

if TYPE_CHECKING:
    from atomicguard.domain.models import Artifact, Context, GuardResult
    from atomicguard.domain.prompts import PromptTemplate


class GeneratorInterface(ABC):
    """
    Port for artifact generation.

    Implementations connect to LLMs or other generation sources.

    Note (Side Effects & Idempotency):
        While generate() formally produces an artifact, implementations
        may induce side effects (filesystem I/O, API calls). In such cases:
        1. The artifact serves as a receipt/manifest of the operation
        2. Guards act as sensors verifying environmental state
        3. Side-effecting generators MUST be idempotent for retry safety
    """

    @abstractmethod
    def generate(
        self, context: "Context", template: Optional["PromptTemplate"] = None
    ) -> "Artifact":
        """
        Generate an artifact based on context.

        Args:
            context: The generation context including specification and feedback
            template: Optional prompt template for structured generation

        Returns:
            A new Artifact containing the generated content
        """
        pass


class GuardInterface(ABC):
    """
    Port for artifact validation.

    Guards are deterministic validators that return ⊤ (pass) or ⊥ (fail with feedback).
    """

    @abstractmethod
    def validate(
        self, artifact: "Artifact", **dependencies: "Artifact"
    ) -> "GuardResult":
        """
        Validate an artifact.

        Args:
            artifact: The artifact to validate
            **dependencies: Artifacts from prior workflow steps (key -> Artifact)

        Returns:
            GuardResult with passed=True/False and optional feedback
        """
        pass


class ArtifactDAGInterface(ABC):
    """
    Port for artifact persistence.

    Implementations provide append-only storage for the Versioned Repository (Definition 4).
    """

    @abstractmethod
    def store(self, artifact: "Artifact") -> str:
        """
        Store an artifact in the DAG.

        Args:
            artifact: The artifact to store

        Returns:
            The artifact_id
        """
        pass

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> "Artifact":
        """
        Retrieve an artifact by ID.

        Args:
            artifact_id: The unique identifier

        Returns:
            The artifact

        Raises:
            KeyError: If artifact not found
        """
        pass

    @abstractmethod
    def get_provenance(self, artifact_id: str) -> list["Artifact"]:
        """
        Trace the retry chain via previous_attempt_id.

        Args:
            artifact_id: Starting artifact

        Returns:
            List of artifacts from oldest to newest in the chain
        """
        pass
