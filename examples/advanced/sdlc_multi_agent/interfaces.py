"""
Interface definitions for Multi-Agent SDLC workflow.

Defines clear contracts between components to enforce separation of concerns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


# =============================================================================
# Workspace Service Interface (Layer 3a)
# =============================================================================


@dataclass
class WorkspaceManifest:
    """Represents files in a workspace."""

    files: list[dict[str, str]]  # [{"path": "...", "content": "..."}]
    metadata: dict[str, Any]


class IWorkspaceService(ABC):
    """Interface for filesystem ↔ DAG synchronization.

    Responsibilities:
    - Materialize artifacts to filesystem
    - Capture filesystem changes to artifact format
    - Manage workspace lifecycle

    Does NOT:
    - Validate content (Guard's job)
    - Decide what to materialize (Orchestrator's job)
    - Store artifacts (DAG's job)
    """

    @abstractmethod
    def create_workspace(self, phase_id: str) -> Path:
        """Create an ephemeral workspace for a phase.

        Args:
            phase_id: Identifier for the phase (e.g., "g_ddd")

        Returns:
            Path to the workspace directory
        """
        pass

    @abstractmethod
    def materialize(self, manifest: WorkspaceManifest, workspace: Path) -> None:
        """Write artifact files to filesystem.

        Args:
            manifest: Files to materialize
            workspace: Target directory

        Raises:
            IOError: If filesystem write fails
        """
        pass

    @abstractmethod
    def capture(
        self, workspace: Path, patterns: list[str] | None = None
    ) -> WorkspaceManifest:
        """Capture filesystem changes into artifact format.

        Args:
            workspace: Source directory
            patterns: Glob patterns (default: ["**/*.py", "**/*.md"])

        Returns:
            WorkspaceManifest ready for artifact storage
        """
        pass

    @abstractmethod
    def cleanup(self, workspace: Path) -> None:
        """Remove workspace directory.

        Args:
            workspace: Directory to remove
        """
        pass


# =============================================================================
# Generator Interface (Layer 3b)
# =============================================================================


@dataclass
class GeneratorResult:
    """Result from a generator execution."""

    content: str  # Generated content (typically JSON string)
    metadata: dict[str, Any]  # Token counts, model info, etc.
    raw_messages: list[dict]  # Raw LLM messages for provenance


class IGenerator(ABC):
    """Interface for LLM-based content generation.

    Responsibilities:
    - Call LLM with prompt
    - Use Claude SDK skills
    - Format input/output

    Does NOT:
    - Validate (Guard's job)
    - Retry (Orchestrator's job)
    - Manage filesystem (WorkspaceService's job)
    - Store artifacts (DAG's job)
    """

    @abstractmethod
    async def generate(
        self, prompt: str, workspace: Path, context: dict[str, Any]
    ) -> GeneratorResult:
        """Generate content using LLM.

        Args:
            prompt: Instruction for the LLM
            workspace: Working directory (for Claude SDK)
            context: Additional context (dependencies, etc.)

        Returns:
            GeneratorResult with content and metadata
        """
        pass


# =============================================================================
# Guard Interface (Layer 3c)
# =============================================================================


@dataclass
class GuardResult:
    """Result from guard validation."""

    passed: bool
    feedback: str  # Empty if passed, error message if failed
    artifacts: dict[str, Any] | None = None  # Extracted metadata


class IGuard(ABC):
    """Interface for deterministic validation.

    Responsibilities:
    - Validate artifact content OR filesystem state
    - Return clear pass/fail verdict with feedback

    Does NOT:
    - Call LLM (must be deterministic)
    - Retry (Orchestrator's job)
    - Store artifacts (DAG's job)
    - Materialize files (WorkspaceService's job)
    """

    @abstractmethod
    def validate(
        self, manifest: WorkspaceManifest, workspace: Path, context: dict[str, Any]
    ) -> GuardResult:
        """Validate generated content.

        Args:
            manifest: Artifact content to validate
            workspace: Filesystem location (for tool-based validation)
            context: Dependencies and configuration

        Returns:
            GuardResult with verdict and feedback
        """
        pass


# =============================================================================
# Orchestrator Interface (Layer 2)
# =============================================================================


@dataclass
class PhaseResult:
    """Result from a single phase execution."""

    phase_id: str
    success: bool
    artifact_id: str | None  # None if failed
    attempts: int
    feedback: str  # Empty if success, error if failed


@dataclass
class WorkflowResult:
    """Result from full workflow execution."""

    success: bool
    completed_phases: list[str]
    failed_phase: str | None
    phase_results: dict[str, PhaseResult]
    total_attempts: int
    budget_remaining: int


class IOrchestrator(ABC):
    """Interface for workflow coordination.

    Responsibilities:
    - Phase sequencing (DDD → Coder → Tester)
    - Retry budget management
    - Coordinate WorkspaceService, Generators, Guards
    - Store artifacts in DAG

    Does NOT:
    - Know how generators work internally
    - Know how guards validate
    - Know how workspace materializes files
    """

    @abstractmethod
    async def execute(self, user_intent: str) -> WorkflowResult:
        """Execute full workflow from user intent.

        Args:
            user_intent: Natural language requirements

        Returns:
            WorkflowResult with success status and artifacts
        """
        pass


# =============================================================================
# Type Aliases for Clarity
# =============================================================================

ArtifactID = str
PhaseID = str
WorkspaceID = str
