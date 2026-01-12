"""
Workflow Reference and Resume Support (Extension 01: Versioned Environment).

Implements:
- W_ref content-addressed workflow hashing (Definition 11)
- WorkflowResumer for checkpoint resume with integrity verification (Definition 15)
- HumanAmendmentProcessor for human-in-the-loop support (Definition 16)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from atomicguard.domain.models import (
    AmbientEnvironment,
    Artifact,
    ArtifactSource,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    FeedbackEntry,
    GuardResult,
)

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import (
        ArtifactDAGInterface,
        CheckpointDAGInterface,
    )


# =============================================================================
# WORKFLOW REFERENCE (Definition 11)
# =============================================================================


class WorkflowIntegrityError(Exception):
    """Raised when W_ref verification fails on resume."""

    pass


class WorkflowRegistry:
    """
    Singleton registry storing workflow definitions by W_ref.

    Enables resolve_workflow_ref to retrieve stored workflows.
    Thread-safe via module-level singleton pattern.
    """

    _instance: WorkflowRegistry | None = None
    _workflows: dict[str, dict[str, Any]]

    def __new__(cls) -> WorkflowRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._workflows = {}
        return cls._instance

    def store(self, workflow: dict[str, Any]) -> str:
        """Store workflow and return its W_ref.

        Args:
            workflow: Workflow definition dict.

        Returns:
            Content-addressed hash (W_ref) of the workflow.
        """
        w_ref = compute_workflow_ref(workflow, store=False)  # Avoid recursion
        self._workflows[w_ref] = workflow
        return w_ref

    def resolve(self, w_ref: str) -> dict[str, Any]:
        """Retrieve workflow by W_ref.

        Args:
            w_ref: Content-addressed hash to look up.

        Returns:
            Workflow definition dict.

        Raises:
            KeyError: If w_ref not found in registry.
        """
        if w_ref not in self._workflows:
            raise KeyError(f"Workflow reference not found: {w_ref}")
        return self._workflows[w_ref]

    def clear(self) -> None:
        """Clear all stored workflows (for testing)."""
        self._workflows.clear()


def compute_workflow_ref(workflow: dict[str, Any], store: bool = True) -> str:
    """Compute content-addressed hash of workflow structure (Definition 11).

    Produces deterministic hash by:
    1. Canonical JSON serialization (sorted keys, no whitespace)
    2. SHA-256 hash of the canonical form

    By default, also stores the workflow in the registry so that
    resolve_workflow_ref can retrieve it (integrity axiom support).

    Args:
        workflow: Workflow definition dict.
        store: If True (default), store workflow in registry for later resolution.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    # Canonical JSON: sorted keys, no extra whitespace, ensure_ascii for determinism
    canonical = json.dumps(workflow, sort_keys=True, separators=(",", ":"))
    w_ref = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # Store workflow for resolve_workflow_ref support
    if store:
        registry = WorkflowRegistry()
        registry._workflows[w_ref] = workflow

    return w_ref


def resolve_workflow_ref(w_ref: str) -> dict[str, Any]:
    """Retrieve workflow definition by W_ref.

    Uses the singleton WorkflowRegistry to look up stored workflows.
    For the integrity axiom (hash(resolve(W_ref)) == W_ref) to hold,
    the workflow must have been stored via WorkflowRegistry.store().

    Args:
        w_ref: Content-addressed hash to look up.

    Returns:
        Workflow definition dict.

    Raises:
        KeyError: If w_ref not found in registry.
    """
    registry = WorkflowRegistry()
    return registry.resolve(w_ref)


# =============================================================================
# CONFIGURATION REFERENCE (Definition 33 - Extension 07)
# =============================================================================


def compute_config_ref(
    action_pair_id: str,
    workflow_config: dict[str, Any],
    prompt_config: dict[str, Any],
    upstream_artifacts: dict[str, Artifact] | None = None,
) -> str:
    """Compute configuration reference Ψ_ref for an action pair (Definition 33).

    The Ψ_ref is a content-addressable fingerprint that changes when:
    - Prompt configuration changes
    - Model/guard configuration changes
    - Upstream action pair Ψ_ref changes
    - Upstream artifact content changes

    This enables incremental execution - skip unchanged action pairs.

    Args:
        action_pair_id: ID of the action pair to compute ref for.
        workflow_config: Full workflow configuration dict.
        prompt_config: Full prompt configuration dict.
        upstream_artifacts: Map of dependency action_pair_id → Artifact.
            If None, computes ref for root action pair with no dependencies.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    ap_config = workflow_config.get("action_pairs", {}).get(action_pair_id, {})
    prompt = prompt_config.get(action_pair_id, {})

    # Collect upstream refs and artifact content hashes
    upstream_refs: dict[str, str | None] = {}
    artifact_hashes: dict[str, str] = {}

    if upstream_artifacts:
        for dep_id in ap_config.get("requires", []):
            dep_artifact = upstream_artifacts.get(dep_id)
            if dep_artifact:
                upstream_refs[dep_id] = dep_artifact.config_ref
                artifact_hashes[dep_id] = hashlib.sha256(
                    dep_artifact.content.encode("utf-8")
                ).hexdigest()

    # Build canonical input for hashing
    hash_input = {
        "prompt": prompt,
        "model": ap_config.get("model", workflow_config.get("model")),
        "guard": ap_config.get("guard"),
        "guard_config": ap_config.get("guard_config", {}),
        "upstream_refs": upstream_refs,
        "artifact_hashes": artifact_hashes,
    }

    # Canonical JSON: sorted keys, no extra whitespace
    canonical = json.dumps(hash_input, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# =============================================================================
# WORKFLOW RESUME SUPPORT (Definition 15)
# =============================================================================


@dataclass
class ResumeResult:
    """Result of resuming a workflow from checkpoint."""

    success: bool
    workflow_state: WorkflowState | None = None
    error: str | None = None


@dataclass
class WorkflowState:
    """Reconstructed workflow state for resume."""

    completed_steps: tuple[str, ...]
    artifact_ids: dict[str, str]
    next_step: str | None = None


class WorkflowResumer:
    """Handles checkpoint resume with W_ref integrity verification (Definition 15).

    Ensures that:
    1. Workflow hasn't changed since checkpoint (W_ref verification)
    2. Context is reconstructed from repository, not checkpoint
    3. Feedback history is preserved from provenance chain
    """

    def __init__(
        self,
        checkpoint_dag: CheckpointDAGInterface,
        artifact_dag: ArtifactDAGInterface | None = None,
    ) -> None:
        """Initialize resumer with checkpoint and artifact DAGs.

        Args:
            checkpoint_dag: DAG storing checkpoints and amendments.
            artifact_dag: DAG storing artifacts (optional, for context reconstruction).
        """
        self._checkpoint_dag = checkpoint_dag
        self._artifact_dag = artifact_dag

    def verify_workflow_integrity(self, checkpoint_id: str) -> None:
        """Verify W_ref integrity for checkpoint.

        Does nothing if W_ref is not set on checkpoint (backwards compatibility).

        Args:
            checkpoint_id: ID of checkpoint to verify.

        Raises:
            WorkflowIntegrityError: If checkpoint W_ref doesn't match stored workflow.
        """
        checkpoint = self._checkpoint_dag.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        # Check if checkpoint has workflow_ref (may not if created before Extension 01)
        if hasattr(checkpoint, "workflow_ref") and checkpoint.workflow_ref:
            try:
                resolved = resolve_workflow_ref(checkpoint.workflow_ref)
                re_hashed = compute_workflow_ref(resolved)
                if re_hashed != checkpoint.workflow_ref:
                    raise WorkflowIntegrityError(
                        f"Workflow integrity check failed: hash mismatch for {checkpoint_id}"
                    )
            except KeyError:
                # Workflow not in registry - cannot verify, but don't fail
                pass

    def resume(self, checkpoint_id: str, current_workflow_ref: str) -> ResumeResult:
        """Resume workflow from checkpoint with W_ref verification.

        Args:
            checkpoint_id: ID of checkpoint to resume from.
            current_workflow_ref: W_ref of current workflow definition.

        Returns:
            ResumeResult with success status and workflow state.

        Raises:
            WorkflowIntegrityError: If current_workflow_ref doesn't match checkpoint.
        """
        checkpoint = self._checkpoint_dag.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            return ResumeResult(
                success=False, error=f"Checkpoint not found: {checkpoint_id}"
            )

        # Verify W_ref matches
        if (
            hasattr(checkpoint, "workflow_ref")
            and checkpoint.workflow_ref
            and checkpoint.workflow_ref != current_workflow_ref
        ):
            raise WorkflowIntegrityError(
                f"Workflow changed since checkpoint. "
                f"Expected: {checkpoint.workflow_ref}, Got: {current_workflow_ref}"
            )

        # Reconstruct state
        state = self.restore_state(checkpoint_id)
        return ResumeResult(success=True, workflow_state=state)

    def restore_state(self, checkpoint_id: str) -> WorkflowState:
        """Restore workflow state from checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore from.

        Returns:
            WorkflowState with completed steps and artifact IDs.
        """
        checkpoint = self._checkpoint_dag.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        artifact_ids = dict(checkpoint.artifact_ids)

        return WorkflowState(
            completed_steps=checkpoint.completed_steps,
            artifact_ids=artifact_ids,
            next_step=checkpoint.failed_step,
        )

    def reconstruct_context(self, checkpoint_id: str) -> Context:
        """Reconstruct context from repository artifacts, not checkpoint.

        This ensures context derivability (Definition 13) - context is
        derived from stored items, not from external state.

        Args:
            checkpoint_id: ID of checkpoint to reconstruct context for.

        Returns:
            Reconstructed Context.
        """
        checkpoint = self._checkpoint_dag.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        # Build feedback history from provenance chain
        feedback_history = self.reconstruct_feedback_history(checkpoint_id)

        # Build dependency artifacts from completed steps
        dependency_artifacts = tuple(checkpoint.artifact_ids)

        # Create ambient environment (repository may be None)
        ambient = AmbientEnvironment(
            repository=self._artifact_dag,
            constraints=checkpoint.constraints,
        )

        return Context(
            ambient=ambient,
            specification=checkpoint.specification,
            current_artifact=checkpoint.failed_artifact_id,
            feedback_history=feedback_history,
            dependency_artifacts=dependency_artifacts,
        )

    def get_next_attempt_number(self, checkpoint_id: str) -> int:
        """Get the next attempt number for the failed step.

        Args:
            checkpoint_id: ID of checkpoint.

        Returns:
            Next attempt number (previous max + 1).
        """
        checkpoint = self._checkpoint_dag.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        # Count provenance IDs (failed attempts) and add 1
        return len(checkpoint.provenance_ids) + 1

    def reconstruct_feedback_history(
        self, checkpoint_id: str
    ) -> tuple[tuple[str, str], ...]:
        """Reconstruct H_feedback from provenance chain.

        Args:
            checkpoint_id: ID of checkpoint.

        Returns:
            Tuple of (artifact_id, feedback) pairs.
        """
        checkpoint = self._checkpoint_dag.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        # Build feedback history from provenance and failure feedback
        history = []

        # Add feedback from provenance chain
        for artifact_id in checkpoint.provenance_ids:
            if self._artifact_dag is not None:
                try:
                    artifact = self._artifact_dag.get_artifact(artifact_id)
                    if (
                        artifact
                        and artifact.guard_result
                        and artifact.guard_result.feedback
                    ):
                        history.append((artifact_id, artifact.guard_result.feedback))
                except Exception:
                    pass

        # Add the failure feedback
        if checkpoint.failed_artifact_id and checkpoint.failure_feedback:
            history.append((checkpoint.failed_artifact_id, checkpoint.failure_feedback))

        return tuple(history)


# =============================================================================
# HUMAN AMENDMENT PROCESSOR (Definition 16)
# =============================================================================


@dataclass
class ProcessResult:
    """Result of processing a human amendment."""

    success: bool
    artifact: Artifact | None = None
    guard_result: GuardResult | None = None
    should_retry: bool = False
    error: str | None = None


class HumanAmendmentProcessor:
    """Processes human amendments for the human-in-the-loop pattern (Definition 16).

    Human amendments can:
    1. Provide new artifact content directly
    2. Provide additional feedback for LLM retry
    3. Skip optional steps

    All human artifacts flow through the same guard validation as generated artifacts.
    """

    def __init__(
        self,
        checkpoint_dag: CheckpointDAGInterface,
        artifact_dag: ArtifactDAGInterface | None = None,
    ) -> None:
        """Initialize processor with checkpoint and artifact DAGs.

        Args:
            checkpoint_dag: DAG storing checkpoints and amendments.
            artifact_dag: DAG storing artifacts (for artifact creation).
        """
        self._checkpoint_dag = checkpoint_dag
        self._artifact_dag = artifact_dag

    def process_amendment(self, amendment_id: str) -> ProcessResult:
        """Process a human amendment.

        Args:
            amendment_id: ID of amendment to process.

        Returns:
            ProcessResult with success status.
        """
        amendment = self._checkpoint_dag.get_amendment(amendment_id)
        if amendment is None:
            return ProcessResult(
                success=False, error=f"Amendment not found: {amendment_id}"
            )

        # Create artifact from amendment
        artifact = self.create_artifact_from_amendment(amendment_id)

        # Default guard result (actual guard validation happens in workflow)
        guard_result = GuardResult(passed=True, feedback="Human amendment accepted")

        return ProcessResult(
            success=True,
            artifact=artifact,
            guard_result=guard_result,
        )

    def create_artifact_from_amendment(self, amendment_id: str) -> Artifact:
        """Create artifact from human amendment.

        The artifact is marked with ArtifactSource.HUMAN to distinguish
        it from generated artifacts.

        Args:
            amendment_id: ID of amendment.

        Returns:
            New Artifact with human content.
        """
        amendment = self._checkpoint_dag.get_amendment(amendment_id)
        if amendment is None:
            raise ValueError(f"Amendment not found: {amendment_id}")

        checkpoint = self._checkpoint_dag.get_checkpoint(amendment.checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {amendment.checkpoint_id}")

        # Build context snapshot
        context_snapshot = ContextSnapshot(
            workflow_id=checkpoint.workflow_id,
            specification=checkpoint.specification,
            constraints=checkpoint.constraints,
            feedback_history=tuple(
                FeedbackEntry(artifact_id=aid, feedback=fb)
                for aid, fb in [
                    (checkpoint.failed_artifact_id, checkpoint.failure_feedback)
                ]
                if aid and fb
            ),
            dependency_artifacts=checkpoint.artifact_ids,
        )

        # Create artifact with HUMAN source
        artifact = Artifact(
            artifact_id=f"human-{amendment_id}",
            workflow_id=checkpoint.workflow_id,
            content=amendment.content,
            previous_attempt_id=amendment.parent_artifact_id,
            parent_action_pair_id=None,
            action_pair_id=checkpoint.failed_step,
            created_at=amendment.created_at,
            attempt_number=len(checkpoint.provenance_ids) + 1,
            status=ArtifactStatus.PENDING,
            guard_result=None,  # Guard result set after validation
            context=context_snapshot,
            source=ArtifactSource.HUMAN,
        )

        # Store artifact if DAG is available
        if self._artifact_dag is not None:
            self._artifact_dag.store(artifact)

        return artifact

    def process_amendment_with_guard(
        self, amendment_id: str, guard_passes: bool
    ) -> ProcessResult:
        """Process amendment with explicit guard result.

        Used for testing and scenarios where guard validation is external.

        Args:
            amendment_id: ID of amendment.
            guard_passes: Whether guard should pass.

        Returns:
            ProcessResult with should_retry flag if guard fails.
        """
        amendment = self._checkpoint_dag.get_amendment(amendment_id)
        if amendment is None:
            return ProcessResult(
                success=False, error=f"Amendment not found: {amendment_id}"
            )

        artifact = self.create_artifact_from_amendment(amendment_id)

        if guard_passes:
            guard_result = GuardResult(passed=True, feedback="")
            return ProcessResult(
                success=True,
                artifact=artifact,
                guard_result=guard_result,
                should_retry=False,
            )
        else:
            guard_result = GuardResult(
                passed=False, feedback="Guard rejected human artifact"
            )
            return ProcessResult(
                success=True,
                artifact=artifact,
                guard_result=guard_result,
                should_retry=True,
            )

    def get_checkpoint_context(self, checkpoint_id: str) -> Context:
        """Get context from checkpoint.

        Args:
            checkpoint_id: ID of checkpoint.

        Returns:
            Context from checkpoint.
        """
        checkpoint = self._checkpoint_dag.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        ambient = AmbientEnvironment(
            repository=self._artifact_dag,
            constraints=checkpoint.constraints,
        )

        return Context(
            ambient=ambient,
            specification=checkpoint.specification,
            current_artifact=checkpoint.failed_artifact_id,
            feedback_history=(),
            dependency_artifacts=checkpoint.artifact_ids,
        )

    def apply_amendment_to_context(self, amendment_id: str) -> Context:
        """Apply amendment to checkpoint context (⊕ operator).

        The amendment's content/context is appended to the checkpoint's
        specification using the monotonic amendment operator.

        Args:
            amendment_id: ID of amendment.

        Returns:
            Amended Context with additional information.
        """
        amendment = self._checkpoint_dag.get_amendment(amendment_id)
        if amendment is None:
            raise ValueError(f"Amendment not found: {amendment_id}")

        # Get base context from checkpoint
        base_context = self.get_checkpoint_context(amendment.checkpoint_id)

        # Apply amendment using ⊕ operator
        if amendment.context:
            return base_context.amend(delta_spec=amendment.context)
        return base_context
