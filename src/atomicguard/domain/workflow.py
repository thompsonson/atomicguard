"""
Workflow Reference Support (Extension 01: Versioned Environment).

Implements:
- W_ref content-addressed workflow hashing (Definition 11)
- Configuration reference Ψ_ref for incremental execution (Definition 33)
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atomicguard.domain.models import Artifact


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
