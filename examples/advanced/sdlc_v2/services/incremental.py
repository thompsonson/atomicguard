"""
Incremental Execution Service (Extension 07: Definitions 33-37).

Implements:
- Configuration reference (Ψ_ref) computation for action pairs
- Change detection via unchanged() predicate
- Invalidation cascade algorithm (Algorithm 1)
- Cached artifact lookup by config_ref
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from atomicguard.domain.models import Artifact, ArtifactStatus
from atomicguard.domain.workflow import compute_config_ref

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import ArtifactDAGInterface


@dataclass
class ExecutionDecision:
    """Result of should_execute check."""

    should_execute: bool
    config_ref: str
    cached_artifact: Artifact | None = None
    reason: str = ""


class IncrementalExecutionService:
    """
    Service for incremental workflow execution (Extension 07).

    Enables selective re-execution based on configuration changes:
    - Skips action pairs with unchanged config_ref and existing accepted artifacts
    - Propagates changes through dependency graph (Merkle propagation)
    - Provides invalidation cascade for partial re-execution
    """

    def __init__(self, artifact_dag: ArtifactDAGInterface) -> None:
        """Initialize with artifact DAG for cache lookups.

        Args:
            artifact_dag: Repository for artifact storage and retrieval.
        """
        self._artifact_dag = artifact_dag

    def should_execute(
        self,
        action_pair_id: str,
        workflow_config: dict,
        prompt_config: dict,
        upstream_artifacts: dict[str, Artifact] | None = None,
    ) -> ExecutionDecision:
        """Determine if action pair should execute (Definition 36).

        unchanged(ap) ⟺ ∃a ∈ ℛ : a.config_ref = Ψ_ref_current(ap) ∧ a.status = ACCEPTED

        Args:
            action_pair_id: ID of the action pair to check.
            workflow_config: Full workflow configuration dict.
            prompt_config: Full prompt configuration dict.
            upstream_artifacts: Map of dependency action_pair_id → Artifact.

        Returns:
            ExecutionDecision with should_execute flag, config_ref, and cached artifact if found.
        """
        # Compute current config_ref (Ψ_ref)
        config_ref = compute_config_ref(
            action_pair_id,
            workflow_config,
            prompt_config,
            upstream_artifacts,
        )

        # Check for existing accepted artifact with same config_ref
        cached = self.lookup_by_config_ref(action_pair_id, config_ref)

        if cached is not None:
            return ExecutionDecision(
                should_execute=False,
                config_ref=config_ref,
                cached_artifact=cached,
                reason="unchanged (config_ref match)",
            )

        return ExecutionDecision(
            should_execute=True,
            config_ref=config_ref,
            cached_artifact=None,
            reason="changed or new",
        )

    def lookup_by_config_ref(
        self, action_pair_id: str, config_ref: str
    ) -> Artifact | None:
        """Find cached artifact matching config_ref (Definition 36).

        Searches the repository for an accepted artifact with matching
        action_pair_id and config_ref.

        Args:
            action_pair_id: Action pair ID to match.
            config_ref: Configuration reference (Ψ_ref) to match.

        Returns:
            Matching artifact if found, None otherwise.
        """
        for artifact in self._artifact_dag.get_all():
            if (
                artifact.action_pair_id == action_pair_id
                and artifact.config_ref == config_ref
                and artifact.status == ArtifactStatus.ACCEPTED
            ):
                return artifact
        return None

    def get_invalidated(
        self,
        changed_aps: set[str],
        dependency_graph: dict[str, list[str]],
    ) -> set[str]:
        """Compute all invalidated action pairs (Algorithm 1).

        Given a set of directly changed action pairs, compute all action pairs
        that need re-execution due to dependency cascade.

        Args:
            changed_aps: Set of action pair IDs with direct changes.
            dependency_graph: Map of action_pair_id → list of upstream dependencies.

        Returns:
            Set of all action pair IDs that need re-execution.
        """
        invalidated = set(changed_aps)

        # Get all action pairs from graph
        all_aps = set(dependency_graph.keys())
        for deps in dependency_graph.values():
            all_aps.update(deps)

        # Process in topological order, propagating invalidation
        sorted_aps = self._topological_sort(dependency_graph)

        for ap in sorted_aps:
            deps = dependency_graph.get(ap, [])
            if any(dep in invalidated for dep in deps):
                invalidated.add(ap)

        return invalidated

    def _topological_sort(self, dependency_graph: dict[str, list[str]]) -> list[str]:
        """Sort action pairs in topological order.

        Root nodes (no dependencies) come first, leaves come last.

        Args:
            dependency_graph: Map of action_pair_id → list of upstream dependencies.

        Returns:
            List of action pair IDs in topological order.
        """
        # Collect all nodes
        all_nodes = set(dependency_graph.keys())
        for deps in dependency_graph.values():
            all_nodes.update(deps)

        # Kahn's algorithm
        in_degree = dict.fromkeys(all_nodes, 0)
        for ap, deps in dependency_graph.items():
            # ap depends on deps, so ap has incoming edges equal to number of deps
            in_degree[ap] = len(deps)

        # Start with nodes that have no dependencies
        queue = [node for node in all_nodes if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Find all nodes that depend on this node
            for ap, deps in dependency_graph.items():
                if node in deps:
                    in_degree[ap] -= 1
                    if in_degree[ap] == 0:
                        queue.append(ap)

        return result

    def compute_execution_plan(
        self,
        workflow_config: dict,
        prompt_config: dict,
    ) -> dict[str, ExecutionDecision]:
        """Compute execution decisions for all action pairs.

        Evaluates each action pair in dependency order, using cached artifacts
        from previously executed steps as upstream dependencies.

        Args:
            workflow_config: Full workflow configuration dict.
            prompt_config: Full prompt configuration dict.

        Returns:
            Map of action_pair_id → ExecutionDecision.
        """
        action_pairs = workflow_config.get("action_pairs", {})

        # Build dependency graph
        dep_graph = {
            ap_id: ap_config.get("requires", [])
            for ap_id, ap_config in action_pairs.items()
        }

        # Sort in execution order
        sorted_aps = self._topological_sort(dep_graph)

        # Track decisions and collected artifacts
        decisions: dict[str, ExecutionDecision] = {}
        artifacts: dict[str, Artifact] = {}

        for ap_id in sorted_aps:
            # Get upstream artifacts from prior decisions
            upstream = {}
            for dep_id in dep_graph.get(ap_id, []):
                if dep_id in decisions:
                    cached = decisions[dep_id].cached_artifact
                    if cached:
                        upstream[dep_id] = cached

            decision = self.should_execute(
                ap_id,
                workflow_config,
                prompt_config,
                upstream or None,
            )
            decisions[ap_id] = decision

            # Track artifact for downstream dependencies
            if decision.cached_artifact:
                artifacts[ap_id] = decision.cached_artifact

        return decisions
