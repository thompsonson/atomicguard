"""
Multi-Agent System (Extension 03: Definitions 19-20).

Implements:
- MultiAgentSystem: MAS = ⟨{Ag₁, ..., Agₙ}, ℛ, G⟩
- AgentState: σᵢ: G → {⊥, ⊤} - Agent's belief about workflow progress

All coordination happens through the shared repository (ℛ).
Agents do not communicate directly - they read/write artifacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atomicguard.domain.extraction import (
    AndPredicate,
    StatusPredicate,
    WorkflowPredicate,
    extract,
)
from atomicguard.domain.models import (
    AmbientEnvironment,
    ArtifactStatus,
    Context,
    GuardResult,
)

if TYPE_CHECKING:
    from atomicguard.domain.interfaces import ArtifactDAGInterface, GuardInterface


# =============================================================================
# AGENT STATE (Definition 20)
# =============================================================================


class AgentState:
    """
    Agent-local state σᵢ: G → {⊥, ⊤} (Definition 20).

    Value object derived from repository state.
    No stored state - always computed from artifacts via extraction.

    The agent's belief about which workflow steps are complete is
    derived from the presence of ACCEPTED artifacts in the shared repository.
    """

    def __init__(self, agent_id: str, repository: ArtifactDAGInterface) -> None:
        """Initialize agent state view.

        Args:
            agent_id: The workflow_id for this agent.
            repository: Shared artifact repository.
        """
        self._agent_id = agent_id
        self._repository = repository

    def is_step_complete(self, action_pair_id: str) -> bool:
        """Check if step g has an ACCEPTED artifact.

        σᵢ(g) = ⊤ when there exists an ACCEPTED artifact for this
        agent's workflow at the given action_pair_id.

        Args:
            action_pair_id: The step/guard identifier to check.

        Returns:
            True if an ACCEPTED artifact exists, False otherwise.
        """
        from atomicguard.domain.extraction import ActionPairPredicate

        # Build predicate: workflow_id matches AND action_pair matches AND status is ACCEPTED
        predicate = AndPredicate(
            WorkflowPredicate(self._agent_id),
            AndPredicate(
                ActionPairPredicate(action_pair_id),
                StatusPredicate(ArtifactStatus.ACCEPTED),
            ),
        )

        # Extract matching artifacts
        results = extract(self._repository, predicate, limit=1)

        return len(results) > 0


# =============================================================================
# AGENT RUNNER (Minimal interface for MAS)
# =============================================================================


class AgentRunner:
    """
    Minimal runner interface for MAS agents.

    Provides execute_step method expected by tests.
    The actual execution follows Definition 7 dynamics.
    """

    def __init__(
        self,
        agent_id: str,
        workflow: dict[str, Any],
        repository: ArtifactDAGInterface,
        guards: dict[str, GuardInterface],
    ) -> None:
        """Initialize agent runner.

        Args:
            agent_id: The workflow_id for this agent.
            workflow: Workflow definition dict with steps.
            repository: Shared artifact repository.
            guards: Shared guard library.
        """
        self._agent_id = agent_id
        self._workflow = workflow
        self._repository = repository
        self._guards = guards

    def execute_step(self, step_id: str) -> None:
        """Execute a single workflow step.

        Follows Definition 7 dynamics:
        1. Check preconditions
        2. Generate artifact
        3. Validate with guard
        4. Store in repository

        Args:
            step_id: The step/guard identifier to execute.
        """
        # This is a placeholder for the protocol
        # Actual execution would use DualStateAgent or similar
        pass


# =============================================================================
# MULTI-AGENT SYSTEM (Definition 19)
# =============================================================================


class MultiAgentSystem:
    """
    Multi-Agent System MAS = ⟨{Ag₁, ..., Agₙ}, ℛ, G⟩ (Definition 19).

    Aggregate root for multi-agent coordination.
    All coordination happens through the shared repository.

    Key invariants:
    - Agents communicate only through ℛ (no direct messages)
    - Guards are deterministic: G(r) same for all agents
    - Agent state σᵢ is derived from ℛ, not stored separately
    """

    def __init__(
        self,
        repository: ArtifactDAGInterface,
        guards: dict[str, GuardInterface] | None = None,
        agents: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize Multi-Agent System.

        Args:
            repository: Shared artifact repository ℛ.
            guards: Shared guard library G (guard_name -> GuardInterface).
            agents: Collection of workflow definitions (agent configurations).
        """
        self._repository = repository
        self._guards = guards or {}
        self._agents: list[dict[str, Any]] = agents or []
        self._agent_index: dict[str, dict[str, Any]] = {}

        # Index agents by id for fast lookup
        for agent in self._agents:
            agent_id = agent.get("id")
            if agent_id:
                self._agent_index[agent_id] = agent

    @property
    def repository(self) -> ArtifactDAGInterface:
        """Get shared repository ℛ."""
        return self._repository

    @property
    def guards(self) -> dict[str, GuardInterface]:
        """Get shared guard library G."""
        return self._guards

    @property
    def agents(self) -> list[dict[str, Any]]:
        """Get registered agent workflows."""
        return self._agents

    def register_agent(self, workflow: dict[str, Any]) -> None:
        """Register an agent workflow with the MAS.

        Args:
            workflow: Workflow definition dict with 'id' and 'steps'.
        """
        self._agents.append(workflow)
        agent_id = workflow.get("id")
        if agent_id:
            self._agent_index[agent_id] = workflow

    def get_agent_state(self, agent_id: str) -> AgentState:
        """Get agent's derived state σᵢ.

        State is computed from repository, not stored.

        Args:
            agent_id: The workflow_id for the agent.

        Returns:
            AgentState view for this agent.
        """
        return AgentState(agent_id, self._repository)

    def get_agent_runner(self, agent_id: str) -> AgentRunner:
        """Get runner for executing agent workflow steps.

        Args:
            agent_id: The workflow_id for the agent.

        Returns:
            AgentRunner for executing workflow steps.
        """
        workflow = self._agent_index.get(agent_id, {"id": agent_id, "steps": []})
        return AgentRunner(agent_id, workflow, self._repository, self._guards)

    def create_agent_context(self, agent_id: str) -> Context:
        """Create execution context for an agent.

        Each agent gets an independent context with its own workflow_id.

        Args:
            agent_id: The workflow_id for the agent.

        Returns:
            Context configured for this agent.
        """
        ambient = AmbientEnvironment(
            repository=self._repository,
            constraints="",
        )
        return Context(
            ambient=ambient,
            specification="",
            workflow_id=agent_id,
        )

    def evaluate_guard(
        self, _agent_id: str, guard_name: str, artifact_id: str
    ) -> GuardResult:
        """Evaluate a guard on an artifact.

        Guards are deterministic: same artifact → same result,
        regardless of which agent evaluates (Theorem 6).

        Args:
            _agent_id: The agent requesting evaluation (unused, for API consistency).
            guard_name: Name of guard in the guard library.
            artifact_id: ID of artifact to validate.

        Returns:
            GuardResult from validation.

        Raises:
            KeyError: If guard_name not in library or artifact not found.
        """
        if guard_name not in self._guards:
            raise KeyError(f"Guard '{guard_name}' not found in library")

        artifact = self._repository.get_artifact(artifact_id)
        if artifact is None:
            raise KeyError(f"Artifact '{artifact_id}' not found")

        guard = self._guards[guard_name]
        return guard.validate(artifact)
