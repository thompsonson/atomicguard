"""
Multi-Agent SDLC Orchestrator.

Coordinates three-phase workflow: DDD → Coder → Tester

Responsibilities:
- Phase sequencing and dependency management
- Retry budget management per phase
- Coordinate WorkspaceService, Generators, Guards
- Store artifacts in DAG (shared source of truth)

Does NOT:
- Know how generators work internally
- Know how guards validate
- Know how workspace materializes files
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atomicguard.domain.models import Artifact, ArtifactSource, ArtifactStatus
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

from .interfaces import (
    IGenerator,
    IGuard,
    IOrchestrator,
    PhaseResult,
    WorkflowResult,
)
from .workspace_service import WorkspaceService


class SDLCOrchestrator(IOrchestrator):
    """Orchestrates Multi-Agent SDLC workflow.

    Three-phase workflow:
    1. DDD Agent: Generate documentation (docs/)
    2. Coder Agent: Generate implementation (src/)
    3. Tester Agent: Validate tests pass

    Key design:
    - DAG is the shared source of truth
    - Workspaces are ephemeral (but persisted for debugging)
    - Option C: Pre-materialize ALL upstream artifacts before phase starts
    """

    # Retry budget per phase
    PHASE_BUDGETS = {
        "g_ddd": 5,
        "g_coder": 11,
        "g_tester": 7,
    }

    def __init__(
        self,
        artifact_dag: FilesystemArtifactDAG,
        workspace_service: WorkspaceService,
        generators: dict[str, IGenerator],
        guards: dict[str, IGuard],
        prompts: dict[str, str],
    ):
        """Initialize orchestrator.

        Args:
            artifact_dag: DAG for artifact storage (source of truth)
            workspace_service: Service for filesystem synchronization
            generators: Map of phase_id -> Generator
            guards: Map of phase_id -> Guard
            prompts: Map of phase_id -> Prompt template
        """
        self.artifact_dag = artifact_dag
        self.workspace_service = workspace_service
        self.generators = generators
        self.guards = guards
        self.prompts = prompts

        self.workflow_id = str(uuid.uuid4())
        self.total_budget = sum(self.PHASE_BUDGETS.values())
        self.budget_used = 0

    async def execute(self, user_intent: str) -> WorkflowResult:
        """Execute full workflow from user intent.

        Workflow:
        1. g_ddd: Generate DDD docs
        2. g_coder: Generate implementation (depends on g_ddd)
        3. g_tester: Validate tests (depends on g_coder)

        Args:
            user_intent: Natural language requirements

        Returns:
            WorkflowResult with success status and artifacts
        """
        phase_results = {}
        completed_phases = []
        artifacts = {}  # phase_id -> Artifact

        # Phase 1: DDD Agent
        ddd_result = await self._execute_phase(
            phase_id="g_ddd",
            prompt=self.prompts["g_ddd"].format(user_intent=user_intent),
            dependencies={},  # No dependencies
        )
        phase_results["g_ddd"] = ddd_result

        if not ddd_result.success:
            return WorkflowResult(
                success=False,
                completed_phases=completed_phases,
                failed_phase="g_ddd",
                phase_results=phase_results,
                total_attempts=self.budget_used,
                budget_remaining=self.total_budget - self.budget_used,
            )

        completed_phases.append("g_ddd")
        artifacts["g_ddd"] = self.artifact_dag.get_artifact(ddd_result.artifact_id)

        # Phase 2: Coder Agent (depends on DDD)
        coder_result = await self._execute_phase(
            phase_id="g_coder",
            prompt=self.prompts["g_coder"],
            dependencies={"g_ddd": artifacts["g_ddd"]},
        )
        phase_results["g_coder"] = coder_result

        if not coder_result.success:
            return WorkflowResult(
                success=False,
                completed_phases=completed_phases,
                failed_phase="g_coder",
                phase_results=phase_results,
                total_attempts=self.budget_used,
                budget_remaining=self.total_budget - self.budget_used,
            )

        completed_phases.append("g_coder")
        artifacts["g_coder"] = self.artifact_dag.get_artifact(coder_result.artifact_id)

        # Phase 3: Tester Agent (depends on Coder)
        tester_result = await self._execute_phase(
            phase_id="g_tester",
            prompt=self.prompts["g_tester"],
            dependencies={"g_coder": artifacts["g_coder"]},
        )
        phase_results["g_tester"] = tester_result

        if not tester_result.success:
            return WorkflowResult(
                success=False,
                completed_phases=completed_phases,
                failed_phase="g_tester",
                phase_results=phase_results,
                total_attempts=self.budget_used,
                budget_remaining=self.total_budget - self.budget_used,
            )

        completed_phases.append("g_tester")

        # All phases succeeded
        return WorkflowResult(
            success=True,
            completed_phases=completed_phases,
            failed_phase=None,
            phase_results=phase_results,
            total_attempts=self.budget_used,
            budget_remaining=self.total_budget - self.budget_used,
        )

    async def _execute_phase(
        self,
        phase_id: str,
        prompt: str,
        dependencies: dict[str, Artifact],
    ) -> PhaseResult:
        """Execute a single phase with retry logic.

        Args:
            phase_id: Phase identifier (e.g., "g_ddd")
            prompt: Prompt for the generator
            dependencies: Upstream artifacts (phase_id -> Artifact)

        Returns:
            PhaseResult with success status and artifact_id
        """
        budget = self.PHASE_BUDGETS.get(phase_id, 3)
        generator = self.generators[phase_id]
        guard = self.guards[phase_id]

        current_prompt = prompt
        feedback_history = []

        for attempt in range(1, budget + 1):
            if self.budget_used >= self.total_budget:
                return PhaseResult(
                    phase_id=phase_id,
                    success=False,
                    artifact_id=None,
                    attempts=attempt,
                    feedback="Total workflow budget exhausted",
                )

            self.budget_used += 1

            # Create workspace for this attempt
            workspace = self.workspace_service.create_workspace(
                f"{phase_id}_attempt_{attempt}"
            )

            # Option C: Pre-materialize ALL upstream artifacts (eager, full context)
            for dep_phase_id, dep_artifact in dependencies.items():
                dep_manifest = self.workspace_service.manifest_from_artifact_content(
                    dep_artifact.content
                )
                self.workspace_service.materialize(dep_manifest, workspace)

            # Run generator
            context = {
                "dependencies": dependencies,
                "attempt": attempt,
                "feedback_history": feedback_history,
            }

            try:
                generator_result = await generator.generate(
                    prompt=current_prompt,
                    workspace=workspace,
                    context=context,
                )
            except Exception as e:
                return PhaseResult(
                    phase_id=phase_id,
                    success=False,
                    artifact_id=None,
                    attempts=attempt,
                    feedback=f"Generator failed: {e}",
                )

            # Parse generator output into manifest
            manifest = self.workspace_service.manifest_from_artifact_content(
                generator_result.content
            )

            # Run guard
            guard_result = guard.validate(
                manifest=manifest,
                workspace=workspace,
                context=context,
            )

            if guard_result.passed:
                # Success! Store artifact in DAG
                artifact_id = str(uuid.uuid4())
                artifact = Artifact(
                    artifact_id=artifact_id,
                    workflow_id=self.workflow_id,
                    action_pair_id=phase_id,
                    content=generator_result.content,
                    created_at=datetime.now(UTC).isoformat(),
                    status=ArtifactStatus.ACCEPTED,
                    source=ArtifactSource.GENERATED,
                    parent_artifact_id=None,
                    dependencies=tuple(dependencies.keys()),
                    metadata=generator_result.metadata,
                )
                self.artifact_dag.store(artifact)

                return PhaseResult(
                    phase_id=phase_id,
                    success=True,
                    artifact_id=artifact_id,
                    attempts=attempt,
                    feedback="",
                )
            else:
                # Failed - add feedback and retry
                feedback_history.append(
                    {
                        "attempt": attempt,
                        "feedback": guard_result.feedback,
                    }
                )

                # Update prompt with feedback
                current_prompt = f"""{prompt}

# Previous Attempt Failed

Attempt {attempt} feedback:
{guard_result.feedback}

Please fix the issues and try again.
"""

        # Exhausted retry budget
        final_feedback = "\n".join(
            f"Attempt {f['attempt']}: {f['feedback']}" for f in feedback_history
        )
        return PhaseResult(
            phase_id=phase_id,
            success=False,
            artifact_id=None,
            attempts=budget,
            feedback=f"Exhausted retry budget ({budget} attempts).\n{final_feedback}",
        )
