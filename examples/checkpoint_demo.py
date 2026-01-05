#!/usr/bin/env python3
"""
Demo script showing checkpoint/resume functionality.

Run with:
    uv run python examples/checkpoint_demo.py

This demonstrates:
1. A workflow that fails (hits Rmax) and creates a checkpoint
2. Listing and viewing the checkpoint
3. Resuming with a human-provided artifact
"""

import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from atomicguard.application.action_pair import ActionPair
from atomicguard.application.workflow import ResumableWorkflow
from atomicguard.domain.interfaces import GeneratorInterface, GuardInterface
from atomicguard.domain.models import (
    AmendmentType,
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    GuardResult,
    HumanAmendment,
    WorkflowStatus,
)
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.persistence.checkpoint import FilesystemCheckpointDAG
from atomicguard.infrastructure.persistence.filesystem import FilesystemArtifactDAG

# =============================================================================
# Mock Components
# =============================================================================


class AlwaysFailGenerator(GeneratorInterface):
    """Generator that always produces failing code."""

    def __init__(self) -> None:
        self._call_count = 0

    def generate(
        self,
        context: Context,
        _template: PromptTemplate | None = None,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
    ) -> Artifact:
        self._call_count += 1
        # Always generates code with a "bug"
        content = f"def add(a, b):\n    return a - b  # BUG: attempt {self._call_count}"

        return Artifact(
            artifact_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            content=content,
            previous_attempt_id=None,
            parent_action_pair_id=None,
            action_pair_id=action_pair_id,
            created_at=datetime.now(UTC).isoformat(),
            attempt_number=self._call_count,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=ContextSnapshot(
                workflow_id=workflow_id,
                specification=context.specification,
                constraints=context.ambient.constraints if context.ambient else "",
                feedback_history=context.feedback_history,
                dependency_artifacts=(),
            ),
        )


class CorrectAddGuard(GuardInterface):
    """Guard that checks if add function returns correct results."""

    @property
    def guard_id(self) -> str:
        return "g_correct_add"

    def validate(self, artifact: Artifact, **_deps: Artifact) -> GuardResult:
        # Simple check: does the code contain "return a + b"?
        if "return a + b" in artifact.content:
            return GuardResult(passed=True, feedback="Correct implementation!")
        return GuardResult(
            passed=False,
            feedback=f"FAILED: Expected 'return a + b' but got:\n{artifact.content}",
        )


# =============================================================================
# Demo Runner
# =============================================================================


def run_demo() -> None:
    """Run the checkpoint/resume demo."""
    print("=" * 60)
    print("CHECKPOINT/RESUME DEMO")
    print("=" * 60)

    # Create temporary directories for artifacts and checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir) / "artifacts"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        # Initialize DAGs
        artifact_dag = FilesystemArtifactDAG(str(artifact_dir))
        checkpoint_dag = FilesystemCheckpointDAG(str(checkpoint_dir))

        # Initialize components
        generator = AlwaysFailGenerator()
        guard = CorrectAddGuard()

        # Create the ActionPair
        action_pair = ActionPair(generator=generator, guard=guard)

        # Create the ResumableWorkflow
        workflow = ResumableWorkflow(
            artifact_dag=artifact_dag,
            checkpoint_dag=checkpoint_dag,
            rmax=2,  # Only 2 retries to hit failure quickly
            constraints="Must use 'return a + b'",
            auto_checkpoint=True,
        )

        # Add a single step
        workflow.add_step(
            guard_id="g_correct_add",
            action_pair=action_pair,
            requires=(),
        )

        # =================================================================
        # PHASE 1: Execute workflow (will fail and create checkpoint)
        # =================================================================
        print("\n" + "-" * 60)
        print("PHASE 1: Initial Execution (will fail after 2 attempts)")
        print("-" * 60)

        result = workflow.execute("Write a function that adds two numbers")

        print(f"\nWorkflow Status: {result.status.value}")

        if result.status == WorkflowStatus.CHECKPOINT:
            checkpoint = result.checkpoint
            assert checkpoint is not None  # for type checker

            print("\n[OK] Checkpoint created!")
            print(f"  Checkpoint ID: {checkpoint.checkpoint_id}")
            print(f"  Failed Step: {checkpoint.failed_step}")
            print(f"  Failure Type: {checkpoint.failure_type.value}")
            feedback_preview = checkpoint.failure_feedback[:80]
            print(f"  Feedback: {feedback_preview}...")

            # =================================================================
            # PHASE 2: Resume with human-provided artifact
            # =================================================================
            print("\n" + "-" * 60)
            print("PHASE 2: Resume with Human Amendment")
            print("-" * 60)

            # Create a human amendment with the correct code
            human_artifact_content = (
                "def add(a, b):\n    return a + b  # Fixed by human"
            )

            amendment = HumanAmendment(
                amendment_id="amd-demo-001",
                checkpoint_id=checkpoint.checkpoint_id,
                amendment_type=AmendmentType.ARTIFACT,
                created_at=datetime.now(UTC).isoformat(),
                created_by="demo-user",
                content=human_artifact_content,
                context="Human fixed the subtraction bug",
                parent_artifact_id=checkpoint.failed_artifact_id,
                additional_rmax=0,
            )

            print("\nHuman provides corrected artifact:")
            print("```python")
            print(human_artifact_content)
            print("```")

            # Resume the workflow
            resume_result = workflow.resume(
                checkpoint_id=checkpoint.checkpoint_id,
                amendment=amendment,
            )

            print(f"\nResume Result: {resume_result.status.value}")

            if resume_result.status == WorkflowStatus.SUCCESS:
                print("\n[OK] Workflow completed successfully!")
                print(f"  Artifacts: {list(resume_result.artifacts.keys())}")

                # Show the final artifact
                final_artifact = resume_result.artifacts.get("g_correct_add")
                if final_artifact:
                    print(f"  Source: {final_artifact.source.value}")
                    print(f"  Status: {final_artifact.status.value}")
            else:
                print(f"\n[FAILED] Resume failed: {resume_result.status.value}")

        else:
            print(f"\nUnexpected status: {result.status.value}")

        # =================================================================
        # PHASE 3: List all checkpoints
        # =================================================================
        print("\n" + "-" * 60)
        print("PHASE 3: List All Checkpoints")
        print("-" * 60)

        checkpoints = checkpoint_dag.list_checkpoints()
        print(f"\nFound {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints:
            print(f"  - {cp.checkpoint_id[:16]}...")
            print(f"    Workflow: {cp.workflow_id[:16]}...")
            print(f"    Failed at: {cp.failed_step}")
            print(f"    Type: {cp.failure_type.value}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
