"""
Prompt and task definitions for the Dual-State Framework.

This module provides:
- PromptTemplate: Structured prompt rendering
- StepDefinition: Single workflow step specification
- TaskDefinition: Complete task with multiple steps

These are domain structures (schemas) only. Actual task content should be
defined by the calling application (e.g., benchmarks), not hardcoded here.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atomicguard.domain.models import Context


# =============================================================================
# PROMPT TEMPLATE (moved from models.py)
# =============================================================================


@dataclass(frozen=True)
class PromptTemplate:
    """Structured prompt template for generator."""

    role: str
    constraints: str
    task: str
    feedback_wrapper: str = (
        "GUARD REJECTION:\n{feedback}\nInstruction: Address the rejection above."
    )

    def render(self, context: "Context") -> str:
        """Render prompt with context including dependency artifacts."""
        parts = [
            f"# ROLE\n{self.role}",
            f"# CONSTRAINTS\n{self.constraints}",
        ]

        if context.ambient.constraints:
            parts.append(f"# CONTEXT\n{context.ambient.constraints}")

        if context.dependency_artifacts and context.ambient.repository:
            dep_parts = []
            for key, artifact_id in context.dependency_artifacts:
                try:
                    artifact = context.ambient.repository.get_artifact(artifact_id)
                    dep_parts.append(f"## {key}\n{artifact.content}")
                except KeyError:
                    logger.debug(
                        "Dependency artifact %s not found, skipping", artifact_id
                    )
            if dep_parts:
                parts.append("# DEPENDENCIES\n" + "\n\n".join(dep_parts))

        if context.feedback_history:
            parts.append("# HISTORY (Context Refinement)")
            for i, (_artifact_content, feedback) in enumerate(context.feedback_history):
                wrapped = self.feedback_wrapper.format(feedback=feedback)
                parts.append(f"--- Attempt {i + 1} ---\n{wrapped}")

        parts.append(f"# TASK\n{self.task}")
        return "\n\n".join(parts)


# =============================================================================
# TASK DEFINITIONS (DS-PDDL semantic layer)
# =============================================================================


@dataclass(frozen=True)
class StepDefinition:
    """Single workflow step specification."""

    step_id: str  # e.g., "g_test", "g_impl"
    prompt: str  # Prompt template with {placeholders}
    guard: str  # Guard type: "syntax", "dynamic_test", "human", etc.
    requires: tuple[str, ...] = ()  # Step IDs this depends on


@dataclass(frozen=True)
class TaskDefinition:
    """Complete task definition with multiple workflow steps."""

    task_id: str  # e.g., "tdd_stack"
    name: str  # Human-readable name
    specification: str  # High-level task description (Ψ)
    steps: tuple[StepDefinition, ...]  # Ordered workflow steps

    def get_step(self, step_id: str) -> StepDefinition | None:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
