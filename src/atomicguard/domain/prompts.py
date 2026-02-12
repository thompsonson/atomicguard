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
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atomicguard.domain.models import Context


# =============================================================================
# PROMPT TEMPLATE (moved from models.py)
# =============================================================================


# Pattern matches {ap_*} placeholders in prompt templates
_PLACEHOLDER_PATTERN = re.compile(r"\{(ap_[a-z_]+)\}")


@dataclass(frozen=True)
class PromptTemplate:
    """Structured prompt template for generator.

    Prompt templates are the single source of truth for what context
    each generator needs. Templates can contain {ap_*} placeholders
    that are automatically substituted with dependency artifact content.

    Supported placeholders:
    - {specification}: Context specification (problem statement)
    - {constraints}: Ambient constraints
    - {feedback}: Latest rejection feedback
    - {ap_*}: Dependency artifact content (e.g., {ap_analysis}, {ap_localise_issue})

    Example:
        {
            "task": "Generate a patch.\\n\\n## Fix Approach\\n{ap_fix_approach}\\n\\n## Test\\n{ap_gen_test}"
        }
    """

    role: str
    constraints: str
    task: str
    feedback_wrapper: str = (
        "GUARD REJECTION:\n{feedback}\nInstruction: Address the rejection above."
    )

    def render(self, context: "Context") -> str:
        """Render prompt with context including dependency artifacts.

        Performs placeholder substitution on all template sections:
        1. Builds a map of {ap_*} -> artifact.content from dependency_artifacts
        2. Substitutes placeholders in role, constraints, task sections
        3. Assembles the final prompt with all sections

        Missing placeholders are replaced with empty string and logged.

        Args:
            context: Execution context with specification and dependencies

        Returns:
            Rendered prompt string ready for the LLM
        """
        # Build placeholder map from dependency artifacts
        dep_map = self._build_dependency_map(context)

        # Substitute placeholders in template sections
        role = self._substitute_placeholders(self.role, dep_map, context)
        constraints = self._substitute_placeholders(self.constraints, dep_map, context)
        task = self._substitute_placeholders(self.task, dep_map, context)

        parts = [
            f"# ROLE\n{role}",
            f"# CONSTRAINTS\n{constraints}",
        ]

        # Add ambient constraints (already processed context)
        if context.ambient.constraints:
            parts.append(f"# CONTEXT\n{context.ambient.constraints}")

        # Add dependency artifacts as structured sections
        # (for backwards compatibility with generators not using placeholders)
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

        # Add feedback history
        if context.feedback_history:
            parts.append("# HISTORY (Context Refinement)")
            for i, (_artifact_content, feedback) in enumerate(context.feedback_history):
                wrapped = self.feedback_wrapper.format(feedback=feedback)
                parts.append(f"--- Attempt {i + 1} ---\n{wrapped}")

        parts.append(f"# TASK\n{task}")
        return "\n\n".join(parts)

    def _build_dependency_map(self, context: "Context") -> dict[str, str]:
        """Build a map of action_pair_id -> artifact.content for placeholders.

        Args:
            context: Execution context with dependency artifacts

        Returns:
            Dict mapping action_pair_id (e.g., "ap_analysis") to artifact content
        """
        dep_map: dict[str, str] = {}

        if not context.dependency_artifacts or not context.ambient.repository:
            return dep_map

        for key, artifact_id in context.dependency_artifacts:
            try:
                artifact = context.ambient.repository.get_artifact(artifact_id)
                dep_map[key] = artifact.content
            except KeyError:
                logger.debug(
                    "Dependency artifact %s not found for placeholder {%s}",
                    artifact_id,
                    key,
                )

        return dep_map

    def _substitute_placeholders(
        self,
        text: str,
        dep_map: dict[str, str],
        context: "Context",
    ) -> str:
        """Substitute {ap_*} placeholders in text with artifact content.

        Standard placeholders:
        - {specification}: Context specification
        - {constraints}: Ambient constraints
        - {feedback}: Latest rejection feedback (if any)
        - {ap_*}: Dependency artifact content

        Args:
            text: Template text with placeholders
            dep_map: Map of action_pair_id -> artifact content
            context: Execution context for standard placeholders

        Returns:
            Text with placeholders substituted
        """
        # First substitute standard placeholders
        result = text.replace("{specification}", context.specification)
        result = result.replace("{constraints}", context.ambient.constraints or "")

        # Substitute feedback placeholder
        if context.feedback_history:
            latest_feedback = context.feedback_history[-1][1]
            result = result.replace("{feedback}", latest_feedback)
        else:
            result = result.replace("{feedback}", "")

        # Find and substitute all {ap_*} placeholders
        def replace_placeholder(match: re.Match[str]) -> str:
            placeholder = match.group(1)  # e.g., "ap_analysis"
            if placeholder in dep_map:
                return dep_map[placeholder]
            else:
                logger.debug(
                    "Placeholder {%s} not found in dependency artifacts", placeholder
                )
                return ""

        result = _PLACEHOLDER_PATTERN.sub(replace_placeholder, result)
        return result


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
