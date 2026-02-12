"""
Hexagonal Architecture Boundary Tests.

Tests for domain model contracts and interface requirements.
"""

import pytest

from atomicguard.domain.models import AmbientEnvironment, Context
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


class TestPromptTemplateContract:
    """Tests for PromptTemplate field requirements."""

    def test_feedback_wrapper_defaults_to_none(self) -> None:
        """feedback_wrapper defaults to None (optional at construction)."""
        template = PromptTemplate(
            role="dev",
            constraints="none",
            task="test",
        )

        assert template.feedback_wrapper is None

    def test_render_errors_when_feedback_without_wrapper(self) -> None:
        """render() raises ValueError if feedback_history exists but no wrapper."""
        template = PromptTemplate(
            role="dev",
            constraints="none",
            task="test",
        )
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(repository=dag, constraints="")
        context = Context(
            ambient=ambient,
            specification="test",
            current_artifact=None,
            feedback_history=(("bad code", "Syntax error"),),
            dependency_artifacts=(),
        )

        with pytest.raises(ValueError, match="feedback_wrapper must be defined"):
            template.render(context)

    def test_escalation_feedback_wrapper_defaults_to_none(self) -> None:
        """escalation_feedback_wrapper is optional with None default."""
        template = PromptTemplate(
            role="dev",
            constraints="none",
            task="test",
        )

        assert template.escalation_feedback_wrapper is None

    def test_feedback_wrapper_can_be_set(self) -> None:
        """PromptTemplate accepts feedback_wrapper when provided."""
        template = PromptTemplate(
            role="dev",
            constraints="none",
            task="test",
            feedback_wrapper="Error: {feedback}",
        )

        assert template.feedback_wrapper == "Error: {feedback}"

    def test_escalation_feedback_wrapper_can_be_set(self) -> None:
        """escalation_feedback_wrapper can be set when needed."""
        template = PromptTemplate(
            role="dev",
            constraints="none",
            task="test",
            feedback_wrapper="{feedback}",
            escalation_feedback_wrapper="Prior cycle: {feedback}",
        )

        assert template.escalation_feedback_wrapper == "Prior cycle: {feedback}"
