"""Tests for domain prompts - PromptTemplate, StepDefinition, TaskDefinition."""

import pytest

from atomicguard.domain.models import AmbientEnvironment, Context
from atomicguard.domain.prompts import PromptTemplate, StepDefinition, TaskDefinition
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


class TestPromptTemplateInit:
    """Tests for PromptTemplate initialization."""

    def test_init_stores_fields(self) -> None:
        """PromptTemplate stores role, constraints, task."""
        template = PromptTemplate(
            role="Python developer",
            constraints="No external imports",
            task="Write a function",
        )

        assert template.role == "Python developer"
        assert template.constraints == "No external imports"
        assert template.task == "Write a function"

    def test_init_default_feedback_wrapper(self) -> None:
        """PromptTemplate has default feedback_wrapper."""
        template = PromptTemplate(
            role="dev",
            constraints="none",
            task="test",
        )

        assert "GUARD REJECTION" in template.feedback_wrapper
        assert "{feedback}" in template.feedback_wrapper

    def test_init_custom_feedback_wrapper(self) -> None:
        """PromptTemplate accepts custom feedback_wrapper."""
        template = PromptTemplate(
            role="dev",
            constraints="none",
            task="test",
            feedback_wrapper="Error: {feedback}",
        )

        assert template.feedback_wrapper == "Error: {feedback}"


class TestPromptTemplateRender:
    """Tests for PromptTemplate.render() method."""

    def test_render_includes_role(self) -> None:
        """render() includes role section."""
        template = PromptTemplate(
            role="Expert Python developer",
            constraints="Pure Python",
            task="Write code",
        )
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(repository=dag, constraints="")
        context = Context(
            ambient=ambient,
            specification="test",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        result = template.render(context)

        assert "# ROLE" in result
        assert "Expert Python developer" in result

    def test_render_includes_constraints(self) -> None:
        """render() includes constraints section."""
        template = PromptTemplate(
            role="dev",
            constraints="No imports allowed",
            task="Write code",
        )
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(repository=dag, constraints="")
        context = Context(
            ambient=ambient,
            specification="test",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        result = template.render(context)

        assert "# CONSTRAINTS" in result
        assert "No imports allowed" in result

    def test_render_includes_task(self) -> None:
        """render() includes task section."""
        template = PromptTemplate(
            role="dev",
            constraints="none",
            task="Implement a stack data structure",
        )
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(repository=dag, constraints="")
        context = Context(
            ambient=ambient,
            specification="test",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        result = template.render(context)

        assert "# TASK" in result
        assert "Implement a stack data structure" in result

    def test_render_includes_ambient_constraints(self) -> None:
        """render() includes ambient constraints when present."""
        template = PromptTemplate(
            role="dev",
            constraints="template constraints",
            task="test",
        )
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(
            repository=dag, constraints="Global constraint: Python 3.12+"
        )
        context = Context(
            ambient=ambient,
            specification="test",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        result = template.render(context)

        assert "# CONTEXT" in result
        assert "Global constraint: Python 3.12+" in result

    def test_render_excludes_context_when_no_ambient_constraints(self) -> None:
        """render() excludes context section when ambient constraints empty."""
        template = PromptTemplate(
            role="dev",
            constraints="template constraints",
            task="test",
        )
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(repository=dag, constraints="")
        context = Context(
            ambient=ambient,
            specification="test",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        result = template.render(context)

        assert "# CONTEXT" not in result

    def test_render_includes_feedback_history(self) -> None:
        """render() includes feedback history when present."""
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
            feedback_history=(
                ("bad code v1", "Syntax error on line 1"),
                ("bad code v2", "Missing return statement"),
            ),
            dependency_artifacts=(),
        )

        result = template.render(context)

        assert "# HISTORY" in result
        assert "Attempt 1" in result
        assert "Syntax error on line 1" in result
        assert "Attempt 2" in result
        assert "Missing return statement" in result

    def test_render_excludes_history_when_empty(self) -> None:
        """render() excludes history section when feedback_history empty."""
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
            feedback_history=(),
            dependency_artifacts=(),
        )

        result = template.render(context)

        assert "# HISTORY" not in result

    def test_render_uses_feedback_wrapper(self) -> None:
        """render() uses feedback_wrapper to format feedback."""
        template = PromptTemplate(
            role="dev",
            constraints="none",
            task="test",
            feedback_wrapper="[ERROR] {feedback} [/ERROR]",
        )
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(repository=dag, constraints="")
        context = Context(
            ambient=ambient,
            specification="test",
            current_artifact=None,
            feedback_history=(("code", "Bad stuff"),),
            dependency_artifacts=(),
        )

        result = template.render(context)

        assert "[ERROR] Bad stuff [/ERROR]" in result


class TestStepDefinition:
    """Tests for StepDefinition dataclass."""

    def test_step_definition_stores_fields(self) -> None:
        """StepDefinition stores all fields."""
        step = StepDefinition(
            step_id="g_test",
            prompt="Write tests for {function}",
            guard="syntax",
            requires=("g_spec",),
        )

        assert step.step_id == "g_test"
        assert step.prompt == "Write tests for {function}"
        assert step.guard == "syntax"
        assert step.requires == ("g_spec",)

    def test_step_definition_default_requires(self) -> None:
        """StepDefinition defaults requires to empty tuple."""
        step = StepDefinition(
            step_id="g_test",
            prompt="Write tests",
            guard="syntax",
        )

        assert step.requires == ()

    def test_step_definition_is_frozen(self) -> None:
        """StepDefinition is immutable."""
        step = StepDefinition(
            step_id="g_test",
            prompt="Write tests",
            guard="syntax",
        )

        with pytest.raises(AttributeError):
            step.step_id = "g_other"  # type: ignore[misc]


class TestTaskDefinition:
    """Tests for TaskDefinition dataclass."""

    def test_task_definition_stores_fields(self) -> None:
        """TaskDefinition stores all fields."""
        steps = (
            StepDefinition(step_id="g_test", prompt="Write tests", guard="syntax"),
            StepDefinition(
                step_id="g_impl",
                prompt="Implement",
                guard="dynamic_test",
                requires=("g_test",),
            ),
        )
        task = TaskDefinition(
            task_id="tdd_task",
            name="TDD Implementation",
            specification="Implement a stack using TDD",
            steps=steps,
        )

        assert task.task_id == "tdd_task"
        assert task.name == "TDD Implementation"
        assert task.specification == "Implement a stack using TDD"
        assert len(task.steps) == 2

    def test_task_definition_get_step_found(self) -> None:
        """get_step() returns step when found."""
        step1 = StepDefinition(step_id="g_test", prompt="Write tests", guard="syntax")
        step2 = StepDefinition(step_id="g_impl", prompt="Implement", guard="syntax")
        task = TaskDefinition(
            task_id="task",
            name="Task",
            specification="spec",
            steps=(step1, step2),
        )

        result = task.get_step("g_impl")

        assert result is not None
        assert result.step_id == "g_impl"

    def test_task_definition_get_step_not_found(self) -> None:
        """get_step() returns None when step not found."""
        step = StepDefinition(step_id="g_test", prompt="Write tests", guard="syntax")
        task = TaskDefinition(
            task_id="task",
            name="Task",
            specification="spec",
            steps=(step,),
        )

        result = task.get_step("g_nonexistent")

        assert result is None

    def test_task_definition_is_frozen(self) -> None:
        """TaskDefinition is immutable."""
        task = TaskDefinition(
            task_id="task",
            name="Task",
            specification="spec",
            steps=(),
        )

        with pytest.raises(AttributeError):
            task.task_id = "other"  # type: ignore[misc]
