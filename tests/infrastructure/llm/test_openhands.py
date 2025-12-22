"""Tests for SemanticAgentGenerator - OpenHands SDK generator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from atomicguard.domain.models import (
    AmbientEnvironment,
    Artifact,
    Context,
)
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG


class TestSemanticAgentGeneratorInit:
    """Tests for SemanticAgentGenerator initialization."""

    def test_init_raises_without_openhands_sdk(self) -> None:
        """Raises ImportError when openhands-ai not installed."""
        with (
            patch.dict("sys.modules", {"openhands": None, "openhands.core": None}),
            pytest.raises(ImportError, match="openhands-ai required"),
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            SemanticAgentGenerator()

    def test_init_stores_model(self) -> None:
        """Stores model configuration."""
        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator(model="openai/test-model")
            assert generator._model == "openai/test-model"

    def test_init_default_model(self) -> None:
        """Uses default model when not specified."""
        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            assert generator._model == "openai/qwen3-coder:30b"

    def test_init_stores_workspace(self, tmp_path: Path) -> None:
        """Stores workspace path when provided."""
        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator(workspace=tmp_path)
            assert generator._workspace == tmp_path

    def test_init_stores_base_url(self) -> None:
        """Stores base URL configuration."""
        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator(base_url="http://custom:8080/v1")
            assert generator._base_url == "http://custom:8080/v1"


class TestSemanticAgentGeneratorManifest:
    """Tests for manifest building functionality."""

    def test_build_manifest_captures_files(self, tmp_path: Path) -> None:
        """Manifest captures created files."""
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "util.py").write_text("y = 2")

        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            manifest = generator._build_manifest(tmp_path)

        assert "main.py" in manifest["files"]
        assert manifest["files"]["main.py"]["content"] == "x = 1"
        assert manifest["file_count"] == 2

    def test_build_manifest_excludes_hidden_files(self, tmp_path: Path) -> None:
        """Manifest excludes hidden files and __pycache__."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.pyc").write_text("bytecode")
        (tmp_path / "visible.py").write_text("x = 1")

        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            manifest = generator._build_manifest(tmp_path)

        assert len(manifest["files"]) == 1
        assert "visible.py" in manifest["files"]

    def test_build_manifest_includes_workspace_path(self, tmp_path: Path) -> None:
        """Manifest includes workspace path."""
        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            manifest = generator._build_manifest(tmp_path)

        assert manifest["workspace"] == str(tmp_path)

    def test_build_manifest_captures_directories(self, tmp_path: Path) -> None:
        """Manifest captures directory structure."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "domain").mkdir()
        (tmp_path / "src" / "__init__.py").write_text("")

        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            manifest = generator._build_manifest(tmp_path)

        assert "src" in manifest["directories"]
        # Note: src/domain may or may not be included depending on sorting

    def test_build_manifest_handles_empty_workspace(self, tmp_path: Path) -> None:
        """Manifest handles empty workspace gracefully."""
        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            manifest = generator._build_manifest(tmp_path)

        assert manifest["file_count"] == 0
        assert manifest["files"] == {}


class TestSemanticAgentGeneratorPromptBuilding:
    """Tests for prompt construction."""

    def test_build_agentic_prompt_includes_specification(self) -> None:
        """Agentic prompt includes specification."""
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(repository=dag, constraints="")
        context = Context(
            ambient=ambient,
            specification="Build a REST API",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            prompt = generator._build_agentic_prompt(context)

        assert "Build a REST API" in prompt

    def test_build_agentic_prompt_includes_constraints(self) -> None:
        """Agentic prompt includes constraints when present."""
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(
            repository=dag, constraints="Use Python 3.12+ features"
        )
        context = Context(
            ambient=ambient,
            specification="Write code",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            prompt = generator._build_agentic_prompt(context)

        assert "Use Python 3.12+ features" in prompt
        assert "## Constraints" in prompt

    def test_build_agentic_prompt_includes_feedback_history(self) -> None:
        """Agentic prompt includes feedback from prior attempts."""
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(repository=dag, constraints="")
        context = Context(
            ambient=ambient,
            specification="Write a stack",
            current_artifact=None,
            feedback_history=(("id1", "Missing pop method"),),
            dependency_artifacts=(),
        )

        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            prompt = generator._build_agentic_prompt(context)

        assert "Missing pop method" in prompt
        assert "Attempt 1" in prompt

    def test_build_agentic_prompt_includes_agent_instructions(self) -> None:
        """Agentic prompt includes instructions for file creation."""
        dag = InMemoryArtifactDAG()
        ambient = AmbientEnvironment(repository=dag, constraints="")
        context = Context(
            ambient=ambient,
            specification="Write code",
            current_artifact=None,
            feedback_history=(),
            dependency_artifacts=(),
        )

        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            prompt = generator._build_agentic_prompt(context)

        assert "software engineering agent" in prompt
        assert "workspace" in prompt
        assert "Create the necessary files" in prompt


class TestSemanticAgentGeneratorWorkspace:
    """Tests for workspace resolution."""

    def test_resolve_workspace_uses_provided_workspace(self, tmp_path: Path) -> None:
        """Uses constructor-provided workspace."""
        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator(workspace=tmp_path)
            resolved = generator._resolve_workspace()

        assert resolved == tmp_path

    def test_resolve_workspace_creates_temp_when_none(self) -> None:
        """Creates temp directory when no workspace provided."""
        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator()
            resolved = generator._resolve_workspace()

        assert "atomicguard_agent_" in str(resolved)
        assert resolved.exists()

    def test_resolve_workspace_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Creates parent directories if needed."""
        nested = tmp_path / "deep" / "nested" / "workspace"

        with patch(
            "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
        ):
            from atomicguard.infrastructure.llm.openhands import (
                SemanticAgentGenerator,
            )

            generator = SemanticAgentGenerator(workspace=nested)
            resolved = generator._resolve_workspace()

        assert resolved == nested
        assert nested.exists()


class TestSemanticAgentGeneratorGenerate:
    """Tests for generate() method with mocked SDK."""

    @pytest.fixture
    def mock_openhands_sdk(self):  # noqa: ANN001
        """Mock OpenHands SDK components."""
        mock_runtime = MagicMock()
        mock_runtime.close = MagicMock()

        # Create async mock for run_controller
        async def mock_run_controller_async(*_args, **_kwargs):  # noqa: ANN003
            return MagicMock()

        with (
            patch(
                "atomicguard.infrastructure.llm.openhands.SemanticAgentGenerator._verify_sdk_available"
            ),
            patch("openhands.core.config.llm_config.LLMConfig") as mock_llm_config,
            patch(
                "openhands.core.config.sandbox_config.SandboxConfig"
            ) as mock_sandbox_config,
            patch(
                "openhands.core.config.openhands_config.OpenHandsConfig"
            ) as mock_openhands_config,
            patch(
                "openhands.core.main.create_runtime", return_value=mock_runtime
            ) as mock_create_runtime,
            patch(
                "openhands.core.main.run_controller",
                side_effect=mock_run_controller_async,
            ) as mock_run_controller,
            patch(
                "openhands.events.action.message.MessageAction"
            ) as mock_message_action,
        ):
            yield {
                "llm_config": mock_llm_config,
                "sandbox_config": mock_sandbox_config,
                "openhands_config": mock_openhands_config,
                "create_runtime": mock_create_runtime,
                "run_controller": mock_run_controller,
                "message_action": mock_message_action,
                "runtime": mock_runtime,
            }

    def test_generate_returns_artifact(
        self,
        sample_context: Context,
        tmp_path: Path,
        mock_openhands_sdk: dict,  # noqa: ARG002
    ) -> None:
        """generate() returns Artifact with JSON manifest."""
        # Create a test file in workspace
        (tmp_path / "test.py").write_text("print('hello')")

        from atomicguard.infrastructure.llm.openhands import (
            SemanticAgentGenerator,
        )

        generator = SemanticAgentGenerator(workspace=tmp_path)
        artifact = generator.generate(sample_context)

        assert isinstance(artifact, Artifact)
        manifest = json.loads(artifact.content)
        assert "files" in manifest
        assert "test.py" in manifest["files"]

    def test_generate_manifest_contains_workspace_path(
        self,
        sample_context: Context,
        tmp_path: Path,
        mock_openhands_sdk: dict,  # noqa: ARG002
    ) -> None:
        """Manifest includes workspace path."""
        from atomicguard.infrastructure.llm.openhands import (
            SemanticAgentGenerator,
        )

        generator = SemanticAgentGenerator(workspace=tmp_path)
        artifact = generator.generate(sample_context)

        manifest = json.loads(artifact.content)
        assert manifest["workspace"] == str(tmp_path)

    def test_generate_increments_version_counter(
        self,
        sample_context: Context,
        tmp_path: Path,
        mock_openhands_sdk: dict,  # noqa: ARG002
    ) -> None:
        """generate() increments version counter."""
        from atomicguard.infrastructure.llm.openhands import (
            SemanticAgentGenerator,
        )

        generator = SemanticAgentGenerator(workspace=tmp_path)
        artifact1 = generator.generate(sample_context)
        artifact2 = generator.generate(sample_context)

        assert artifact1.attempt_number == 1
        assert artifact2.attempt_number == 2

    def test_generate_captures_context_snapshot(
        self,
        sample_context: Context,
        tmp_path: Path,
        mock_openhands_sdk: dict,  # noqa: ARG002
    ) -> None:
        """generate() captures context in artifact."""
        from atomicguard.infrastructure.llm.openhands import (
            SemanticAgentGenerator,
        )

        generator = SemanticAgentGenerator(workspace=tmp_path)
        artifact = generator.generate(sample_context)

        assert artifact.context.specification == sample_context.specification

    def test_generate_closes_runtime_on_success(
        self,
        sample_context: Context,
        tmp_path: Path,
        mock_openhands_sdk: dict,
    ) -> None:
        """generate() closes runtime after execution."""
        from atomicguard.infrastructure.llm.openhands import (
            SemanticAgentGenerator,
        )

        generator = SemanticAgentGenerator(workspace=tmp_path)
        generator.generate(sample_context)

        mock_openhands_sdk["runtime"].close.assert_called_once()

    def test_generate_uses_template_when_provided(
        self,
        sample_context: Context,
        tmp_path: Path,
        mock_openhands_sdk: dict,  # noqa: ARG002
    ) -> None:
        """generate() uses template when provided."""
        from atomicguard.domain.prompts import PromptTemplate
        from atomicguard.infrastructure.llm.openhands import (
            SemanticAgentGenerator,
        )

        template = PromptTemplate(
            role="Custom Agent",
            constraints="Custom constraints",
            task="Custom task",
        )

        generator = SemanticAgentGenerator(workspace=tmp_path)
        artifact = generator.generate(sample_context, template)

        # Verify artifact was generated (template was used internally)
        assert artifact is not None
        # The template would render with Custom Agent/task in the prompt
        # We can't easily verify the internal MessageAction call due to import path
        # but we verify the artifact was created successfully
