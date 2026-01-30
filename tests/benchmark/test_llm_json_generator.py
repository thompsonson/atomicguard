"""Tests for LLMJsonGenerator."""

import json
from unittest.mock import MagicMock, patch

import pytest

from atomicguard.domain.models import (
    AmbientEnvironment,
    ArtifactStatus,
    Context,
)
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.persistence.memory import InMemoryArtifactDAG

from examples.advanced.g_plan_benchmark.generators.llm_json_generator import (
    LLMJsonGenerator,
    LLMJsonGeneratorConfig,
)


@pytest.fixture
def analysis_template() -> PromptTemplate:
    """Prompt template matching the g_analysis entry in prompts.json."""
    return PromptTemplate(
        role="You are a software engineering triage specialist.",
        constraints=(
            "Analyze the problem and return a JSON object:\n"
            '{\n  "problem_type": "bug_fix" | "feature",\n'
            '  "language": "python" | "unknown"\n}\n'
            "Return ONLY the JSON object."
        ),
        task="Analyze this problem and classify it.",
        feedback_wrapper="ANALYSIS REJECTED:\n{feedback}\n\nFix the analysis.",
    )


@pytest.fixture
def sample_context() -> Context:
    """Build a Context for testing."""
    return Context(
        ambient=AmbientEnvironment(
            repository=InMemoryArtifactDAG(), constraints=""
        ),
        specification="Fix the TypeError in the login handler.",
        current_artifact=None,
        feedback_history=(),
        dependency_artifacts=(),
    )


VALID_ANALYSIS_JSON = json.dumps(
    {
        "problem_type": "bug_fix",
        "language": "python",
        "severity": "high",
        "key_signals": ["TypeError"],
        "affected_area": "login handler",
        "rationale": "Type error in auth module.",
    }
)


class TestLLMJsonGeneratorConfig:
    """Test config defaults and inheritance."""

    def test_default_config(self):
        config = LLMJsonGeneratorConfig()
        assert config.model == "qwen2.5-coder:14b"
        assert config.backend == "ollama"
        assert config.temperature == 0.7

    def test_custom_config(self):
        config = LLMJsonGeneratorConfig(
            model="custom-model",
            backend="huggingface",
            temperature=0.3,
        )
        assert config.model == "custom-model"
        assert config.backend == "huggingface"
        assert config.temperature == 0.3


class TestLLMJsonGeneratorInit:
    """Test initialization."""

    @patch(
        "examples.advanced.g_plan_benchmark.generators.llm_plan_generator.OpenAI",
        create=True,
    )
    def test_init_ollama(self, mock_openai_cls):
        """Ollama backend initialises OpenAI client."""
        # Patch the import inside the generator
        with patch(
            "examples.advanced.g_plan_benchmark.generators.llm_plan_generator.LLMPlanGenerator._init_ollama"
        ) as mock_init:
            gen = LLMJsonGenerator(model="test-model", backend="ollama")
            mock_init.assert_called_once()

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            LLMJsonGenerator(backend="invalid")


class TestLLMJsonGeneratorGenerate:
    """Test generate() with mocked LLM calls."""

    def _make_generator(self) -> LLMJsonGenerator:
        """Create a generator with mocked Ollama init."""
        with patch(
            "examples.advanced.g_plan_benchmark.generators.llm_plan_generator.LLMPlanGenerator._init_ollama"
        ):
            gen = LLMJsonGenerator(model="test-model", backend="ollama")
        return gen

    def test_generate_returns_artifact(self, analysis_template, sample_context):
        gen = self._make_generator()
        gen._call_llm = MagicMock(return_value=VALID_ANALYSIS_JSON)

        artifact = gen.generate(
            context=sample_context,
            template=analysis_template,
            action_pair_id="g_analysis",
            workflow_id="test-workflow",
        )

        assert artifact.status == ArtifactStatus.PENDING
        assert artifact.action_pair_id == "g_analysis"
        assert artifact.workflow_id == "test-workflow"
        assert json.loads(artifact.content)["problem_type"] == "bug_fix"

    def test_generate_extracts_json_from_markdown(
        self, analysis_template, sample_context
    ):
        gen = self._make_generator()
        wrapped = f"```json\n{VALID_ANALYSIS_JSON}\n```"
        gen._call_llm = MagicMock(return_value=wrapped)

        artifact = gen.generate(
            context=sample_context,
            template=analysis_template,
            action_pair_id="g_analysis",
        )

        data = json.loads(artifact.content)
        assert data["problem_type"] == "bug_fix"

    def test_generate_handles_llm_failure(
        self, analysis_template, sample_context
    ):
        gen = self._make_generator()
        gen._call_llm = MagicMock(side_effect=RuntimeError("LLM timeout"))

        artifact = gen.generate(
            context=sample_context,
            template=analysis_template,
            action_pair_id="g_analysis",
        )

        data = json.loads(artifact.content)
        assert "error" in data

    def test_generate_increments_version(
        self, analysis_template, sample_context
    ):
        gen = self._make_generator()
        gen._call_llm = MagicMock(return_value=VALID_ANALYSIS_JSON)

        a1 = gen.generate(
            context=sample_context,
            template=analysis_template,
            action_pair_id="g_analysis",
        )
        a2 = gen.generate(
            context=sample_context,
            template=analysis_template,
            action_pair_id="g_analysis",
        )

        assert a1.attempt_number == 1
        assert a2.attempt_number == 2

    def test_generate_unique_artifact_ids(
        self, analysis_template, sample_context
    ):
        gen = self._make_generator()
        gen._call_llm = MagicMock(return_value=VALID_ANALYSIS_JSON)

        a1 = gen.generate(
            context=sample_context,
            template=analysis_template,
            action_pair_id="g_analysis",
        )
        a2 = gen.generate(
            context=sample_context,
            template=analysis_template,
            action_pair_id="g_analysis",
        )

        assert a1.artifact_id != a2.artifact_id

    def test_context_snapshot_preserved(
        self, analysis_template, sample_context
    ):
        gen = self._make_generator()
        gen._call_llm = MagicMock(return_value=VALID_ANALYSIS_JSON)

        artifact = gen.generate(
            context=sample_context,
            template=analysis_template,
            action_pair_id="g_analysis",
            workflow_id="test-wf",
        )

        assert artifact.context.workflow_id == "test-wf"
        assert "TypeError" in artifact.context.specification

    def test_uses_template_for_prompt(
        self, analysis_template, sample_context
    ):
        gen = self._make_generator()
        gen._call_llm = MagicMock(return_value=VALID_ANALYSIS_JSON)

        gen.generate(
            context=sample_context,
            template=analysis_template,
            action_pair_id="g_analysis",
        )

        # Verify the system message includes role + constraints
        call_args = gen._call_llm.call_args[0][0]
        system_msg = call_args[0]
        assert system_msg["role"] == "system"
        assert "triage specialist" in system_msg["content"]
        assert "JSON object" in system_msg["content"]


class TestLLMJsonGeneratorIsGenerator:
    """Verify LLMJsonGenerator implements GeneratorInterface."""

    def test_is_generator_interface(self):
        from atomicguard.domain.interfaces import GeneratorInterface

        assert issubclass(LLMJsonGenerator, GeneratorInterface)
