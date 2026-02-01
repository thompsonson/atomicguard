"""Tests for HuggingFaceGenerator."""

from unittest.mock import MagicMock, patch

import pytest

from atomicguard.domain.models import Artifact, ArtifactStatus
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.infrastructure.llm.huggingface import (
    HuggingFaceGenerator,
    HuggingFaceGeneratorConfig,
)

# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestHuggingFaceGeneratorConfig:
    """Tests for HuggingFaceGeneratorConfig defaults."""

    def test_default_model(self) -> None:
        config = HuggingFaceGeneratorConfig()
        assert config.model == "Qwen/Qwen2.5-Coder-32B-Instruct"

    def test_default_timeout(self) -> None:
        config = HuggingFaceGeneratorConfig()
        assert config.timeout == 120.0

    def test_default_temperature(self) -> None:
        config = HuggingFaceGeneratorConfig()
        assert config.temperature == 0.7

    def test_default_max_tokens(self) -> None:
        config = HuggingFaceGeneratorConfig()
        assert config.max_tokens == 4096

    def test_default_api_key_is_none(self) -> None:
        config = HuggingFaceGeneratorConfig()
        assert config.api_key is None

    def test_default_provider_is_none(self) -> None:
        config = HuggingFaceGeneratorConfig()
        assert config.provider is None

    def test_custom_values(self) -> None:
        config = HuggingFaceGeneratorConfig(
            model="custom/model",
            api_key="hf_test",
            provider="together",
            timeout=60.0,
            temperature=0.5,
            max_tokens=2048,
        )
        assert config.model == "custom/model"
        assert config.api_key == "hf_test"
        assert config.provider == "together"
        assert config.timeout == 60.0
        assert config.temperature == 0.5
        assert config.max_tokens == 2048


# =============================================================================
# INIT TESTS
# =============================================================================


class TestHuggingFaceGeneratorInit:
    """Tests for HuggingFaceGenerator initialization."""

    def test_raises_without_api_key_or_env(self) -> None:
        """Init raises ValueError when no API key available."""
        pytest.importorskip("huggingface_hub")
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="HF_TOKEN"),
        ):
            HuggingFaceGenerator(HuggingFaceGeneratorConfig(api_key=None))

    def test_uses_env_token(self) -> None:
        """Init reads HF_TOKEN from environment when no explicit key given."""
        hf_hub = pytest.importorskip("huggingface_hub")
        with (
            patch.dict("os.environ", {"HF_TOKEN": "hf_env_token"}),
            patch.object(
                hf_hub, "InferenceClient", return_value=MagicMock()
            ) as mock_client_cls,
        ):
            HuggingFaceGenerator(HuggingFaceGeneratorConfig())
            mock_client_cls.assert_called_once()
            assert mock_client_cls.call_args.kwargs["api_key"] == "hf_env_token"

    def test_uses_explicit_api_key_over_env(self) -> None:
        """Init prefers explicit api_key over HF_TOKEN env var."""
        hf_hub = pytest.importorskip("huggingface_hub")
        with (
            patch.dict("os.environ", {"HF_TOKEN": "hf_env_token"}),
            patch.object(
                hf_hub, "InferenceClient", return_value=MagicMock()
            ) as mock_client_cls,
        ):
            HuggingFaceGenerator(HuggingFaceGeneratorConfig(api_key="hf_explicit"))
            mock_client_cls.assert_called_once()
            assert mock_client_cls.call_args.kwargs["api_key"] == "hf_explicit"

    def test_legacy_kwargs_init(self) -> None:
        """Init works with legacy kwargs."""
        pytest.importorskip("huggingface_hub")
        gen = HuggingFaceGenerator(api_key="hf_test", model="custom/model")
        assert gen._model == "custom/model"

    def test_raises_without_huggingface_hub(self) -> None:
        """Init raises ImportError when huggingface_hub not installed."""
        with (
            patch.dict("sys.modules", {"huggingface_hub": None}),
            pytest.raises(ImportError, match="huggingface_hub"),
        ):
            HuggingFaceGenerator(HuggingFaceGeneratorConfig(api_key="hf_test"))


# =============================================================================
# CODE EXTRACTION TESTS
# =============================================================================


class TestExtractCode:
    """Tests for _extract_code() method."""

    @pytest.fixture
    def generator(self) -> HuggingFaceGenerator:
        """Create generator for testing extraction (no API call needed)."""
        pytest.importorskip("huggingface_hub")
        return HuggingFaceGenerator(HuggingFaceGeneratorConfig(api_key="hf_test"))

    def test_empty_string_returns_empty(self, generator: HuggingFaceGenerator) -> None:
        result = generator._extract_code("")
        assert result == ""

    def test_whitespace_only_returns_empty(
        self, generator: HuggingFaceGenerator
    ) -> None:
        result = generator._extract_code("   \n\t\n   ")
        assert result == ""

    def test_python_code_block_extracts_code(
        self, generator: HuggingFaceGenerator
    ) -> None:
        content = """Here is the code:
```python
def add(a, b):
    return a + b
```
That should work."""
        result = generator._extract_code(content)
        assert result == "def add(a, b):\n    return a + b"

    def test_generic_code_block_extracts_code(
        self, generator: HuggingFaceGenerator
    ) -> None:
        content = """Here is the code:
```
x = 1
y = 2
```
Done."""
        result = generator._extract_code(content)
        assert result == "x = 1\ny = 2"

    def test_prefers_python_block_over_generic(
        self, generator: HuggingFaceGenerator
    ) -> None:
        content = """
```
generic code
```
```python
python code
```
"""
        result = generator._extract_code(content)
        assert result == "python code"

    def test_code_starting_with_def(self, generator: HuggingFaceGenerator) -> None:
        content = """Some explanation
def hello():
    print("hi")

More text"""
        result = generator._extract_code(content)
        assert result.startswith("def hello():")

    def test_code_starting_with_import(self, generator: HuggingFaceGenerator) -> None:
        content = """Here's what you need:
import os
import sys

def main():
    pass"""
        result = generator._extract_code(content)
        assert result.startswith("import os")

    def test_code_starting_with_class(self, generator: HuggingFaceGenerator) -> None:
        content = """The solution is:
class MyClass:
    def __init__(self):
        pass"""
        result = generator._extract_code(content)
        assert result.startswith("class MyClass:")

    def test_plain_text_returns_empty(self, generator: HuggingFaceGenerator) -> None:
        content = "I cannot generate that code because it violates safety guidelines."
        result = generator._extract_code(content)
        assert result == ""

    def test_multiple_python_blocks_extracts_first(
        self, generator: HuggingFaceGenerator
    ) -> None:
        content = """First:
```python
first = 1
```
Second:
```python
second = 2
```"""
        result = generator._extract_code(content)
        assert result == "first = 1"


# =============================================================================
# GENERATE TESTS (with mocked API)
# =============================================================================


class TestHuggingFaceGeneratorGenerate:
    """Tests for generate() with mocked HuggingFace API."""

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Create a mock chat_completion response."""
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = "```python\ndef add(a, b):\n    return a + b\n```"
        response.choices = [choice]
        return response

    @pytest.fixture
    def generator(self, mock_response: MagicMock) -> HuggingFaceGenerator:
        """Create generator with mocked client."""
        pytest.importorskip("huggingface_hub")
        gen = HuggingFaceGenerator(HuggingFaceGeneratorConfig(api_key="hf_test"))
        gen._client = MagicMock()
        gen._client.chat_completion.return_value = mock_response
        return gen

    @pytest.fixture
    def template(self) -> PromptTemplate:
        """Create a minimal PromptTemplate for testing."""
        return PromptTemplate(role="test", constraints="", task="test")

    def test_generate_returns_artifact(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        artifact = generator.generate(sample_context, template)
        assert isinstance(artifact, Artifact)

    def test_generate_extracts_code_from_response(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        artifact = generator.generate(sample_context, template)
        assert artifact.content == "def add(a, b):\n    return a + b"

    def test_generate_sets_pending_status(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        artifact = generator.generate(sample_context, template)
        assert artifact.status == ArtifactStatus.PENDING

    def test_generate_increments_version_counter(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        generator.generate(sample_context, template)
        assert generator._version_counter == 1
        generator.generate(sample_context, template)
        assert generator._version_counter == 2

    def test_generate_sets_attempt_number(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        a1 = generator.generate(sample_context, template)
        a2 = generator.generate(sample_context, template)
        assert a1.attempt_number == 1
        assert a2.attempt_number == 2

    def test_generate_assigns_unique_artifact_ids(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        a1 = generator.generate(sample_context, template)
        a2 = generator.generate(sample_context, template)
        assert a1.artifact_id != a2.artifact_id

    def test_generate_sets_workflow_id(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        artifact = generator.generate(sample_context, template, workflow_id="wf-123")
        assert artifact.workflow_id == "wf-123"

    def test_generate_sets_action_pair_id(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        artifact = generator.generate(
            sample_context, template, action_pair_id="ap-test"
        )
        assert artifact.action_pair_id == "ap-test"

    def test_generate_calls_chat_completion(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        generator.generate(sample_context, template)
        generator._client.chat_completion.assert_called_once()

    def test_generate_passes_model(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        generator.generate(sample_context, template)
        call_kwargs = generator._client.chat_completion.call_args
        assert call_kwargs.kwargs["model"] == "Qwen/Qwen2.5-Coder-32B-Instruct"

    def test_generate_passes_temperature(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        generator.generate(sample_context, template)
        call_kwargs = generator._client.chat_completion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7

    def test_generate_passes_max_tokens(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        generator.generate(sample_context, template)
        call_kwargs = generator._client.chat_completion.call_args
        assert call_kwargs.kwargs["max_tokens"] == 4096

    def test_generate_uses_template_when_provided(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
    ) -> None:
        from atomicguard.domain.prompts import PromptTemplate

        template = PromptTemplate(
            role="test role", constraints="test constraints", task="test task"
        )
        generator.generate(sample_context, template)

        call_kwargs = generator._client.chat_completion.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "test role" in user_msg
        assert "test task" in user_msg

    def test_generate_stores_context_snapshot(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        artifact = generator.generate(sample_context, template)
        assert artifact.context.specification == sample_context.specification
        assert artifact.context.constraints == sample_context.ambient.constraints

    def test_generate_sets_guard_result_none(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        artifact = generator.generate(sample_context, template)
        assert artifact.guard_result is None

    def test_generate_handles_empty_response(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        generator._client.chat_completion.return_value.choices[0].message.content = ""
        artifact = generator.generate(sample_context, template)
        assert artifact.content == ""

    def test_generate_handles_none_content(
        self,
        generator: HuggingFaceGenerator,
        sample_context,  # noqa: ANN001
        template: PromptTemplate,
    ) -> None:
        generator._client.chat_completion.return_value.choices[0].message.content = None
        artifact = generator.generate(sample_context, template)
        assert artifact.content == ""
