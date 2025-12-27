"""
Internal generators for ADD workflow.

These generators are used internally by ADDGenerator to:
1. Parse documentation and extract architecture gates
2. Generate pytest-arch test code from gates
3. Write test files to the filesystem
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from examples.base import load_prompts
from pydantic import ValidationError
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import PromptedOutput
from pydantic_ai.providers.ollama import OllamaProvider

from atomicguard.domain.interfaces import GeneratorInterface
from atomicguard.domain.models import (
    Artifact,
    ArtifactStatus,
    Context,
    ContextSnapshot,
    FeedbackEntry,
)
from atomicguard.domain.prompts import PromptTemplate

from .models import (
    ArtifactManifest,
    FileToWrite,
    GatesExtractionResult,
    ProjectConfig,
    TestSuite,
)

# =============================================================================
# ConfigExtractorGenerator (Action Pair 0)
# =============================================================================


class ConfigExtractorGenerator(GeneratorInterface):
    """
    Extracts ProjectConfig (Ω) from documentation using LLM.

    This is Action Pair 0 in the ADD workflow. It extracts global constraints
    that apply to all subsequent action pairs.

    Per the paper's Hierarchical Context Composition:
        ℰ (Ambient Environment) = ⟨ℛ, Ω⟩

    The extracted ProjectConfig becomes Ω in context.ambient.constraints.
    """

    def __init__(
        self,
        model: str = "ollama:qwen2.5-coder:14b",
        base_url: str | None = None,
        prompts_file: str | Path | None = None,
        prompt_template: PromptTemplate | None = None,
    ):
        self._model = model
        self._base_url = base_url

        # Load from file if provided
        if prompts_file and prompt_template is None:
            prompts = load_prompts(Path(prompts_file))
            prompt_template = prompts.get("config_extraction")

        self._prompt_template = prompt_template

        # Require prompt template - no fallback to hardcoded prompts
        if prompt_template is None:
            raise ValueError(
                "prompt_template is required for ConfigExtractorGenerator - "
                "provide prompts_file or prompt_template"
            )
        system_prompt = f"{prompt_template.role}\n\n{prompt_template.constraints}"

        pydantic_model = _create_ollama_model(model, base_url)
        self._agent = Agent(
            pydantic_model,
            output_type=PromptedOutput(ProjectConfig),
            system_prompt=system_prompt,
            retries=0,  # Retries handled by AtomicGuard layer
        )

    def generate(
        self,
        context: Context,
        template: Any = None,  # noqa: ARG002
        action_pair_id: str = "add_config_extractor",
        workflow_id: str = "unknown",
        *,
        previous_attempt_id: str | None = None,
        attempt_number: int = 1,
        parent_action_pair_id: str | None = None,
    ) -> Artifact:
        """
        Extract project configuration from documentation.

        Args:
            context: Contains documentation in specification field
            template: Unused
            action_pair_id: Action pair identifier (from workflow)
            workflow_id: UUID of the workflow execution
            previous_attempt_id: ID of previous failed attempt for provenance
            attempt_number: Current attempt number (1-indexed)

        Returns:
            Artifact containing ProjectConfig as JSON
        """
        logger.debug("[ConfigExtractor] Building prompt...")
        prompt = f"Extract project configuration from:\n\n{context.specification}"

        # Add retry feedback if present
        if context.feedback_history:
            feedback = context.feedback_history[-1][1]  # Last feedback message
            if self._prompt_template and self._prompt_template.feedback_wrapper:
                feedback_prompt = self._prompt_template.feedback_wrapper.format(
                    feedback=feedback
                )
                prompt += f"\n\n{feedback_prompt}"
            else:
                prompt += f"\n\nPrevious attempt feedback: {feedback}"
            logger.debug("[ConfigExtractor] Including feedback from previous attempt")

        logger.debug(f"[ConfigExtractor] Prompt length: {len(prompt)} chars")

        messages: list = []
        try:
            logger.info("[ConfigExtractor] Calling LLM...")
            with capture_run_messages() as messages:
                result = self._agent.run_sync(prompt)
            logger.info("[ConfigExtractor] Got valid structured response")
            content = result.output.model_dump_json(indent=2)
        except UnexpectedModelBehavior as e:
            logger.warning(f"[ConfigExtractor] Output validation failed: {e}")
            model_response = ""
            if messages:
                for msg in messages:
                    if hasattr(msg, "text") and msg.text:
                        model_response += msg.text
                if model_response:
                    logger.debug(
                        f"[ConfigExtractor] Raw model response: {model_response[:500]}..."
                    )
            content = json.dumps(
                {
                    "error": "output_validation_failed",
                    "details": str(e),
                    "hint": "Model output did not match expected schema",
                    "model_response": model_response[:2000]
                    if model_response
                    else "unknown",
                }
            )
        except ValidationError as e:
            logger.warning(f"[ConfigExtractor] Schema validation failed: {e}")
            content = json.dumps({"error": "validation_failed", "details": str(e)})

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=content,
            previous_attempt_id=previous_attempt_id,
            parent_action_pair_id=parent_action_pair_id,
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=attempt_number,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=_create_context_snapshot(context, workflow_id),
        )


logger = logging.getLogger("add_workflow")


def _create_context_snapshot(context: Context, workflow_id: str) -> ContextSnapshot:
    """Create a ContextSnapshot from a Context."""
    return ContextSnapshot(
        workflow_id=workflow_id,
        specification=context.specification[:500],  # Truncate for storage
        constraints=context.ambient.constraints,
        feedback_history=tuple(
            FeedbackEntry(artifact_id=aid, feedback=fb)
            for aid, fb in context.feedback_history
        ),
        dependency_artifacts=context.dependency_artifacts,
    )


def _create_ollama_model(
    model: str, base_url: str | None = None
) -> OpenAIChatModel | str:
    """Create PydanticAI model, using OllamaProvider if base_url is specified.

    Args:
        model: Model name, optionally prefixed with "ollama:"
        base_url: Ollama API base URL (e.g., "http://host:11434/v1")

    Returns:
        OpenAIChatModel configured with OllamaProvider if base_url given,
        otherwise the model string for PydanticAI's default handling.
    """
    if base_url:
        # Strip ollama: prefix if present
        model_name = model.removeprefix("ollama:")
        # Ensure /v1 suffix
        if not base_url.endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"
        return OpenAIChatModel(
            model_name=model_name,
            provider=OllamaProvider(base_url=base_url),
        )
    # No base_url - use simple string format (relies on OLLAMA_BASE_URL env var)
    return model


# =============================================================================
# DocParserGenerator
# =============================================================================


class DocParserGenerator(GeneratorInterface):
    """
    Extracts architecture gates from documentation using PydanticAI.

    Uses structured output to ensure gates are properly typed and validated.
    Uses PromptedOutput mode to avoid tool call format issues with Ollama.
    """

    def __init__(
        self,
        model: str = "ollama:qwen2.5-coder:14b",
        base_url: str | None = None,
        prompts_file: str | Path | None = None,
        prompt_template: PromptTemplate | None = None,
    ):
        self._model = model
        self._base_url = base_url

        # Load from file if provided
        if prompts_file and prompt_template is None:
            prompts = load_prompts(Path(prompts_file))
            prompt_template = prompts.get("gates_extraction")

        self._prompt_template = prompt_template

        # Require prompt template - no fallback to hardcoded prompts
        if prompt_template is None:
            raise ValueError(
                "prompt_template is required for DocParserGenerator - "
                "provide prompts_file or prompt_template"
            )
        system_prompt = f"{prompt_template.role}\n\n{prompt_template.constraints}"

        pydantic_model = _create_ollama_model(model, base_url)
        self._agent = Agent(
            pydantic_model,
            output_type=PromptedOutput(GatesExtractionResult),
            system_prompt=system_prompt,
            retries=0,  # Retries handled by AtomicGuard layer
        )

    def generate(
        self,
        context: Context,
        template: Any = None,  # noqa: ARG002
        action_pair_id: str = "add_doc_parser",
        workflow_id: str = "unknown",
        *,
        previous_attempt_id: str | None = None,
        attempt_number: int = 1,
        parent_action_pair_id: str | None = None,
    ) -> Artifact:
        """
        Parse documentation and extract architecture gates.

        Args:
            context: Contains documentation in specification field
            template: Unused
            action_pair_id: Action pair identifier (from workflow)
            workflow_id: UUID of the workflow execution
            previous_attempt_id: ID of previous failed attempt for provenance
            attempt_number: Current attempt number (1-indexed)

        Returns:
            Artifact containing GatesExtractionResult as JSON
        """
        logger.debug("[DocParser] Building prompt...")
        prompt = f"Extract architecture gates from this documentation:\n\n{context.specification}"

        # Add retry feedback if present
        if context.feedback_history:
            feedback = context.feedback_history[-1][1]  # Last feedback message
            if self._prompt_template and self._prompt_template.feedback_wrapper:
                feedback_prompt = self._prompt_template.feedback_wrapper.format(
                    feedback=feedback
                )
                prompt += f"\n\n{feedback_prompt}"
            else:
                prompt += f"\n\nPrevious attempt feedback: {feedback}"
            logger.debug("[DocParser] Including feedback from previous attempt")

        logger.debug(f"[DocParser] Prompt length: {len(prompt)} chars")

        messages: list = []
        try:
            logger.info("[DocParser] Calling Ollama...")
            with capture_run_messages() as messages:
                result = self._agent.run_sync(prompt)
            logger.info("[DocParser] Got valid structured response")
            content = result.output.model_dump_json(indent=2)
        except UnexpectedModelBehavior as e:
            # Output validation failed - return error artifact for AtomicGuard retry
            # Include model response for debugging
            logger.warning(f"[DocParser] Output validation failed: {e}")
            model_response = ""
            if messages:
                # Filter for ModelResponse objects (not ModelRequest)
                for msg in messages:
                    if hasattr(msg, "text") and msg.text:
                        model_response += msg.text
                if model_response:
                    logger.debug(
                        f"[DocParser] Raw model response: {model_response[:500]}..."
                    )
                else:
                    logger.debug("[DocParser] No text response captured from model")
            content = json.dumps(
                {
                    "error": "output_validation_failed",
                    "details": str(e),
                    "hint": "Model output did not match expected schema",
                    "model_response": model_response[:2000]
                    if model_response
                    else "unknown",
                }
            )
        except ValidationError as e:
            # Schema validation failed - return error as artifact content
            # Guard will detect this and trigger retry
            logger.warning(f"[DocParser] Schema validation failed: {e}")
            content = json.dumps({"error": "validation_failed", "details": str(e)})

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=content,
            previous_attempt_id=previous_attempt_id,
            parent_action_pair_id=parent_action_pair_id,
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=attempt_number,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=_create_context_snapshot(context, workflow_id),
        )


# =============================================================================
# TestCodeGenerator
# =============================================================================


class TestCodeGenerator(GeneratorInterface):
    """
    Generates pytest-arch tests from extracted architecture gates.

    Receives gates from DocParserGenerator via internal_state and
    produces a complete TestSuite with test functions.
    Uses PromptedOutput mode to avoid tool call format issues with Ollama.
    """

    def __init__(
        self,
        model: str = "ollama:qwen2.5-coder:14b",
        base_url: str | None = None,
        prompts_file: str | Path | None = None,
        prompt_template: PromptTemplate | None = None,
    ):
        self._model = model
        self._base_url = base_url

        # Load from file if provided
        if prompts_file and prompt_template is None:
            prompts = load_prompts(Path(prompts_file))
            prompt_template = prompts.get("test_generation")

        self._prompt_template = prompt_template

        # Require prompt template - no fallback to hardcoded prompts
        if prompt_template is None:
            raise ValueError(
                "prompt_template is required for TestCodeGenerator - "
                "provide prompts_file or prompt_template"
            )
        system_prompt = f"{prompt_template.role}\n\n{prompt_template.constraints}"

        pydantic_model = _create_ollama_model(model, base_url)
        self._agent = Agent(
            pydantic_model,
            output_type=PromptedOutput(TestSuite),
            system_prompt=system_prompt,
            retries=0,  # Retries handled by AtomicGuard layer
        )

    def generate(
        self,
        context: Context,
        template: Any = None,  # noqa: ARG002
        action_pair_id: str = "add_test_codegen",
        workflow_id: str = "unknown",
        *,
        previous_attempt_id: str | None = None,
        attempt_number: int = 1,
        parent_action_pair_id: str | None = None,
    ) -> Artifact:
        """
        Generate test code from architecture gates.

        Reads gates from context.dependency_artifacts["gates"] (paper-aligned).
        Per paper: Generators access prior artifact IDs via context.dependency_artifacts,
        then retrieve full artifacts from ℛ (context.ambient.repository).

        Args:
            context: Generation context with dependencies from prior action pairs
            template: Unused
            action_pair_id: Action pair identifier (from workflow)
            workflow_id: UUID of the workflow execution
            previous_attempt_id: ID of previous failed attempt for provenance
            attempt_number: Current attempt number (1-indexed)

        Returns:
            Artifact containing TestSuite as JSON
        """
        # Get gates artifact ID from context.dependency_artifacts (paper-aligned)
        # Then retrieve full artifact from ℛ
        gates_id = context.get_dependency("gates")

        if gates_id is None:
            logger.warning("[TestCodeGen] No 'gates' in context.dependency_artifacts")
            content = json.dumps(
                {
                    "error": "missing_gates",
                    "details": "No 'gates' in context.dependency_artifacts",
                }
            )
        else:
            # Retrieve full artifact from ℛ (repository)
            gates_artifact = context.ambient.repository.get_artifact(gates_id)
            gates = GatesExtractionResult.model_validate_json(gates_artifact.content)
            logger.debug(
                f"[TestCodeGen] Found {len(gates.gates)} gates to generate tests for"
            )

            # Get source_root from Ω (context.ambient.constraints)
            # Per paper: ℰ (Ambient Environment) = ⟨ℛ, Ω⟩
            source_root = ""
            if context.ambient.constraints:
                try:
                    project_config = ProjectConfig.model_validate_json(
                        context.ambient.constraints
                    )
                    source_root = project_config.source_root
                    logger.debug(f"[TestCodeGen] Got source_root from Ω: {source_root}")
                except Exception as e:
                    logger.warning(f"[TestCodeGen] Failed to parse Ω: {e}")

            # Build fixture configuration
            if source_root:
                fixture_config = f"""
FIXTURE CONFIGURATION:
Source root from global constraints (Ω): {source_root}
Generate the fixture as:
@pytest.fixture(scope="module")
def evaluable():
    return get_evaluable_architecture("{source_root}", "{source_root}")
"""
            else:
                fixture_config = """
FIXTURE CONFIGURATION:
No source root found in Ω. Use placeholder:
@pytest.fixture(scope="module")
def evaluable():
    return get_evaluable_architecture("/project", "/project/src")
"""
                logger.warning("[TestCodeGen] No source_root in Ω, using placeholder")

            prompt = f"""Generate pytest-arch tests for these architecture gates:

{gates.model_dump_json(indent=2)}
{fixture_config}
Layer boundaries to enforce:
{chr(10).join(f"- {b}" for b in gates.layer_boundaries)}
"""
            # Add retry feedback if present
            if context.feedback_history:
                feedback = context.feedback_history[-1][1]
                if self._prompt_template and self._prompt_template.feedback_wrapper:
                    feedback_prompt = self._prompt_template.feedback_wrapper.format(
                        feedback=feedback
                    )
                    prompt += f"\n\n{feedback_prompt}"
                else:
                    prompt += f"\n\nFix these issues from previous attempt: {feedback}"
                logger.debug("[TestCodeGen] Including feedback from previous attempt")

            messages: list = []
            try:
                logger.info("[TestCodeGen] Calling Ollama...")
                with capture_run_messages() as messages:
                    result = self._agent.run_sync(prompt)
                logger.info("[TestCodeGen] Got valid structured response")
                content = result.output.model_dump_json(indent=2)
            except UnexpectedModelBehavior as e:
                # Output validation failed - return error artifact for AtomicGuard retry
                # Include model response for debugging
                logger.warning(f"[TestCodeGen] Output validation failed: {e}")
                model_response = ""
                if messages:
                    # Filter for ModelResponse objects (not ModelRequest)
                    for msg in messages:
                        if hasattr(msg, "text") and msg.text:
                            model_response += msg.text
                    if model_response:
                        logger.debug(
                            f"[TestCodeGen] Raw model response: {model_response[:500]}..."
                        )
                    else:
                        logger.debug(
                            "[TestCodeGen] No text response captured from model"
                        )
                content = json.dumps(
                    {
                        "error": "output_validation_failed",
                        "details": str(e),
                        "hint": "Model output did not match expected schema",
                        "model_response": model_response[:2000]
                        if model_response
                        else "unknown",
                    }
                )
            except ValidationError as e:
                logger.warning(f"[TestCodeGen] Schema validation failed: {e}")
                content = json.dumps({"error": "validation_failed", "details": str(e)})

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=content,
            previous_attempt_id=previous_attempt_id,
            parent_action_pair_id=parent_action_pair_id,
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=attempt_number,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=_create_context_snapshot(context, workflow_id),
        )


# =============================================================================
# FileWriterGenerator
# =============================================================================


class FileWriterGenerator(GeneratorInterface):
    """
    Deterministic generator that writes test files to filesystem.

    No LLM involved - simply assembles TestSuite into files and writes them.
    """

    def __init__(self, workdir: str | Path | None = None):
        self._workdir = Path(workdir) if workdir else Path.cwd()

    def generate(
        self,
        context: Context,
        template: Any = None,  # noqa: ARG002
        action_pair_id: str = "add_file_writer",
        workflow_id: str = "unknown",
        *,
        previous_attempt_id: str | None = None,
        attempt_number: int = 1,
        parent_action_pair_id: str | None = None,
    ) -> Artifact:
        """
        Write test files to filesystem.

        Reads test_suite from context.dependency_artifacts["test_suite"] (paper-aligned).
        Per paper: Generators access prior artifact IDs via context.dependency_artifacts,
        then retrieve full artifacts from ℛ (context.ambient.repository).

        Args:
            context: Generation context with dependencies from prior action pairs
            template: Unused
            action_pair_id: Action pair identifier (from workflow)
            workflow_id: UUID of the workflow execution
            previous_attempt_id: ID of previous failed attempt for provenance
            attempt_number: Current attempt number (1-indexed)

        Returns:
            Artifact containing ArtifactManifest as JSON
        """
        # Get test_suite artifact ID from context.dependency_artifacts (paper-aligned)
        # Then retrieve full artifact from ℛ
        test_suite_id = context.get_dependency("test_suite")

        if test_suite_id is None:
            logger.warning(
                "[FileWriter] No 'test_suite' in context.dependency_artifacts"
            )
            content = json.dumps(
                {
                    "error": "missing_test_suite",
                    "details": "No 'test_suite' in context.dependency_artifacts",
                }
            )
        else:
            # Retrieve full artifact from ℛ (repository)
            test_suite_artifact = context.ambient.repository.get_artifact(test_suite_id)
            suite = TestSuite.model_validate_json(test_suite_artifact.content)
            logger.debug(f"[FileWriter] Found TestSuite with {len(suite.tests)} tests")

            # Assemble test file content
            logger.debug("[FileWriter] Assembling test file content...")
            test_content = self._assemble_test_file(suite)

            files = [
                FileToWrite(
                    path="tests/__init__.py",
                    content="",
                ),
                FileToWrite(
                    path="tests/architecture/__init__.py",
                    content="",
                ),
                FileToWrite(
                    path="tests/architecture/test_gates.py",
                    content=test_content,
                ),
            ]

            # Write files to filesystem
            for f in files:
                path = self._workdir / f.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f.content)
                logger.debug(f"[FileWriter] Wrote {f.path} ({len(f.content)} chars)")

            manifest = ArtifactManifest(
                files=files,
                test_count=len(suite.tests),
                gates_covered=[t.gate_id for t in suite.tests],
            )
            logger.info(
                f"[FileWriter] Created manifest: {manifest.test_count} tests, {len(manifest.files)} files"
            )
            content = manifest.model_dump_json(indent=2)

        return Artifact(
            artifact_id=str(uuid4()),
            workflow_id=workflow_id,
            content=content,
            previous_attempt_id=previous_attempt_id,
            parent_action_pair_id=parent_action_pair_id,
            action_pair_id=action_pair_id,
            created_at=datetime.now().isoformat(),
            attempt_number=attempt_number,
            status=ArtifactStatus.PENDING,
            guard_result=None,
            feedback="",
            context=_create_context_snapshot(context, workflow_id),
        )

    def _assemble_test_file(self, suite: TestSuite) -> str:
        """Assemble TestSuite into a complete Python test file."""
        lines = [
            '"""',
            suite.module_docstring,
            '"""',
            "",
        ]

        # Add imports
        for imp in suite.imports:
            lines.append(imp)
        lines.append("")

        # Add fixtures
        for fixture in suite.fixtures:
            lines.append(fixture)
            lines.append("")

        # Add tests
        for test in suite.tests:
            lines.append(test.test_code)
            lines.append("")

        return "\n".join(lines)
