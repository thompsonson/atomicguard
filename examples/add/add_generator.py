"""
ADDGenerator: Architecture-Driven Development generator.

This generator implements GeneratorInterface and internally orchestrates
multiple action pairs to extract architecture gates from documentation
and generate pytest-arch tests.
"""

import json
import logging
from pathlib import Path
from typing import Any

from atomicguard.domain.exceptions import EscalationRequired
from atomicguard.domain.interfaces import ArtifactDAGInterface, GeneratorInterface
from atomicguard.domain.models import (
    AmbientEnvironment,
    Artifact,
    Context,
    ContextSnapshot,
    GuardResult,
)
from atomicguard.domain.prompts import PromptTemplate
from atomicguard.guards import CompositeGuard

from .generators import (
    ConfigExtractorGenerator,
    DocParserGenerator,
    FileWriterGenerator,
    TestCodeGenerator,
)
from .guards import (
    ArtifactStructureGuard,
    ConfigGuard,
    GatesExtractedGuard,
    PytestArchAPIGuard,
    TestNamingGuard,
    TestSyntaxGuard,
)
from .models import GatesExtractionResult, ProjectConfig, TestSuite

logger = logging.getLogger("add_workflow")


class ADDGenerator(GeneratorInterface):
    """
    Architecture-Driven Development generator.

    Internally orchestrates 4 action pairs:
    0. ConfigExtractor → ConfigGuard (extracts Ω)
    1. DocParser → GatesExtractedGuard
    2. TestCodeGen → TestSyntaxGuard + TestNamingGuard
    3. FileWriter → ArtifactStructureGuard

    Per the paper's Hierarchical Context Composition:
        C_total = ⟨ℰ, C_local, H_feedback⟩
        ℰ (Ambient Environment) = ⟨ℛ, Ω⟩

    Action Pair 0 extracts Ω (Global Constraints) from the specification.
    After AP0, a NEW Context is created with Ω in context.ambient.constraints.
    All subsequent action pairs receive Ω via the updated context.

    From the parent workflow's perspective, this generator is atomic -
    it receives documentation context and returns a JSON manifest of
    generated test files.

    The internal action pairs use their own retry loop, independent of
    the parent workflow's retry loop.
    """

    def __init__(
        self,
        model: str = "ollama:qwen2.5-coder:14b",
        base_url: str | None = None,
        rmax: int = 3,
        workdir: Path | None = None,
        min_gates: int = 1,
        min_tests: int = 1,
        artifact_dag: ArtifactDAGInterface | None = None,
        prompts: dict[str, PromptTemplate] | None = None,
    ):
        """
        Initialize ADDGenerator.

        Args:
            model: Ollama model to use for LLM calls
            base_url: Ollama API base URL (e.g., "http://host:11434/v1")
            rmax: Maximum retries per internal action pair
            workdir: Directory to write test files to
            min_gates: Minimum number of gates required
            min_tests: Minimum number of tests required
            artifact_dag: Optional DAG for persisting internal artifacts
            prompts: Prompt templates for each action pair (keys: gates_extraction, test_generation)
        """
        self._model = model
        self._base_url = base_url
        self._rmax = rmax
        self._workdir = workdir or Path.cwd()
        self._min_gates = min_gates
        self._min_tests = min_tests
        self._artifact_dag = artifact_dag
        self._prompts = prompts or {}

    def generate(
        self,
        context: Context,
        template: PromptTemplate | None = None,  # noqa: ARG002
    ) -> Artifact:
        """
        Generate architecture tests from documentation.

        Args:
            context: Contains architecture documentation in specification field
            template: Unused

        Returns:
            Artifact containing ArtifactManifest JSON

        Raises:
            EscalationRequired: If any internal action pair exhausts retries
        """
        logger.info("[ADD] Starting generation pipeline")
        logger.debug(
            f"[ADD] Doc length: {len(context.specification)} chars, rmax={self._rmax}"
        )

        # Action Pair 0: Extract project config (Ω)
        # Per paper: Ω = Global Constraints, stored in context.ambient.constraints
        logger.info("[ADD] === Action Pair 0: Config Extraction ===")
        config_artifact, config_result = self._execute_config_extraction(context)
        logger.info(f"[ADD] Extracted Ω: source_root={config_result.source_root}")

        # Create NEW Context with populated Ω
        # Context is frozen (immutable), so we must create a new instance
        updated_ambient = AmbientEnvironment(
            repository=context.ambient.repository,
            constraints=config_result.model_dump_json(),
        )
        context = Context(
            ambient=updated_ambient,
            specification=context.specification,
            current_artifact=context.current_artifact,
            feedback_history=context.feedback_history,
            dependencies=context.dependencies,
        )
        logger.debug("[ADD] Created new Context with Ω in ambient.constraints")

        # Action Pair 1: Extract gates
        # Artifact stored in ℛ (ArtifactDAG) by _run_action_pair()
        logger.info("[ADD] === Action Pair 1: Gates Extraction ===")
        gates_artifact, gates_result = self._execute_gates_extraction(context)
        logger.info(f"[ADD] Extracted {len(gates_result.gates)} gates")

        # Create context with gates artifact in dependencies for AP2
        # Per paper: Generators access prior artifacts via context.dependencies
        context = Context(
            ambient=context.ambient,
            specification=context.specification,
            current_artifact=context.current_artifact,
            feedback_history=(),  # Reset for new action pair
            dependencies=(("gates", gates_artifact),),
        )
        logger.debug("[ADD] Added gates_artifact to context.dependencies")

        # Action Pair 2: Generate tests
        logger.info("[ADD] === Action Pair 2: Test Generation ===")
        test_artifact, test_suite = self._execute_test_generation(context)
        logger.info(f"[ADD] Generated {len(test_suite.tests)} tests")

        # Create context with test_suite artifact in dependencies for AP3
        context = Context(
            ambient=context.ambient,
            specification=context.specification,
            current_artifact=context.current_artifact,
            feedback_history=(),  # Reset for new action pair
            dependencies=(("test_suite", test_artifact),),
        )
        logger.debug("[ADD] Added test_artifact to context.dependencies")

        # Action Pair 3: Write files
        logger.info("[ADD] === Action Pair 3: File Writing ===")
        manifest_artifact = self._execute_file_writing(context)
        logger.info("[ADD] Files written successfully")

        return manifest_artifact

    def _execute_config_extraction(
        self,
        context: Context,
    ) -> tuple[Artifact, ProjectConfig]:
        """
        Execute config extraction action pair with retry loop.

        This is Action Pair 0 - it extracts Ω (Global Constraints) from
        the specification before any other action pairs run.

        Returns:
            Tuple of (artifact stored in ℛ, parsed ProjectConfig)
        """
        generator = ConfigExtractorGenerator(
            self._model,
            base_url=self._base_url,
            prompt_template=self._prompts.get("config_extraction"),
        )
        guard = ConfigGuard()

        artifact = self._run_action_pair(
            generator=generator,
            guard=guard,
            context=context,
            pair_name="config_extraction",
        )

        # Parse result
        data = json.loads(artifact.content)
        return artifact, ProjectConfig.model_validate(data)

    def _execute_gates_extraction(
        self,
        context: Context,
    ) -> tuple[Artifact, GatesExtractionResult]:
        """
        Execute gate extraction action pair with retry loop.

        Returns:
            Tuple of (artifact stored in ℛ, parsed GatesExtractionResult)
        """
        generator = DocParserGenerator(
            self._model,
            base_url=self._base_url,
            prompt_template=self._prompts.get("gates_extraction"),
        )
        guard = GatesExtractedGuard(min_gates=self._min_gates)

        artifact = self._run_action_pair(
            generator=generator,
            guard=guard,
            context=context,
            pair_name="gates_extraction",
        )

        # Parse result
        data = json.loads(artifact.content)
        return artifact, GatesExtractionResult.model_validate(data)

    def _execute_test_generation(
        self,
        context: Context,
    ) -> tuple[Artifact, TestSuite]:
        """
        Execute test generation action pair with retry loop.

        Reads gates from context.dependencies["gates"].

        Returns:
            Tuple of (artifact stored in ℛ, parsed TestSuite)
        """
        generator = TestCodeGenerator(
            self._model,
            base_url=self._base_url,
            prompt_template=self._prompts.get("test_generation"),
        )
        guard = CompositeGuard(
            TestSyntaxGuard(), TestNamingGuard(), PytestArchAPIGuard()
        )

        artifact = self._run_action_pair(
            generator=generator,
            guard=guard,
            context=context,
            pair_name="test_generation",
        )

        # Parse result
        data = json.loads(artifact.content)
        return artifact, TestSuite.model_validate(data)

    def _execute_file_writing(
        self,
        context: Context,
    ) -> Artifact:
        """
        Execute file writing action pair with retry loop.

        Reads test_suite from context.dependencies["test_suite"].

        Returns:
            Artifact containing ArtifactManifest JSON
        """
        generator = FileWriterGenerator(workdir=self._workdir)
        guard = ArtifactStructureGuard(min_tests=self._min_tests)

        return self._run_action_pair(
            generator=generator,
            guard=guard,
            context=context,
            pair_name="file_writing",
        )

    def _run_action_pair(
        self,
        generator: GeneratorInterface,
        guard: Any,  # GuardInterface or CompositeGuard
        context: Context,
        pair_name: str,
    ) -> Artifact:
        """
        Run an action pair with retry logic.

        Similar to DualStateAgent but simpler - stores artifacts to ℛ (DAG)
        for debugging and tracks provenance across retries.

        Generators access prior artifacts via context.dependencies (paper-aligned).
        """
        feedback_history: list[tuple[Artifact, str]] = []
        current_context = context
        previous_attempt_id: str | None = None

        for attempt in range(self._rmax + 1):
            logger.debug(f"[{pair_name}] Attempt {attempt + 1}/{self._rmax + 1}")

            # Generate with provenance tracking
            # Prior artifacts are accessed via context.dependencies (paper-aligned)
            logger.debug(f"[{pair_name}] Calling generator...")
            artifact = generator.generate(
                current_context,
                template=None,
                previous_attempt_id=previous_attempt_id,  # type: ignore
                attempt_number=attempt + 1,  # type: ignore
            )
            logger.debug(f"[{pair_name}] Generated {len(artifact.content)} chars")

            # Store artifact in DAG for debugging
            if self._artifact_dag:
                self._artifact_dag.store(artifact)
                logger.debug(f"[{pair_name}] Stored artifact {artifact.artifact_id}")

            # Validate
            logger.debug(f"[{pair_name}] Running guard validation...")
            result: GuardResult = guard.validate(artifact)

            if result.passed:
                logger.info(f"[{pair_name}] ✓ Passed: {result.feedback}")
                return artifact

            if result.fatal:
                logger.error(f"[{pair_name}] ✗ FATAL: {result.feedback}")
                raise EscalationRequired(artifact, result.feedback)

            # Accumulate feedback for retry
            logger.warning(
                f"[{pair_name}] ✗ Failed (attempt {attempt + 1}): {result.feedback}"
            )
            feedback_history.append((artifact, result.feedback))
            previous_attempt_id = artifact.artifact_id
            current_context = self._refine_context(context, feedback_history)

        # Exhausted retries - escalate
        logger.error(f"[{pair_name}] Exhausted {self._rmax + 1} attempts")
        raise EscalationRequired(
            artifact,
            f"Action pair '{pair_name}' exhausted {self._rmax} retries. "
            f"Last feedback: {result.feedback}",
        )

    def _refine_context(
        self,
        original_context: Context,
        feedback_history: list[tuple[Artifact, str]],
    ) -> Context:
        """Create refined context with accumulated feedback."""
        return Context(
            ambient=original_context.ambient,
            specification=original_context.specification,
            current_artifact=feedback_history[-1][0].content
            if feedback_history
            else None,
            feedback_history=tuple(
                (artifact.artifact_id, feedback)
                for artifact, feedback in feedback_history
            ),
            dependencies=original_context.dependencies,
        )

    def _create_context_snapshot(self, context: Context) -> ContextSnapshot:
        """Create a ContextSnapshot from a Context."""
        return ContextSnapshot(
            specification=context.specification,
            constraints=context.ambient.constraints,
            feedback_history=(),
            dependency_ids=(),
        )
