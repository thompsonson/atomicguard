"""
ActionPair: Atomic generation-verification transaction.

Paper Definition 6: A = ⟨a_gen, G⟩
Precondition ρ is handled at the Workflow level.
"""

from atomicguard.domain.interfaces import GeneratorInterface, GuardInterface
from atomicguard.domain.models import Artifact, Context, GuardResult
from atomicguard.domain.prompts import PromptTemplate


class ActionPair:
    """
    Atomic generation-verification transaction.

    Couples a generator with a guard to form an indivisible unit.
    The precondition ρ is handled at the Workflow level.
    """

    def __init__(
        self,
        generator: GeneratorInterface,
        guard: GuardInterface,
        prompt_template: PromptTemplate,
    ):
        """
        Args:
            generator: The artifact generator
            guard: The validator for generated artifacts
            prompt_template: Structured prompt template for generation
        """
        self._generator = generator
        self._guard = guard
        self._prompt_template = prompt_template

    @property
    def generator(self) -> GeneratorInterface:
        """Access the generator (read-only)."""
        return self._generator

    @property
    def guard(self) -> GuardInterface:
        """Access the guard (read-only)."""
        return self._guard

    def execute(
        self,
        context: Context,
        dependencies: dict[str, Artifact] | None = None,
        action_pair_id: str = "unknown",
        workflow_id: str = "unknown",
    ) -> tuple[Artifact, GuardResult]:
        """
        Execute the atomic generate-then-validate transaction.

        Args:
            context: Generation context
            dependencies: Artifacts from prior workflow steps
            action_pair_id: Identifier for this action pair
            workflow_id: UUID of the workflow execution instance

        Returns:
            Tuple of (generated artifact, guard result)
        """
        dependencies = dependencies or {}
        artifact = self._generator.generate(
            context, self._prompt_template, action_pair_id, workflow_id
        )

        # If the generator flagged an error (e.g. PydanticAI validation failure),
        # skip the domain guard — the generator error IS the rejection.
        generator_error = artifact.metadata.get("generator_error")
        if generator_error:
            result = GuardResult(
                passed=False,
                feedback=generator_error,
                guard_name="GeneratorValidation",
            )
            return artifact, result

        result = self._guard.validate(artifact, **dependencies)
        return artifact, result
