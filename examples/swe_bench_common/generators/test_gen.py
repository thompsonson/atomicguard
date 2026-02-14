"""TestGenerator: Generates failing test code that reproduces a bug.

Uses PydanticAI structured output to produce test code via the
GeneratedTest model.  The artifact content is the raw test code
string (not JSON) so that downstream guards and generators can
consume it directly.
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator
from examples.swe_bench_common.models import GeneratedTest

from atomicguard.domain.models import Context


class TestGenerator(PydanticAIGenerator[GeneratedTest]):
    """Generator that produces failing test code to reproduce a bug.

    Reads analysis from prior step via dependency artifacts.
    Outputs raw test code (not JSON) stored as Artifact.content
    so that TestSyntaxGuard and PatchGenerator can consume it directly.

    Context comes from prompt templates via {ap_analysis} placeholder.
    """

    output_type = GeneratedTest

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.3)
        super().__init__(**kwargs)

    def _process_output(self, output: GeneratedTest, context: Context) -> str:  # noqa: ARG002
        """Return raw test code so guards and downstream generators can consume it directly."""
        return output.test_code
