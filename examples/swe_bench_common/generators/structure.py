"""StructureGenerator: Analyzes project structure, imports, and dependencies.

Provides high-level understanding of the codebase layout to inform
downstream steps like localization and test generation.
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator

from examples.swe_bench_common.models import ProjectStructure


class StructureGenerator(PydanticAIGenerator[ProjectStructure]):
    """Generator that analyzes project structure.

    Outputs a ProjectStructure JSON identifying top-level modules,
    test framework, test directories, import conventions, and
    key dependencies.

    Context comes from prompt templates via {specification} placeholder.
    """

    output_type = ProjectStructure

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.1)
        super().__init__(**kwargs)
