"""ClassificationGenerator: Classifies problem instances by complexity.

Part of the meta-level pipeline in Arms 20-21. Classifies the problem
into a category (trivial_fix, single_file_bug, multi_file_bug, api_change,
refactor) that determines which workflow template to generate.

Used by: ap_classify_problem in Arms 20, 21
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator

from examples.swe_bench_common.models import ProblemClassification


class ClassificationGenerator(PydanticAIGenerator[ProblemClassification]):
    """Generator that classifies problem instances by type and complexity.

    Reads the problem statement and repository file listing. Outputs a
    category and complexity estimate used by ap_generate_workflow to
    select the appropriate pipeline.

    Context comes from prompt templates via {specification} placeholder.
    """

    output_type = ProblemClassification

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.1)
        super().__init__(**kwargs)
