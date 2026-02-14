"""TestLocalizationGenerator: Locates existing test files and patterns.

Identifies test files, patterns, and fixtures related to the buggy code
to guide test generation in the appropriate style.
"""

from typing import Any

from examples.base.generators import PydanticAIGenerator
from examples.swe_bench_common.models import TestLocalization


class TestLocalizationGenerator(PydanticAIGenerator[TestLocalization]):
    """Generator that locates existing test files and patterns.

    Uses the localization from ap_localise_issue and structure from
    ap_structure to identify relevant test files and patterns.

    Context comes from prompt templates via {ap_localise_issue} and
    {ap_structure} placeholders.
    """

    output_type = TestLocalization

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("temperature", 0.1)
        super().__init__(**kwargs)
