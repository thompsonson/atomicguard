"""Multi-language test generator.

Subclasses :class:`TestGenerator` from swe_bench_common. PydanticAI
handles structured output parsing via the ``GeneratedTest`` model.

Language-specific test framework instructions are included in the
prompt templates (prompts.json) rather than being injected here.
"""

from typing import Any

from examples.swe_bench_common.generators import TestGenerator

from ..language import LanguageConfig


class MultiLangTestGenerator(TestGenerator):
    """Language-aware test generator.

    Extends TestGenerator for multi-language support. Language-specific
    test framework instructions are defined in prompt templates.
    Context comes from prompt templates via {ap_analysis} placeholder.
    """

    def __init__(self, language_config: LanguageConfig, **kwargs: Any):
        super().__init__(**kwargs)
        self._lang = language_config
