"""Multi-language patch generator.

Subclasses :class:`PatchGenerator` from swe_bench_common to customize
the code block tag based on target language.
"""

from typing import Any

from examples.swe_bench_common.generators import PatchGenerator

from ..language import LanguageConfig


class MultiLangPatchGenerator(PatchGenerator):
    """Language-aware patch generator.

    Extends PatchGenerator to use language-specific code fence tags.
    Context comes from prompt templates via placeholders; file content
    injection uses the appropriate language tag.
    """

    def __init__(self, language_config: LanguageConfig, **kwargs: Any):
        # Set code_block_tag based on language config
        kwargs["code_block_tag"] = language_config.code_block_tag
        super().__init__(**kwargs)
        self._lang = language_config
