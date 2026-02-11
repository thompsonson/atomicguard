"""Language configuration registry for multi-language SWE-Bench Pro support.

Provides per-language settings used by generators, guards, and the
experiment runner to adapt prompts and validation for Python, Go,
JavaScript, and TypeScript instances.
"""

import ast
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("swe_bench_pro.language")


@dataclass(frozen=True)
class LanguageConfig:
    """Configuration for a single programming language."""

    name: str
    code_block_tag: str
    file_extensions: tuple[str, ...]
    test_framework: str
    test_function_pattern: str
    syntax_check_fn: Callable[[str], tuple[bool, str]] | None = None


def _check_python_syntax(code: str) -> tuple[bool, str]:
    """Check Python syntax using ``ast.parse``."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"


def _check_basic_braces(code: str) -> tuple[bool, str]:
    """Heuristic brace-balance check for C-family languages."""
    opens = code.count("{")
    closes = code.count("}")
    if opens != closes:
        return False, f"Unbalanced braces: {opens} open, {closes} close"
    return True, ""


LANGUAGE_CONFIGS: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        name="python",
        code_block_tag="python",
        file_extensions=(".py",),
        test_framework="pytest",
        test_function_pattern=r"(def test_|class Test)",
        syntax_check_fn=_check_python_syntax,
    ),
    "go": LanguageConfig(
        name="go",
        code_block_tag="go",
        file_extensions=(".go",),
        test_framework="go test",
        test_function_pattern=r"func Test\w+",
        syntax_check_fn=_check_basic_braces,
    ),
    "javascript": LanguageConfig(
        name="javascript",
        code_block_tag="javascript",
        file_extensions=(".js", ".jsx", ".mjs"),
        test_framework="jest/mocha",
        test_function_pattern=r"(describe\(|it\(|test\()",
        syntax_check_fn=_check_basic_braces,
    ),
    "typescript": LanguageConfig(
        name="typescript",
        code_block_tag="typescript",
        file_extensions=(".ts", ".tsx"),
        test_framework="jest/mocha",
        test_function_pattern=r"(describe\(|it\(|test\()",
        syntax_check_fn=_check_basic_braces,
    ),
}


def get_language_config(language: str) -> LanguageConfig:
    """Look up a language configuration by name.

    Raises:
        ValueError: If *language* is not one of the supported languages.
    """
    config = LANGUAGE_CONFIGS.get(language.lower())
    if config is None:
        supported = ", ".join(sorted(LANGUAGE_CONFIGS))
        raise ValueError(
            f"Unsupported language {language!r}. Supported languages: {supported}"
        )
    return config


def detect_test_functions(code: str, language: str) -> bool:
    """Return *True* if *code* contains test function patterns for *language*."""
    config = get_language_config(language)
    return bool(re.search(config.test_function_pattern, code))
