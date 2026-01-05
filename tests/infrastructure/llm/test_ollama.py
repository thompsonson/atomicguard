"""Tests for OllamaGenerator code extraction."""

import pytest

from atomicguard.infrastructure.llm.ollama import OllamaGenerator, OllamaGeneratorConfig


class TestExtractCode:
    """Tests for _extract_code() method."""

    @pytest.fixture
    def generator(self) -> OllamaGenerator:
        """Create generator for testing (doesn't require actual Ollama connection)."""
        # Mock the openai import at module level is not needed since we just
        # test _extract_code which doesn't use the client
        pytest.importorskip("openai")
        return OllamaGenerator(OllamaGeneratorConfig())

    def test_empty_string_returns_empty(self, generator: OllamaGenerator) -> None:
        """Empty input returns empty string."""
        result = generator._extract_code("")
        assert result == ""

    def test_whitespace_only_returns_empty(self, generator: OllamaGenerator) -> None:
        """Whitespace-only input returns empty string."""
        result = generator._extract_code("   \n\t\n   ")
        assert result == ""

    def test_python_code_block_extracts_code(self, generator: OllamaGenerator) -> None:
        """Python markdown block extracts code correctly."""
        content = """Here is the code:
```python
def add(a, b):
    return a + b
```
That should work."""

        result = generator._extract_code(content)
        assert result == "def add(a, b):\n    return a + b"

    def test_generic_code_block_extracts_code(self, generator: OllamaGenerator) -> None:
        """Generic markdown block extracts code when no python block."""
        content = """Here is the code:
```
x = 1
y = 2
```
Done."""

        result = generator._extract_code(content)
        assert result == "x = 1\ny = 2"

    def test_prefers_python_block_over_generic(
        self, generator: OllamaGenerator
    ) -> None:
        """Python block is preferred when both exist."""
        content = """
```
generic code
```
```python
python code
```
"""
        result = generator._extract_code(content)
        assert result == "python code"

    def test_code_starting_with_def(self, generator: OllamaGenerator) -> None:
        """Code starting with 'def ' is extracted when no blocks found."""
        content = """Some explanation
def hello():
    print("hi")

More text"""

        result = generator._extract_code(content)
        assert result.startswith("def hello():")

    def test_code_starting_with_import(self, generator: OllamaGenerator) -> None:
        """Code starting with 'import ' is extracted when no blocks found."""
        content = """Here's what you need:
import os
import sys

def main():
    pass"""

        result = generator._extract_code(content)
        assert result.startswith("import os")

    def test_code_starting_with_class(self, generator: OllamaGenerator) -> None:
        """Code starting with 'class ' is extracted when no blocks found."""
        content = """The solution is:
class MyClass:
    def __init__(self):
        pass"""

        result = generator._extract_code(content)
        assert result.startswith("class MyClass:")

    def test_plain_text_returns_empty(self, generator: OllamaGenerator) -> None:
        """Plain text without code patterns returns empty string (the fix)."""
        content = "I cannot generate that code because it violates safety guidelines."
        result = generator._extract_code(content)
        assert result == ""

    def test_plain_text_with_code_mention_returns_empty(
        self, generator: OllamaGenerator
    ) -> None:
        """Text that mentions code but isn't code returns empty."""
        content = """I would suggest writing a function like this:
- First, define your parameters
- Then, implement the logic
- Finally, return the result

Let me know if you need more help!"""

        result = generator._extract_code(content)
        assert result == ""

    def test_partial_code_block_returns_empty(self, generator: OllamaGenerator) -> None:
        """Incomplete code blocks return empty (not matched by regex)."""
        content = """Here's the code:
```python
def incomplete():
    pass
"""  # Missing closing ```
        result = generator._extract_code(content)
        # Falls through to def pattern which should match
        assert "def incomplete" in result

    def test_multiple_python_blocks_extracts_first(
        self, generator: OllamaGenerator
    ) -> None:
        """Multiple python blocks - first one is extracted."""
        content = """First:
```python
first = 1
```
Second:
```python
second = 2
```"""

        result = generator._extract_code(content)
        assert result == "first = 1"
