"""Tests for the render_template function.

This test file is used by the DynamicTestGuard to validate LLM-generated code.
The guard expects an `implementation` module with the `render_template` function.
"""

from implementation import render_template


class TestRenderTemplate:
    """Test cases for template rendering."""

    def test_variable_substitution(self):
        """Replace {{ variable }} with value from context."""
        result = render_template("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_variables(self):
        """Replace multiple variables in template."""
        result = render_template(
            "{{ greeting }}, {{ name }}!",
            {"greeting": "Hi", "name": "Alice"},
        )
        assert result == "Hi, Alice!"

    def test_no_variables(self):
        """Template without variables returns unchanged."""
        result = render_template("No vars here", {})
        assert result == "No vars here"

    def test_missing_key_unchanged(self):
        """Missing key leaves placeholder unchanged."""
        result = render_template("Hi {{ missing }}", {})
        assert result == "Hi {{ missing }}"

    def test_conditional_true(self):
        """Conditional with truthy value includes text."""
        result = render_template(
            "{% if show %}Visible{% endif %}",
            {"show": True},
        )
        assert result == "Visible"

    def test_conditional_false(self):
        """Conditional with falsy value removes block."""
        result = render_template(
            "{% if show %}Hidden{% endif %}",
            {"show": False},
        )
        assert result == ""

    def test_conditional_missing_key(self):
        """Conditional with missing key removes block."""
        result = render_template(
            "{% if missing %}Text{% endif %}",
            {},
        )
        assert result == ""

    def test_conditional_with_surrounding_text(self):
        """Conditional in middle of text."""
        result = render_template(
            "Before {% if show %}middle{% endif %} after",
            {"show": True},
        )
        assert result == "Before middle after"

    def test_variable_with_spaces(self):
        """Handle variable names with surrounding spaces."""
        result = render_template("Hello {{  name  }}!", {"name": "World"})
        assert result == "Hello World!"

    def test_empty_string_value(self):
        """Empty string value replaces placeholder."""
        result = render_template("Value: {{ value }}", {"value": ""})
        assert result == "Value: "

    def test_integer_value(self):
        """Integer value is converted to string."""
        result = render_template("Count: {{ count }}", {"count": 42})
        assert result == "Count: 42"
