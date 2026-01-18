"""Test generator for TDD workflow."""

from openai import OpenAI
from typing import Any


class TestGenerator:
    """Generates pytest test functions from task specifications."""

    def __init__(self, model_config: dict[str, Any]):
        """Initialize test generator with model configuration.

        Args:
            model_config: Dictionary containing model name, base_url, api_key, etc.
        """
        self.model_name = model_config["name"]
        self.client = OpenAI(
            base_url=model_config["base_url"],
            api_key=model_config["api_key"],
        )
        self.temperature = model_config.get("temperature", 0.2)
        self.max_tokens = model_config.get("max_tokens", 2048)

    def generate(self, task_spec: dict[str, Any]) -> str:
        """Generate pytest test code from task specification.

        Args:
            task_spec: Task specification dictionary with 'specification' field

        Returns:
            Generated Python test code as a string
        """
        prompt = self._build_prompt(task_spec)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python developer specializing in test-driven development. Generate comprehensive pytest test functions based on specifications."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        generated_code = response.choices[0].message.content

        # Extract code from markdown code blocks if present
        if "```python" in generated_code:
            code_start = generated_code.find("```python") + 9
            code_end = generated_code.find("```", code_start)
            generated_code = generated_code[code_start:code_end].strip()
        elif "```" in generated_code:
            code_start = generated_code.find("```") + 3
            code_end = generated_code.find("```", code_start)
            generated_code = generated_code[code_start:code_end].strip()

        return generated_code

    def _build_prompt(self, task_spec: dict[str, Any]) -> str:
        """Build the prompt for test generation.

        Args:
            task_spec: Task specification dictionary

        Returns:
            Formatted prompt string
        """
        specification = task_spec.get("specification", "")
        task_name = task_spec.get("name", "Unknown Task")

        prompt = f"""Generate pytest test functions for the following task:

Task: {task_name}

Specification:
{specification}

Requirements:
1. Write comprehensive pytest test functions that cover the examples and edge cases
2. Include at least 5-7 test cases covering normal cases, edge cases, and error conditions
3. Use descriptive test function names (test_<scenario>)
4. Include docstrings explaining what each test validates
5. Make tests independent and self-contained
6. Do NOT include the implementation - only test functions
7. Import pytest at the top
8. Assume the function being tested will be imported from a module called 'solution'

Output ONLY the Python test code, no explanations."""

        return prompt
