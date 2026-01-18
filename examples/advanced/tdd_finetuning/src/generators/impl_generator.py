"""Implementation generator for TDD workflow."""

from openai import OpenAI
from typing import Any


class ImplGenerator:
    """Generates implementation code that passes given tests."""

    def __init__(self, model_config: dict[str, Any]):
        """Initialize implementation generator with model configuration.

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

    def generate(self, task_spec: dict[str, Any], test_code: str) -> str:
        """Generate implementation code that passes the given tests.

        Args:
            task_spec: Task specification dictionary
            test_code: Generated test code that implementation must satisfy

        Returns:
            Generated Python implementation code as a string
        """
        prompt = self._build_prompt(task_spec, test_code)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python developer. Generate clean, efficient implementations that pass all provided tests."
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

    def _build_prompt(self, task_spec: dict[str, Any], test_code: str) -> str:
        """Build the prompt for implementation generation.

        Args:
            task_spec: Task specification dictionary
            test_code: Test code to satisfy

        Returns:
            Formatted prompt string
        """
        specification = task_spec.get("specification", "")
        task_name = task_spec.get("name", "Unknown Task")

        prompt = f"""Generate a Python implementation for the following task that passes all the provided tests.

Task: {task_name}

Specification:
{specification}

Test Code:
```python
{test_code}
```

Requirements:
1. Write clean, efficient Python code that passes ALL the tests
2. Use type hints for function signatures
3. Include a brief docstring for the main function
4. Follow Python best practices and PEP 8 style
5. Handle edge cases shown in the tests
6. Do NOT include test code - only the implementation
7. Do NOT include imports (except typing if needed for type hints)

Output ONLY the Python implementation code, no explanations."""

        return prompt
