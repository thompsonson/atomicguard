"""
Coder Generator: Implementation Code Generation.

Generates Python implementation based on DDD documentation.
"""

import json
from pathlib import Path
from typing import Any

from .base import BaseGenerator
from ..interfaces import GeneratorResult


class CoderGenerator(BaseGenerator):
    """Generate implementation code from DDD documentation.

    Responsibilities:
    - Read DDD documentation from workspace
    - Generate domain entities and value objects
    - Generate application services
    - Generate infrastructure implementations
    - Follow architectural constraints

    Output format:
    {
        "files": [
            {"path": "src/domain/entities/evaluation.py", "content": "..."},
            {"path": "src/domain/value_objects/provider.py", "content": "..."},
            ...
        ],
        "metadata": {
            "total_files": 5,
            "total_lines": 250
        }
    }
    """

    SYSTEM_PROMPT = """You are an expert Python developer implementing Domain-Driven Design patterns.

Your task is to generate implementation code based on DDD documentation.

Input: You will have access to these files in the workspace:
- docs/domain_model.md
- docs/infrastructure_requirements.md
- docs/project_structure.md
- docs/ubiquitous_language.md

Requirements:
1. Follow the layer structure from project_structure.md
2. Respect architectural gates from infrastructure_requirements.md
3. Implement entities with state transitions from domain_model.md
4. Use type hints (Python 3.10+)
5. Keep domain layer pure (no external dependencies)
6. Use factory patterns where specified
7. Include docstrings with business rules

Output as JSON:
{
  "files": [
    {"path": "src/domain/entities/example.py", "content": "python code here"},
    ...
  ]
}

IMPORTANT:
- Domain layer: ONLY standard library imports
- No implementation details in domain
- Use abstractions (protocols) for infrastructure
- Include __init__.py for all packages
"""

    async def generate(
        self, prompt: str, workspace: Path, context: dict[str, Any]
    ) -> GeneratorResult:
        """Generate implementation code from DDD docs.

        Args:
            prompt: Instruction (typically "Generate implementation")
            workspace: Working directory (DDD docs are here)
            context: Dependencies (contains DDD documentation)

        Returns:
            GeneratorResult with implementation files
        """
        # Read DDD documentation from workspace
        docs_dir = workspace / "docs"
        ddd_docs = {}

        if docs_dir.exists():
            for doc_file in ["domain_model.md", "infrastructure_requirements.md",
                             "project_structure.md", "ubiquitous_language.md"]:
                doc_path = docs_dir / doc_file
                if doc_path.exists():
                    ddd_docs[doc_file] = doc_path.read_text(encoding="utf-8")

        # Construct prompt with documentation context
        full_prompt = f"""{prompt}

# Available Documentation

"""
        for doc_name, doc_content in ddd_docs.items():
            full_prompt += f"## {doc_name}\n\n{doc_content[:2000]}...\n\n"

        full_prompt += """
Generate Python implementation files following the DDD documentation.
Output as JSON with structure: {"files": [{"path": "...", "content": "..."}]}
"""

        # Call LLM
        content, raw_messages = await self._call_llm(
            prompt=full_prompt, system=self.SYSTEM_PROMPT, temperature=0.7
        )

        # Parse JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content

            data = json.loads(json_content)

            # Validate structure
            if "files" not in data:
                raise ValueError("Missing 'files' key in generated JSON")

            # Write files to workspace
            for file_entry in data["files"]:
                file_path = workspace / file_entry["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(file_entry["content"], encoding="utf-8")

            # Compute metadata
            total_lines = sum(
                file_entry["content"].count("\n") for file_entry in data["files"]
            )
            metadata = {
                "total_files": len(data["files"]),
                "total_lines": total_lines,
            }

            return GeneratorResult(
                content=json.dumps(data, indent=2),
                metadata=metadata,
                raw_messages=raw_messages,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If parsing fails, return error in result
            error_data = {
                "files": [],
                "error": f"Failed to parse LLM output: {e}",
                "raw_output": content[:500],
            }
            return GeneratorResult(
                content=json.dumps(error_data, indent=2),
                metadata={"error": str(e)},
                raw_messages=raw_messages,
            )
