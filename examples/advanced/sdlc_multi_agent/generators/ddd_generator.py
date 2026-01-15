"""
DDD Generator: Domain-Driven Design Documentation Generation.

Generates DDD artifacts from user intent:
- domain_model.md
- infrastructure_requirements.md
- project_structure.md
- ubiquitous_language.md
"""

import json
from pathlib import Path
from typing import Any

from .base import BaseGenerator
from ..interfaces import GeneratorResult


class DDDGenerator(BaseGenerator):
    """Generate DDD documentation from user requirements.

    Responsibilities:
    - Transform user intent into formal DDD documentation
    - Identify bounded contexts, aggregates, entities
    - Define ubiquitous language
    - Specify architectural gates

    Output format:
    {
        "files": [
            {"path": "docs/domain_model.md", "content": "..."},
            {"path": "docs/infrastructure_requirements.md", "content": "..."},
            {"path": "docs/project_structure.md", "content": "..."},
            {"path": "docs/ubiquitous_language.md", "content": "..."}
        ],
        "metadata": {
            "entities": ["Evaluation", "Provider", "Result"],
            "gates": ["Gate 1: Domain Purity", "Gate 3: Factory Pattern"]
        }
    }
    """

    SYSTEM_PROMPT = """You are a Domain-Driven Design expert. Your task is to transform user requirements into formal DDD documentation.

Generate four documentation files:

1. **domain_model.md**:
   - Entities (with state transitions)
   - Value objects (with immutability constraints)
   - Aggregates (with boundaries)
   - Repository interfaces (abstractions only)

2. **infrastructure_requirements.md**:
   - Architectural gates (numbered)
   - Layer rules (domain, application, infrastructure)
   - Dependency injection constraints
   - Factory patterns

3. **project_structure.md**:
   - Directory tree
   - Layer assignments
   - Module responsibilities

4. **ubiquitous_language.md**:
   - Term definitions
   - Domain concepts

Output as JSON with this structure:
{
  "files": [
    {"path": "docs/domain_model.md", "content": "markdown content here"},
    ...
  ]
}

IMPORTANT:
- No implementation details in domain model
- No concrete classes in interfaces
- Include state transition rules for entities
"""

    async def generate(
        self, prompt: str, workspace: Path, context: dict[str, Any]
    ) -> GeneratorResult:
        """Generate DDD documentation from user intent.

        Args:
            prompt: User requirements/intent
            workspace: Working directory (files will be written here)
            context: Additional context

        Returns:
            GeneratorResult with DDD documentation
        """
        # Call LLM to generate DDD docs
        content, raw_messages = await self._call_llm(
            prompt=prompt, system=self.SYSTEM_PROMPT, temperature=0.7
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

            # Extract metadata
            metadata = data.get("metadata", {})
            if not metadata:
                # Extract entities and gates from content
                entities = []
                gates = []
                for file_entry in data["files"]:
                    content_text = file_entry.get("content", "")
                    if "domain_model.md" in file_entry["path"]:
                        # Simple extraction of entity names (lines starting with "### ")
                        for line in content_text.split("\n"):
                            if line.startswith("### ") and "Entity:" in line:
                                entity = line.replace("### ", "").replace(
                                    " Entity:", ""
                                ).strip()
                                entities.append(entity)
                    if "infrastructure_requirements.md" in file_entry["path"]:
                        # Extract gate names
                        for line in content_text.split("\n"):
                            if line.startswith("**Gate "):
                                gates.append(line.strip())

                metadata = {"entities": entities, "gates": gates}

            # Write files to workspace
            for file_entry in data["files"]:
                file_path = workspace / file_entry["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(file_entry["content"], encoding="utf-8")

            # Return structured result
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
