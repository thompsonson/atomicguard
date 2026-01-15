"""
Identity Generator: Pass-through generator for deterministic phases.

Used for phases that don't require LLM generation (e.g., Tester).
"""

import json
from pathlib import Path
from typing import Any

from ..interfaces import GeneratorResult, IGenerator


class IdentityGenerator(IGenerator):
    """Pass-through generator that returns empty content.

    Used for phases where the "generation" is actually deterministic
    validation (like running pytest).

    Responsibilities:
    - Return empty content (actual work done by Guard)
    - Maintain interface consistency

    Does NOT:
    - Call LLM
    - Generate content
    """

    async def generate(
        self, prompt: str, workspace: Path, context: dict[str, Any]
    ) -> GeneratorResult:
        """Return empty content (pass-through).

        Args:
            prompt: Ignored (not used)
            workspace: Ignored (not used)
            context: Ignored (not used)

        Returns:
            GeneratorResult with empty content

        Note:
            For Tester, the actual validation happens in the Guard
            (AllTestsPassGuard) which runs pytest. This generator is
            just a placeholder to maintain workflow consistency.
        """
        return GeneratorResult(
            content=json.dumps({"files": [], "metadata": {"pass_through": True}}),
            metadata={"pass_through": True},
            raw_messages=[],
        )
