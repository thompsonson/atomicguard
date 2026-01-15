"""
Generators for Multi-Agent SDLC workflow.

Responsibilities:
- Call LLM with prompts
- Use Claude SDK skills (via filesystem operations)
- Format input/output

Does NOT:
- Validate content (Guard's job)
- Retry logic (Orchestrator's job)
- Manage workspace lifecycle (WorkspaceService's job)
- Store artifacts (DAG's job)
"""

from .base import BaseGenerator
from .coder_generator import CoderGenerator
from .ddd_generator import DDDGenerator
from .identity_generator import IdentityGenerator

__all__ = [
    "BaseGenerator",
    "DDDGenerator",
    "CoderGenerator",
    "IdentityGenerator",
]
