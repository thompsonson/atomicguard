"""
Guards for Multi-Agent SDLC workflow.

Responsibilities:
- Validate artifact content OR filesystem state
- Return clear pass/fail verdict with feedback
- Deterministic validation only

Does NOT:
- Call LLM (must be deterministic)
- Retry logic (Orchestrator's job)
- Store artifacts (DAG's job)
- Materialize files (WorkspaceService's job)
"""

from .all_tests_pass_guard import AllTestsPassGuard
from .documentation_guard import DocumentationGuard

__all__ = [
    "DocumentationGuard",
    "AllTestsPassGuard",
]
