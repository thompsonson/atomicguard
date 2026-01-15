"""
Multi-Agent SDLC Workflow - Proof of Concept.

Demonstrates AtomicGuard's multi-agent coordination with clear separation of concerns:
- Three agents: DDD → Coder → Tester
- DAG as shared source of truth
- WorkspaceService for filesystem ↔ DAG synchronization
- Retry budget management per phase

Key Components:
- WorkspaceService: Bidirectional filesystem ↔ DAG synchronization
- Generators: DDD, Coder, Identity (LLM-based content generation)
- Guards: Documentation, AllTestsPass (deterministic validation)
- Orchestrator: Phase sequencing, retry management, coordination

Architecture Principles:
1. DAG is the source of truth
2. Workspaces are ephemeral (persisted for debugging)
3. Option C: Pre-materialize ALL upstream artifacts before each phase
4. Clear interfaces between layers
"""

__version__ = "1.0.0"
