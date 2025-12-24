"""
AtomicGuard GUI Example - Live Workflow Monitoring with Gradio.

This module provides a web-based GUI for monitoring AtomicGuard workflow execution
in real-time, including:
- Workflow visualization (DAG of steps)
- Live execution status and retry tracking
- Log streaming
- Artifact browsing with provenance chains
"""

from .app import create_app

__all__ = ["create_app"]
