"""
Standalone Artifact Viewer for AtomicGuard workflows.

This example provides a simple Gradio-based UI for browsing artifacts
from AtomicGuard workflow executions without requiring workflow execution.
"""

from .app import create_viewer_app, main
from .artifact_browser import create_artifact_browser

__all__ = ["create_viewer_app", "main", "create_artifact_browser"]
