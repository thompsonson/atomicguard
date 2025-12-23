"""
Base module for AtomicGuard examples.

Provides reusable utilities for building workflow runners:
- Configuration loading (load_prompts, load_workflow_config)
- Logging setup (setup_logging)
- CLI utilities (common_options, add_options)
- Console output (print_header, print_error, print_success, etc.)
- Guard building (build_guard, register_guard)
- Workflow execution (BaseWorkflowRunner, StandardWorkflowRunner)
"""

from .cli import add_options, common_options
from .config import (
    load_prompts,
    load_workflow_config,
    normalize_base_url,
    normalize_model_name,
)
from .console import (
    console,
    error_console,
    print_error,
    print_failure,
    print_header,
    print_provenance,
    print_steps,
    print_success,
    print_workflow_info,
)
from .exceptions import ConfigurationError
from .guards import GUARD_REGISTRY, build_guard, register_guard
from .logging_setup import setup_logging
from .workflow import (
    BaseWorkflowRunner,
    StandardWorkflowRunner,
    display_workflow_result,
    save_workflow_results,
)

__all__ = [
    # Exceptions
    "ConfigurationError",
    # Config
    "load_prompts",
    "load_workflow_config",
    "normalize_base_url",
    "normalize_model_name",
    # Logging
    "setup_logging",
    # CLI
    "common_options",
    "add_options",
    # Console
    "console",
    "error_console",
    "print_header",
    "print_error",
    "print_success",
    "print_failure",
    "print_workflow_info",
    "print_steps",
    "print_provenance",
    # Guards
    "GUARD_REGISTRY",
    "build_guard",
    "register_guard",
    # Workflow
    "BaseWorkflowRunner",
    "StandardWorkflowRunner",
    "save_workflow_results",
    "display_workflow_result",
]
