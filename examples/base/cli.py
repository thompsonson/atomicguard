"""Click CLI utilities for AtomicGuard examples."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import click


def common_options[F: Callable[..., Any]](func: F) -> F:
    """
    Decorator adding common CLI options to a click command.

    Options added:
        --host: Ollama API URL
        --model: Model override
        --prompts: Path to prompts.json
        --workflow: Path to workflow.json
        --output: Path to save results JSON
        --artifact-dir: Directory for artifact storage
        --log-file: Path to log file
        -v/--verbose: Enable verbose logging
    """

    @click.option(
        "--host",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)",
    )
    @click.option(
        "--model",
        default=None,
        help="Override model specified in workflow.json",
    )
    @click.option(
        "--prompts",
        default=None,
        type=click.Path(exists=True),
        help="Path to prompts.json (default: ./prompts.json)",
    )
    @click.option(
        "--workflow",
        default=None,
        type=click.Path(exists=True),
        help="Path to workflow.json (default: ./workflow.json)",
    )
    @click.option(
        "--output",
        default=None,
        type=click.Path(),
        help="Path to save results JSON",
    )
    @click.option(
        "--artifact-dir",
        default=None,
        type=click.Path(),
        help="Directory for artifact storage",
    )
    @click.option(
        "--log-file",
        default=None,
        type=click.Path(),
        help="Path to log file",
    )
    @click.option(
        "-v",
        "--verbose",
        is_flag=True,
        help="Enable verbose (DEBUG) logging to console",
    )
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def add_options[F: Callable[..., Any]](func: F) -> F:
    """
    Decorator adding ADD-specific CLI options.

    Options added:
        --docs: Path to architecture documentation
        --workdir: Output directory for generated tests
        --rmax: Maximum retry attempts
        --min-gates: Minimum gates required
        --min-tests: Minimum tests required
    """

    @click.option(
        "--docs",
        default=None,
        type=click.Path(exists=True),
        help="Path to architecture documentation",
    )
    @click.option(
        "--workdir",
        default=None,
        type=click.Path(),
        help="Output directory for generated tests",
    )
    @click.option(
        "--rmax",
        default=3,
        type=int,
        help="Maximum retry attempts per action pair (default: 3)",
    )
    @click.option(
        "--min-gates",
        default=3,
        type=int,
        help="Minimum number of gates required (default: 3)",
    )
    @click.option(
        "--min-tests",
        default=3,
        type=int,
        help="Minimum number of tests required (default: 3)",
    )
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
