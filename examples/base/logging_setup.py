"""Logging configuration for AtomicGuard examples."""

from __future__ import annotations

import logging
import os


def setup_logging(
    logger_name: str,
    log_file: str | None = None,
    verbose: bool = False,
    child_loggers: list[str] | None = None,
) -> logging.Logger:
    """
    Configure dual-handler logging (console + file).

    Args:
        logger_name: Name for the logger (e.g., "add_workflow", "tdd_workflow")
        log_file: Path to log file (None for no file logging)
        verbose: Enable DEBUG level on console (default INFO)
        child_loggers: Additional loggers to configure with same handlers

    Returns:
        Configured logger instance
    """
    # Console handler (shared)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)-8s | %(name)s | %(message)s")
    )

    # File handler (shared, if path provided)
    file_handler = None
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    # Configure main logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    if file_handler:
        logger.addHandler(file_handler)

    # Configure child loggers (e.g., "add_workflow" when running "sdlc_workflow")
    for child_name in child_loggers or []:
        child_logger = logging.getLogger(child_name)
        child_logger.setLevel(logging.DEBUG)
        child_logger.addHandler(console_handler)
        if file_handler:
            child_logger.addHandler(file_handler)

    # Also configure examples.sdlc.generators.* loggers
    for module in ["examples.sdlc.generators.bdd", "examples.sdlc.generators.coder"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.DEBUG)
        module_logger.addHandler(console_handler)
        if file_handler:
            module_logger.addHandler(file_handler)

    # Suppress noisy 3rd party loggers
    for noisy in ["httpx", "openai", "httpcore", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger
