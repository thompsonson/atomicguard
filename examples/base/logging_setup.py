"""Logging configuration for AtomicGuard examples."""

from __future__ import annotations

import logging
import os


def setup_logging(
    logger_name: str,
    log_file: str | None = None,
    verbose: bool = False,
) -> logging.Logger:
    """
    Configure dual-handler logging (console + file).

    Args:
        logger_name: Name for the logger (e.g., "add_workflow", "tdd_workflow")
        log_file: Path to log file (None for no file logging)
        verbose: Enable DEBUG level on console (default INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
    logger.addHandler(console_handler)

    # File handler (if path provided)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    # Suppress noisy 3rd party loggers
    for noisy in ["httpx", "openai", "httpcore", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger
