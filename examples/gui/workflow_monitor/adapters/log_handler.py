"""Custom logging handler that emits events for the GUI."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..state.events import AnyEvent, LogEvent, WorkflowEvent


class EventLogHandler(logging.Handler):
    """
    Logging handler that converts log records to GUI events.

    This handler intercepts log messages and emits them as LogEvent
    objects for display in the GUI log viewer.
    """

    def __init__(
        self,
        emit_callback: Callable[[AnyEvent], None],
        min_level: int = logging.DEBUG,
    ) -> None:
        """
        Initialize the event log handler.

        Args:
            emit_callback: Function to call with LogEvent objects
            min_level: Minimum log level to capture
        """
        super().__init__(level=min_level)
        self._emit = emit_callback
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Convert log record to event and emit."""
        try:
            message = self.format(record)

            event = LogEvent(
                timestamp=WorkflowEvent.now(),
                event_type="log",
                level=record.levelname,
                message=message,
                logger_name=record.name,
            )

            self._emit(event)

        except RecursionError:
            # Prevent recursion if logging is called during emit
            raise
        except Exception:
            # Don't let logging errors break the application
            self.handleError(record)


def setup_gui_logging(
    emit_callback: Callable[[AnyEvent], None],
    level: int = logging.DEBUG,
) -> tuple[logging.Logger, EventLogHandler]:
    """
    Set up logging with GUI event emission.

    Creates a handler attached to the ROOT logger to capture ALL log messages
    from atomicguard and its submodules for GUI display.

    Args:
        emit_callback: Function to call with events
        level: Logging level

    Returns:
        Tuple of (atomicguard logger, handler) for cleanup
    """
    # Create event handler
    handler = EventLogHandler(emit_callback, min_level=level)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))

    # Attach to ROOT logger to capture all logs from all modules
    root_logger = logging.getLogger()
    root_logger.setLevel(level)  # CRITICAL: Root logger defaults to WARNING
    root_logger.addHandler(handler)

    # Ensure atomicguard and examples loggers propagate and have appropriate level
    for name in ["atomicguard", "examples.gui"]:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = True

    # Suppress noisy third-party loggers
    for noisy in [
        "httpx",
        "openai",
        "httpcore",
        "urllib3",
        "gradio",
        "hpack",
        "httpx._client",
        "matplotlib",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Return the atomicguard logger for use by the workflow
    return logging.getLogger("atomicguard"), handler


def cleanup_gui_logging(
    _logger: logging.Logger,
    handler: EventLogHandler,
) -> None:
    """
    Remove the GUI logging handler.

    Call this when closing the GUI to clean up logging.

    Args:
        _logger: Logger (unused, kept for backwards compatibility)
        handler: Handler to remove from root logger
    """
    # Remove from root logger
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
