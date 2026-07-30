"""
Centralized logging configuration.

Design Decision:
    - Single logger factory ensures consistent format across all modules.
    - Each module calls get_logger(__name__) to get a named logger.
    - Named loggers allow per-module log filtering in production.
    - StreamHandler writes to stdout (12-factor app: log to stdout, not files).
    - In production, stdout can be captured by a log aggregator (e.g., Datadog).
"""

import logging
import sys

from app.core.config import get_settings


def configure_logging() -> None:
    """
    Configure the root logger for the entire application.

    This must be called ONCE at application startup (in main.py lifespan).
    All subsequent calls to get_logger() will inherit this configuration.
    """
    settings = get_settings()

    formatter = logging.Formatter(
        fmt=settings.log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    # Avoid adding duplicate handlers on hot reload (uvicorn --reload)
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Suppress noisy third-party loggers in production
    if settings.environment == "production":
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for the given module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing started")

    Args:
        name: The module name, typically passed as __name__.

    Returns:
        logging.Logger: A configured named logger instance.
    """
    return logging.getLogger(name)
