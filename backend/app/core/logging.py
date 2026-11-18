"""Structured logging configuration.

Provides a single :func:`configure_logging` entry point used by ``main.py``.
Supports a JSON formatter (for production / log aggregation) and a readable
console formatter (for local development), selected via ``LOG_FORMAT``.
"""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger

from app.core.config import settings

_CONFIGURED = False


class _RequestIdFilter(logging.Filter):
    """Ensure every record carries ``request_id`` (default "-")."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def configure_logging() -> None:
    """Configure the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)

    if settings.log_format.lower() == "json":
        formatter: logging.Formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | req=%(request_id)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Uvicorn access logs duplicate our request-logging middleware.
    logging.getLogger("uvicorn.access").propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
