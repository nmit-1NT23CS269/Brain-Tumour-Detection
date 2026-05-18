"""Logging configuration helpers."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .paths import LOGS_DIR, ensure_project_dirs


def configure_logging(name: str = "neuroscan") -> logging.Logger:
    """Return a configured logger shared across the application."""
    ensure_project_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        LOGS_DIR / "neuroscan.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
