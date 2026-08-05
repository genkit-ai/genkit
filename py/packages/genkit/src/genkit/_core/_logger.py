# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Internal logger for genkit core. Not part of public API."""

from __future__ import annotations

import logging
import os

import structlog
from structlog.typing import FilteringBoundLogger

from genkit._core._environment import is_dev_environment

# Libraries that log every HTTP request or poll. Under Dev UI, health checks
# and span exports generate noise unless GENKIT_LOG=debug is explicitly set.
QUIET_LOGGERS = (
    'httpx',
    'httpcore',
    'uvicorn.access',
    'uvicorn.error',
)


def resolve_level() -> int:
    """Resolve logging level from GENKIT_LOG environment variable."""
    raw = os.environ.get('GENKIT_LOG', 'info').strip().lower()
    return {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
        'fatal': logging.CRITICAL,
    }.get(raw, logging.INFO)


def configure_logging(*, shared_tty: bool | None = None) -> None:
    """Configure genkit console logging and mute noisy HTTP/health poll loggers.

    Safe to call more than once. Default level is ``info``; override with
    ``GENKIT_LOG=debug|info|warn|error``.
    """
    if shared_tty is None:
        shared_tty = is_dev_environment()

    if not shared_tty:
        return

    level = resolve_level()
    quiet_level = level if level == logging.DEBUG else max(level, logging.WARNING)

    for name in QUIET_LOGGERS:
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET:
            logger.setLevel(quiet_level)


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog bound logger with a concrete return type for checkers."""
    return structlog.get_logger(name)
