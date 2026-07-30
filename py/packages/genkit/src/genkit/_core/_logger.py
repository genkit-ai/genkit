# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Internal logger for genkit core. Not part of public API."""

from __future__ import annotations

import logging
import os
import threading

import structlog
from structlog.typing import FilteringBoundLogger

CONFIGURED = False
LOCK = threading.Lock()

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
    }.get(raw, logging.INFO)


def configure_logging(*, force: bool = False) -> None:
    """Configure genkit console logging and mute noisy HTTP/health poll loggers.

    Safe to call more than once. Default level is ``info``; override with
    ``GENKIT_LOG=debug|info|warn|error``.
    """
    global CONFIGURED
    with LOCK:
        if CONFIGURED and not force:
            return

        CONFIGURED = True
        level = resolve_level()

        for name in QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG if level <= logging.DEBUG else logging.WARNING)


# Configure logger levels on import
configure_logging()


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog bound logger with a concrete return type for checkers."""
    return structlog.get_logger(name)
