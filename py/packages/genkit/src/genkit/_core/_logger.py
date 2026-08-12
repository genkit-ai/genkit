# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Internal logger for genkit core. Not part of public API."""

from __future__ import annotations

import logging
import os

import structlog
from structlog.typing import FilteringBoundLogger

from genkit._core._environment import is_dev_environment

# Environment variable name
GENKIT_LOG = 'GENKIT_LOG'

DEFAULT_LOG_LEVEL = logging.INFO

_LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
    'fatal': logging.CRITICAL,
}

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
    raw = os.environ.get(GENKIT_LOG, 'info').strip().lower()
    return _LOG_LEVELS.get(raw, DEFAULT_LOG_LEVEL)


def unrecognized_level() -> str | None:
    """Return the raw ``GENKIT_LOG`` value when it names no known level.

    Returns:
        The value as set, or ``None`` when it is absent, empty, or recognized.
    """
    raw = os.environ.get(GENKIT_LOG, '')
    if not raw.strip() or raw.strip().lower() in _LOG_LEVELS:
        return None
    return raw


def _app_chose_level() -> bool:
    """Report whether the application configured structlog with a level of its own."""
    if not structlog.is_configured():
        return False
    unfiltered = structlog.make_filtering_bound_logger(logging.NOTSET)
    return structlog.get_config().get('wrapper_class') is not unfiltered


def configure_structlog_level() -> bool:
    """Apply ``GENKIT_LOG`` to structlog unless the application chose a level.

    structlog is unconfigured by default, and its defaults emit DEBUG to stdout
    through ``PrintLoggerFactory``, bypassing :mod:`logging` entirely. Muting the
    stdlib loggers therefore cannot quiet genkit's own events.

    Returns:
        ``True`` when the level was applied, ``False`` when an
        application-configured level was left untouched.
    """
    if _app_chose_level():
        return False
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(resolve_level()))
    ignored = unrecognized_level()
    if ignored is not None:
        get_logger(__name__).warning(
            'ignoring unrecognized log level',
            variable=GENKIT_LOG,
            value=ignored,
            using=logging.getLevelName(resolve_level()),
        )
    return True


def configure_logging(*, shared_tty: bool | None = None) -> None:
    """Configure genkit console logging and mute noisy HTTP/health poll loggers.

    Safe to call more than once. Default level is ``info``; override with
    ``GENKIT_LOG=debug|info|warn|error``.
    """
    configure_structlog_level()

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


def is_debug_enabled(logger: FilteringBoundLogger) -> bool:
    """Report whether ``logger`` emits DEBUG events.

    Args:
        logger: Logger to inspect.

    Returns:
        ``True`` when DEBUG events are emitted, including when ``logger``
        exposes no usable level check.
    """
    for attr in ('is_enabled_for', 'isEnabledFor'):
        check = getattr(logger, attr, None)
        if not callable(check):
            continue
        try:
            return bool(check(logging.DEBUG))
        except Exception:
            return True
    return True
