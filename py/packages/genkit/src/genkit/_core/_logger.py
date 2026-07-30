# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Internal logger for genkit core. Not part of public API.

Under ``genkit start`` (``GENKIT_ENV=dev``), the Python process shares one
terminal with the CLI. This module keeps that stream readable: message-first
lines, a real log-level floor, and quiet third-party request spam.

For plain scripts, we still put an INFO floor on genkit's own structlog so
``logger.debug('generate response', ...)`` payload dumps stay off unless the
caller opts in with ``GENKIT_LOG=debug`` — without rewriting the process root
logger an app may already own.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import MutableMapping
from typing import Any

import structlog
from structlog.typing import FilteringBoundLogger

from genkit._core._environment import is_dev_environment

CONFIGURED = False
LOCK = threading.Lock()

# Libraries that log every HTTP hop. Under Dev UI those hops are health checks
# and span exports — useful in GENKIT_LOG=debug, noise otherwise.
QUIET_LOGGERS = (
    'httpx',
    'httpcore',
    'uvicorn.access',
    'uvicorn.error',
    'opentelemetry',
    'opentelemetry.sdk',
    'opentelemetry.instrumentation',
    # google-genai logs "AFC is enabled with max remote calls: N" at INFO on
    # every tool-using generate — fine for debug, noisy in the shared TTY.
    'google.genai',
    'google_genai',
)


def resolve_level() -> int:
    raw = os.environ.get('GENKIT_LOG', 'info').strip().lower()
    return {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'error': logging.ERROR,
    }.get(raw, logging.INFO)


def configure_logging(*, shared_tty: bool | None = None, force: bool = False) -> None:
    """Configure genkit console logging.

    Safe to call more than once. Default level is ``info``; override with
    ``GENKIT_LOG=debug|info|warn|error``.

    When ``shared_tty`` is true (``genkit start`` / ``GENKIT_ENV=dev``), take over
    the process console: message-first lines, quiet third-party request loggers,
    and ``basicConfig(force=True)``. When false, only ensure genkit's structlog
    has an INFO floor so debug payload dumps don't print — leave the caller's
    root logger alone.
    """
    global CONFIGURED
    with LOCK:
        if shared_tty is None:
            shared_tty = is_dev_environment()

        if CONFIGURED and not force:
            return

        level = resolve_level()

        if shared_tty:
            CONFIGURED = True
            logging.basicConfig(
                format='%(message)s',
                stream=sys.stderr,
                level=level,
                force=True,
            )

            for name in QUIET_LOGGERS:
                # Keep a little headroom when the user asked for debug.
                logging.getLogger(name).setLevel(logging.DEBUG if level <= logging.DEBUG else logging.WARNING)

            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    console_renderer,
                ],
                wrapper_class=structlog.make_filtering_bound_logger(level),
                logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
                cache_logger_on_first_use=True,
            )
            return

        # Plain scripts / production: don't stomp apps that already configured
        # structlog themselves.
        if structlog.is_configured() and not force:
            CONFIGURED = True
            return

        CONFIGURED = True
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            cache_logger_on_first_use=True,
        )


def console_renderer(
    _logger: object,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> str:
    """Render a short, message-first line for a human watching the terminal."""
    level = str(event_dict.pop('level', 'info')).lower()
    event = event_dict.pop('event', '')
    # Exception text already appended by format_exc_info when present.
    exc_info = event_dict.pop('exception', None)
    extras = ' '.join(f'{key}={value!r}' for key, value in event_dict.items() if value is not None)
    message = str(event)
    if extras:
        message = f'{message} ({extras})' if message else extras

    if level in ('warning', 'warn'):
        line = f'Warn: {message}'
    elif level == 'error':
        line = f'Error: {message}'
    elif level == 'debug':
        line = f'Debug: {message}'
    else:
        # info stays bare so readiness lines don't look like framework chatter
        line = message

    if exc_info:
        return f'{line}\n{exc_info}'
    return line


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog bound logger with a concrete return type for checkers."""
    return structlog.get_logger(name)


# Install the floor before other genkit modules cache loggers at import time.
# Shared-TTY takeover only when this process is the genkit start runtime.
configure_logging(shared_tty=is_dev_environment())
