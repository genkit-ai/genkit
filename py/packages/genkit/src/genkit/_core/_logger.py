# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Internal logger for genkit core. Not part of public API."""

from __future__ import annotations

import logging
import os
from typing import cast

import structlog
from structlog.typing import BindableLogger, FilteringBoundLogger

from genkit._core._environment import is_dev_environment
from genkit._core._trace._log_exporter import enable_log_export

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

# Shared genkit start TTY. Uvicorn is our reflection health polls. httpx logs
# INFO on every hop (Gemini generateContent, the app's own clients). Collector
# POSTs use urllib, so they never hit this logger. GENKIT_LOG=debug turns these
# back on.
QUIET_LOGGERS = (
    'uvicorn.access',
    'uvicorn.error',
    'httpx',
    'httpcore',
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
    """Configure genkit console logging and mute reflection access logs on a shared TTY.

    Safe to call more than once. Default console level is ``info``; override
    with ``GENKIT_LOG=debug|info|warn|error``. In ``GENKIT_ENV=dev``, debug
    records also stream to the Dev UI when a telemetry URL is set.
    """
    configure_structlog_level()
    _maybe_enable_log_export()

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


def _maybe_enable_log_export() -> None:
    """Turn on Dev UI log streaming when a telemetry URL is already known."""
    if not is_dev_environment():
        return
    url = os.environ.get('GENKIT_TELEMETRY_SERVER', '').strip()
    if not url:
        return
    enable_log_export(url=url)


_EMIT_LEVELS: dict[str, int] = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'warn': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
    'exception': logging.ERROR,
    'msg': logging.INFO,
}


class ExportTee:
    """Forwards every record to the Dev UI sink, then to the console logger.

    The console still honours ``GENKIT_LOG``. The export sink is debug and
    above, so a quiet terminal can still fill the trace viewer's log panel.
    """

    def __init__(self, bound: BindableLogger, attrs: dict[str, object] | None = None) -> None:
        self._bound = bound
        self._attrs = attrs or {}

    def bind(self, **new_values: object) -> ExportTee:
        return ExportTee(self._bound.bind(**new_values), attrs={**self._attrs, **new_values})

    def unbind(self, *keys: str) -> ExportTee:
        kept = {k: v for k, v in self._attrs.items() if k not in keys}
        return ExportTee(self._bound.unbind(*keys), attrs=kept)

    def new(self, **new_values: object) -> ExportTee:
        return ExportTee(self._bound.new(**new_values), attrs=dict(new_values))

    def is_enabled_for(self, level: int) -> bool:
        from genkit._core._trace._log_exporter import log_export_is_enabled

        if level >= logging.DEBUG and log_export_is_enabled():
            return True
        for attr in ('is_enabled_for', 'isEnabledFor'):
            check = getattr(self._bound, attr, None)
            if not callable(check):
                continue
            try:
                return bool(check(level))
            except Exception:
                return True
        return True

    def _emit(self, level: int, event: str, args: tuple[object, ...], kw: dict[str, object]) -> None:
        from genkit._core._trace._log_exporter import emit_log

        emit_log(level=level, event=_interpolate_event(event, args), attrs={**self._attrs, **kw})

    def __getattr__(self, name: str) -> object:
        target = getattr(self._bound, name)
        if name == 'log' and callable(target):

            def emit_log(level: int, event: str, *args: object, **kw: object) -> object:
                self._emit(level, event, args, kw)
                return target(level, event, *args, **kw)

            return emit_log
        if name not in _EMIT_LEVELS or not callable(target):
            return target
        emit_level = _EMIT_LEVELS[name]

        def emit_then(event: str, *args: object, **kw: object) -> object:
            self._emit(emit_level, event, args, kw)
            return target(event, *args, **kw)

        return emit_then


def _interpolate_event(event: str, args: tuple[object, ...]) -> str:
    """Fill printf-style ``%s`` so the Dev UI panel matches the console line."""
    if not args:
        return event
    try:
        return event % args
    except (TypeError, ValueError):
        return event


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog bound logger that also feeds the Dev UI log sink."""
    return cast(FilteringBoundLogger, ExportTee(structlog.get_logger(name)))


def is_debug_enabled(logger: FilteringBoundLogger) -> bool:
    """Report whether ``logger`` emits DEBUG events.

    Args:
        logger: Logger to inspect.

    Returns:
        ``True`` when DEBUG events are emitted, including when the Dev UI
        log sink is on (console may still be quieter) or when ``logger``
        exposes no usable level check.
    """
    from genkit._core._trace._log_exporter import log_export_is_enabled

    if log_export_is_enabled():
        return True
    for attr in ('is_enabled_for', 'isEnabledFor'):
        check = getattr(logger, attr, None)
        if not callable(check):
            continue
        try:
            return bool(check(logging.DEBUG))
        except Exception:
            return True
    return True
