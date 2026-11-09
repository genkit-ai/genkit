# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Genkit logging. ``GenkitLogger`` and ``get_logger`` are public via ``genkit``; the rest is internal."""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Awaitable, Callable
from typing import Protocol, cast
from weakref import WeakKeyDictionary

import structlog
from structlog.typing import BindableLogger

from genkit._core._environment import is_dev_environment
from genkit._core._telemetry._log_exporter import LOG_EXPORT_THREAD_NAME, enable_log_export

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

# The reflection server runs uvicorn on this thread. uvicorn hard-codes its
# logger names, so the reflection server's startup and health-poll access lines
# share `uvicorn.error` / `uvicorn.access` with the app's own uvicorn server.
REFLECTION_THREAD_NAME = 'genkit-reflection-server'

# Loggers the reflection server writes to. Genkit only adds a filter that drops
# records from REFLECTION_THREAD_NAME; levels, handlers and propagation stay as
# the app configured them.
REFLECTION_LOGGERS = (
    'uvicorn.access',
    'uvicorn.error',
)

# Shared genkit start TTY. httpx and httpcore log INFO on every hop (model
# calls, the app's own clients, and Dev UI log export batches on
# LOG_EXPORT_THREAD_NAME). GENKIT_LOG=debug turns these back on.
QUIET_LOGGERS = (
    'httpx',
    'httpcore',
)

_genkit_quieted_levels: WeakKeyDictionary[logging.Logger, int] = WeakKeyDictionary()


def resolve_level() -> int:
    """Resolve logging level from GENKIT_LOG environment variable."""
    raw = os.environ.get(GENKIT_LOG, 'info').strip().lower()
    return _LOG_LEVELS.get(raw, DEFAULT_LOG_LEVEL)


def _quiet_level() -> int:
    """Return the quiet threshold: WARNING (or stricter GENKIT_LOG), unless debug."""
    level = resolve_level()
    return level if level == logging.DEBUG else max(level, logging.WARNING)


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
    if structlog.is_configured():
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(resolve_level()))
    else:
        # structlog's default factory prints to stdout, which corrupts
        # `python cli.py | jq .`. Logs go to stderr; an app that already
        # picked a logger factory keeps it.
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(resolve_level()),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        )
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
    """Configure genkit console logging and mute client HTTP hops on a shared TTY.

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

    quiet_level = _quiet_level()

    for name in QUIET_LOGGERS:
        logger = logging.getLogger(name)
        if logger.level == logging.NOTSET or _genkit_quieted_levels.get(logger) == logger.level:
            logger.setLevel(quiet_level)
            _genkit_quieted_levels[logger] = quiet_level
        _attach_thread_filter(logger, LOG_EXPORT_THREAD_NAME)


class ReflectionLogFilter(logging.Filter):
    """Drop internal Genkit worker records below the quiet level by thread name.

    Records from any other thread pass untouched, so the app's uvicorn startup,
    access lines, and HTTP client logs keep the level and handlers the app
    configured. Checks ``threading.current_thread().name`` rather than
    ``record.threadName`` so ``logging.logThreads = False`` cannot bypass it.
    The quiet level is WARNING, or ``GENKIT_LOG`` when it is stricter.
    ``GENKIT_LOG=debug`` lets every record through.
    """

    def __init__(self, thread_names: frozenset[str] = frozenset({REFLECTION_THREAD_NAME})) -> None:
        super().__init__()
        self.thread_names = thread_names

    def filter(self, record: logging.LogRecord) -> bool:
        """Return ``False`` for quiet records emitted on ``self.thread_names``."""
        if threading.current_thread().name not in self.thread_names:
            return True
        return record.levelno >= _quiet_level()


def _attach_thread_filter(logger: logging.Logger, thread_name: str) -> None:
    """Attach a :class:`ReflectionLogFilter` for ``thread_name`` if absent."""
    for existing in logger.filters:
        if isinstance(existing, ReflectionLogFilter) and thread_name in existing.thread_names:
            return
    logger.addFilter(ReflectionLogFilter(frozenset({thread_name})))


def has_app_uvicorn_logging() -> bool:
    """Report whether the host application configured uvicorn loggers."""
    for name in ('uvicorn', 'uvicorn.error', 'uvicorn.access'):
        logger = logging.getLogger(name)
        if logger.handlers or logger.level != logging.NOTSET:
            return True
    return False


def quiet_reflection_logs() -> None:
    """Attach :class:`ReflectionLogFilter` to the reflection server's loggers.

    Safe to call more than once. Levels, handlers and propagation on those
    loggers are left alone.
    """
    for name in REFLECTION_LOGGERS:
        _attach_thread_filter(logging.getLogger(name), REFLECTION_THREAD_NAME)


def quiet_log_export_logs() -> None:
    """Attach :class:`ReflectionLogFilter` for the log exporter to ``QUIET_LOGGERS``."""
    for name in QUIET_LOGGERS:
        _attach_thread_filter(logging.getLogger(name), LOG_EXPORT_THREAD_NAME)


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
    'fatal': logging.CRITICAL,
    'exception': logging.ERROR,
    'msg': logging.INFO,
}

# structlog's async twins: ``ainfo``, ``adebug``, ``afatal``, ...
_ASYNC_EMIT_LEVELS: dict[str, int] = {f'a{name}': level for name, level in _EMIT_LEVELS.items()}


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

    def try_unbind(self, *keys: str) -> ExportTee:
        kept = {k: v for k, v in self._attrs.items() if k not in keys}
        return ExportTee(self._bound.try_unbind(*keys), attrs=kept)

    def new(self, **new_values: object) -> ExportTee:
        return ExportTee(self._bound.new(**new_values), attrs=dict(new_values))

    def is_enabled_for(self, level: int) -> bool:
        from genkit._core._telemetry._log_exporter import log_export_is_enabled

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
        from genkit._core._telemetry._log_exporter import emit_log

        emit_log(level=level, event=_interpolate_event(event, args), attrs={**self._attrs, **kw})

    def __getattr__(self, name: str) -> object:
        target = getattr(self._bound, name)
        if name == 'log' and callable(target):

            def emit_log(level: int, event: str, *args: object, **kw: object) -> object:
                self._emit(level, event, args, kw)
                return target(level, event, *args, **kw)

            return emit_log
        if name == 'alog' and callable(target):
            alog = cast(Callable[..., Awaitable[object]], target)

            async def aemit_log(level: int, event: str, *args: object, **kw: object) -> object:
                self._emit(level, event, args, kw)
                return await alog(level, event, *args, **kw)

            return aemit_log
        if name in _ASYNC_EMIT_LEVELS and callable(target):
            aemit_level = _ASYNC_EMIT_LEVELS[name]
            amethod = cast(Callable[..., Awaitable[object]], target)

            async def aemit_then(event: str, *args: object, **kw: object) -> object:
                self._emit(aemit_level, event, args, kw)
                return await amethod(event, *args, **kw)

            return aemit_then
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


class GenkitLogger(Protocol):
    """The calls ``get_logger`` promises. Nothing here names the logging backend.

    Each method takes an event string, optional printf-style ``args``, and
    keyword ``fields`` that are recorded as structured attributes.
    """

    def debug(self, event: str, *args: object, **fields: object) -> None:
        """Log at DEBUG."""
        ...

    def info(self, event: str, *args: object, **fields: object) -> None:
        """Log at INFO."""
        ...

    def warning(self, event: str, *args: object, **fields: object) -> None:
        """Log at WARNING."""
        ...

    def error(self, event: str, *args: object, **fields: object) -> None:
        """Log at ERROR."""
        ...

    def exception(self, event: str, *args: object, **fields: object) -> None:
        """Log at ERROR with the active exception's traceback."""
        ...


def get_logger(name: str | None = None) -> GenkitLogger:
    """Return a logger whose lines also reach the Dev UI trace panel.

    The logger has ``debug``, ``info``, ``warning``, ``error`` and
    ``exception``. Each takes an event string and keyword fields. Under
    ``genkit start``, each line is also exported to the Dev UI with the active
    trace and span IDs.

    Example:
        ```python
        from genkit import Genkit, get_logger

        # 1. Create a module logger
        ai = Genkit()
        logger = get_logger(__name__)


        # 2. Log inside a flow; lines attach to the flow's trace
        @ai.flow()
        async def lookup_menu(restaurant_id: str) -> int:
            logger.info('menu lookup', restaurant_id=restaurant_id, items=12)
            # => console: menu lookup  restaurant_id=r-42 items=12
            # => Dev UI:  same line, attached to the lookup_menu trace

            logger.debug('cache miss', restaurant_id=restaurant_id)
            # => console: nothing unless GENKIT_LOG=debug
            # => Dev UI:  cache miss, still exported under `genkit start`
            return 12
        ```
    """
    return cast(GenkitLogger, ExportTee(structlog.get_logger(name)))


def is_debug_enabled(logger: GenkitLogger) -> bool:
    """Report whether ``logger`` emits DEBUG events.

    Args:
        logger: Logger to inspect.

    Returns:
        ``True`` when DEBUG events are emitted, including when the Dev UI
        log sink is on (console may still be quieter) or when ``logger``
        exposes no usable level check.
    """
    from genkit._core._telemetry._log_exporter import log_export_is_enabled

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
