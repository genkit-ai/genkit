# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the logger module."""

import logging
import os
import threading
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest
import structlog
from structlog.processors import JSONRenderer
from structlog.testing import capture_logs

from genkit._core._logger import (
    GENKIT_LOG,
    LOG_EXPORT_THREAD_NAME,
    QUIET_LOGGERS,
    REFLECTION_LOGGERS,
    REFLECTION_THREAD_NAME,
    ReflectionLogFilter,
    _genkit_quieted_levels,
    configure_logging,
    configure_structlog_level,
    get_logger,
    has_app_uvicorn_logging,
    is_debug_enabled,
    quiet_reflection_logs,
    resolve_level,
    unrecognized_level,
)


def test_resolve_level() -> None:
    """Test resolve_level with different GENKIT_LOG values."""
    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'debug'}):
        assert resolve_level() == logging.DEBUG

    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'info'}):
        assert resolve_level() == logging.INFO

    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'warn'}):
        assert resolve_level() == logging.WARNING

    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'warning'}):
        assert resolve_level() == logging.WARNING

    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'error'}):
        assert resolve_level() == logging.ERROR

    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'critical'}):
        assert resolve_level() == logging.CRITICAL

    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'fatal'}):
        assert resolve_level() == logging.CRITICAL

    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'invalid'}):
        assert resolve_level() == logging.INFO


def _record(level: int) -> logging.LogRecord:
    return logging.LogRecord('uvicorn.access', level, __file__, 1, 'GET /api/__health', None, None)


def _filter_on_thread(log_filter: logging.Filter, record: logging.LogRecord, thread_name: str) -> bool:
    result: list[bool] = []
    thread = threading.Thread(target=lambda: result.append(log_filter.filter(record)), name=thread_name)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    return result[0]


@pytest.mark.parametrize(
    ('genkit_log', 'level', 'passes'),
    [
        ('info', logging.INFO, False),
        ('info', logging.WARNING, True),
        ('debug', logging.INFO, True),
        ('error', logging.WARNING, False),
        ('error', logging.ERROR, True),
    ],
)
def test_reflection_filter_quiets_only_the_reflection_thread(genkit_log: str, level: int, passes: bool) -> None:
    """Reflection records follow max(WARNING, GENKIT_LOG); debug lets them all through."""
    log_filter = ReflectionLogFilter()
    with mock.patch.dict(os.environ, {GENKIT_LOG: genkit_log}):
        assert _filter_on_thread(log_filter, _record(level), REFLECTION_THREAD_NAME) is passes
        assert log_filter.filter(_record(level)) is True


def test_reflection_filter_works_when_log_threads_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling logging.logThreads leaves record.threadName None without bypassing the filter."""
    monkeypatch.setattr(logging, 'logThreads', False)
    log_filter = ReflectionLogFilter()
    record = _record(logging.INFO)
    assert record.threadName is None

    with mock.patch.dict(os.environ, {GENKIT_LOG: 'info'}):
        assert _filter_on_thread(log_filter, record, REFLECTION_THREAD_NAME) is False
        assert log_filter.filter(record) is True


def test_quiet_reflection_logs_adds_one_filter_and_keeps_levels() -> None:
    """Repeat calls attach the filter once and never touch levels or handlers."""
    loggers = [logging.getLogger(name) for name in REFLECTION_LOGGERS]
    saved = [(lg, lg.level, list(lg.handlers), list(lg.filters)) for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.INFO)
        quiet_reflection_logs()
        quiet_reflection_logs()
        for lg, _, handlers, _ in saved:
            assert sum(isinstance(f, ReflectionLogFilter) for f in lg.filters) == 1
            assert lg.level == logging.INFO
            assert lg.handlers == handlers
    finally:
        for lg, level, handlers, filters in saved:
            lg.setLevel(level)
            lg.handlers = handlers
            lg.filters = filters


def test_quiet_loggers_are_httpx_and_httpcore() -> None:
    """Dev UI mutes client HTTP hops on a shared TTY, not provider SDKs."""
    assert QUIET_LOGGERS == (
        'httpx',
        'httpcore',
    )
    assert 'uvicorn.access' not in QUIET_LOGGERS
    assert 'uvicorn.error' not in QUIET_LOGGERS


def test_configure_logging_mutes_quiet_loggers() -> None:
    """Test that configure_logging sets QUIET_LOGGERS to WARNING on a shared TTY."""
    with (
        mock.patch.dict(os.environ, {'GENKIT_LOG': 'info'}),
        mock.patch('logging.getLogger') as mock_get_logger,
    ):
        mock_get_logger.return_value.level = logging.NOTSET
        configure_logging(shared_tty=True)
        for name in QUIET_LOGGERS:
            mock_get_logger.assert_any_call(name)
        expected_calls = [mock.call(logging.WARNING)] * len(QUIET_LOGGERS)
        mock_get_logger.return_value.setLevel.assert_has_calls(expected_calls, any_order=True)


def test_configure_logging_allows_debug() -> None:
    """Test that GENKIT_LOG=debug sets QUIET_LOGGERS to DEBUG."""
    with (
        mock.patch.dict(os.environ, {'GENKIT_LOG': 'debug'}),
        mock.patch('logging.getLogger') as mock_get_logger,
    ):
        mock_get_logger.return_value.level = logging.NOTSET
        configure_logging(shared_tty=True)
        for name in QUIET_LOGGERS:
            mock_get_logger.assert_any_call(name)
        expected_calls = [mock.call(logging.DEBUG)] * len(QUIET_LOGGERS)
        mock_get_logger.return_value.setLevel.assert_has_calls(expected_calls, any_order=True)


def test_configure_logging_leaves_loggers_alone_in_prod() -> None:
    """Test that configure_logging does not alter logger levels in non-dev env."""
    with mock.patch('logging.getLogger') as mock_get_logger:
        configure_logging(shared_tty=False)
        mock_get_logger.assert_not_called()


def test_configure_logging_respects_user_configured_levels() -> None:
    """Test that configure_logging does not overwrite explicitly set logger levels."""
    with (
        mock.patch.dict(os.environ, {'GENKIT_LOG': 'info'}),
        mock.patch('logging.getLogger') as mock_get_logger,
    ):
        mock_get_logger.return_value.level = logging.DEBUG
        configure_logging(shared_tty=True)
        mock_get_logger.return_value.setLevel.assert_not_called()


def test_configure_logging_filters_log_export_thread_even_when_app_enables_httpx_info() -> None:
    """Log-export httpx INFO hops are dropped while the app's own httpx INFO hops pass."""
    loggers = [logging.getLogger(name) for name in QUIET_LOGGERS]
    saved = [(lg, lg.level, list(lg.handlers), list(lg.filters), lg.propagate) for lg in loggers]
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    httpx_logger = logging.getLogger('httpx')
    try:
        for lg in loggers:
            _genkit_quieted_levels.pop(lg, None)
            lg.filters = []
        httpx_logger.setLevel(logging.INFO)
        httpx_logger.handlers = [_Capture()]
        httpx_logger.propagate = False

        with mock.patch.dict(os.environ, {GENKIT_LOG: 'info'}):
            configure_logging(shared_tty=True)
            configure_logging(shared_tty=True)

            assert httpx_logger.level == logging.INFO
            assert sum(isinstance(f, ReflectionLogFilter) for f in httpx_logger.filters) == 1

            export_thread = threading.Thread(
                target=lambda: httpx_logger.info('POST http://127.0.0.1:4033/api/otlp "HTTP/1.1 200 OK"'),
                name=LOG_EXPORT_THREAD_NAME,
            )
            export_thread.start()
            export_thread.join(timeout=5)
            httpx_logger.info('GET https://menu.example.com/dishes "HTTP/1.1 200 OK"')

        assert [r.getMessage() for r in records] == ['GET https://menu.example.com/dishes "HTTP/1.1 200 OK"']

        with mock.patch.dict(os.environ, {GENKIT_LOG: 'debug'}):
            debug_thread = threading.Thread(
                target=lambda: httpx_logger.info('POST http://127.0.0.1:4033/api/otlp "HTTP/1.1 200 OK"'),
                name=LOG_EXPORT_THREAD_NAME,
            )
            debug_thread.start()
            debug_thread.join(timeout=5)

        assert 'POST http://127.0.0.1:4033/api/otlp "HTTP/1.1 200 OK"' in [r.getMessage() for r in records]
    finally:
        for lg, level, handlers, filters, propagate in saved:
            _genkit_quieted_levels.pop(lg, None)
            lg.setLevel(level)
            lg.handlers = handlers
            lg.filters = filters
            lg.propagate = propagate


def test_configure_logging_updates_quiet_level_across_calls() -> None:
    """Repeated configure_logging calls update Genkit-managed quiet levels without duplicating filters."""
    loggers = [logging.getLogger(name) for name in QUIET_LOGGERS]
    saved = [(lg, lg.level, list(lg.handlers), list(lg.filters)) for lg in loggers]
    try:
        for lg in loggers:
            _genkit_quieted_levels.pop(lg, None)
            lg.setLevel(logging.NOTSET)
            lg.filters = []

        with mock.patch.dict(os.environ, {GENKIT_LOG: 'info'}):
            configure_logging(shared_tty=True)
        for lg in loggers:
            assert lg.level == logging.WARNING

        with mock.patch.dict(os.environ, {GENKIT_LOG: 'debug'}):
            configure_logging(shared_tty=True)
        for lg in loggers:
            assert lg.level == logging.DEBUG
            assert sum(isinstance(f, ReflectionLogFilter) for f in lg.filters) == 1
    finally:
        for lg, level, handlers, filters in saved:
            _genkit_quieted_levels.pop(lg, None)
            lg.setLevel(level)
            lg.handlers = handlers
            lg.filters = filters


def test_has_app_uvicorn_logging() -> None:
    """Reports True when uvicorn loggers have handlers or non-NOTSET level."""
    loggers = [logging.getLogger(n) for n in ('uvicorn', 'uvicorn.error', 'uvicorn.access')]
    saved = [(lg, lg.level, list(lg.handlers)) for lg in loggers]
    try:
        for lg in loggers:
            lg.level = logging.NOTSET
            lg.handlers = []
        assert has_app_uvicorn_logging() is False

        logging.getLogger('uvicorn.error').setLevel(logging.INFO)
        assert has_app_uvicorn_logging() is True
        logging.getLogger('uvicorn.error').setLevel(logging.NOTSET)

        dummy_handler = logging.StreamHandler()
        logging.getLogger('uvicorn.access').handlers = [dummy_handler]
        assert has_app_uvicorn_logging() is True
    finally:
        for lg, level, handlers in saved:
            lg.level = level
            lg.handlers = handlers


@pytest.fixture
def _restore_structlog() -> Iterator[None]:
    """Restore structlog's global configuration around a test."""
    saved = structlog.get_config().copy()
    was_configured = structlog.is_configured()
    yield
    structlog.reset_defaults()
    if was_configured:
        structlog.configure(**saved)


@pytest.mark.usefixtures('_restore_structlog')
def test_unrecognized_level() -> None:
    """Only a set, non-empty, unrecognized GENKIT_LOG value is reported, verbatim."""
    with mock.patch.dict(os.environ, clear=True):
        assert unrecognized_level() is None

    for value in ['', '   ', 'debug', '  WARN  ']:
        with mock.patch.dict(os.environ, {GENKIT_LOG: value}):
            assert unrecognized_level() is None

    with mock.patch.dict(os.environ, {GENKIT_LOG: 'dbug'}):
        assert unrecognized_level() == 'dbug'


@pytest.mark.usefixtures('_restore_structlog')
def test_configure_structlog_level_drops_debug_events() -> None:
    """The applied level filters DEBUG events out while keeping INFO and above."""
    structlog.reset_defaults()
    with mock.patch.dict(os.environ, clear=True):
        assert configure_structlog_level() is True

        logger = get_logger('test')
        assert is_debug_enabled(logger) is False
        with capture_logs() as entries:
            logger.debug('noisy', response={'blob': 'x' * 1000})
            logger.info('kept')
        assert [entry['event'] for entry in entries] == ['kept']


@pytest.mark.usefixtures('_restore_structlog')
def test_configure_structlog_level_honours_debug_opt_in() -> None:
    """GENKIT_LOG=debug turns the genkit debug logs back on."""
    structlog.reset_defaults()
    with mock.patch.dict(os.environ, {GENKIT_LOG: 'debug'}):
        assert configure_structlog_level() is True

        logger = get_logger('test')
        assert is_debug_enabled(logger) is True
        with capture_logs() as entries:
            logger.debug('noisy')
        assert 'noisy' in [entry['event'] for entry in entries]


@pytest.mark.usefixtures('_restore_structlog')
def test_configure_structlog_level_applies_when_app_configured_processors_only() -> None:
    """An app that configured only processors has not chosen a level."""
    structlog.reset_defaults()
    structlog.configure(processors=[JSONRenderer()])
    with mock.patch.dict(os.environ, clear=True):
        assert configure_structlog_level() is True
        assert is_debug_enabled(get_logger('test')) is False


@pytest.mark.usefixtures('_restore_structlog')
def test_configure_structlog_level_leaves_an_app_chosen_level_alone() -> None:
    """An app that configured its own wrapper_class keeps its level."""
    structlog.reset_defaults()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG))
    with mock.patch.dict(os.environ, clear=True):
        assert configure_structlog_level() is False
        assert is_debug_enabled(get_logger('test')) is True


@pytest.mark.usefixtures('_restore_structlog')
def test_configure_structlog_level_is_idempotent() -> None:
    """A second call reports that nothing was applied."""
    structlog.reset_defaults()
    with mock.patch.dict(os.environ, clear=True):
        assert configure_structlog_level() is True
        assert configure_structlog_level() is False


@pytest.mark.usefixtures('_restore_structlog')
def test_unrecognized_level_warns_and_falls_back() -> None:
    """A misspelled GENKIT_LOG logs a warning instead of failing silently."""
    structlog.reset_defaults()
    with mock.patch.dict(os.environ, {GENKIT_LOG: 'dbug'}), capture_logs() as entries:
        assert configure_structlog_level() is True

    warnings = [e for e in entries if e['event'] == 'ignoring unrecognized log level']
    assert len(warnings) == 1
    assert warnings[0]['value'] == 'dbug'
    assert warnings[0]['using'] == 'INFO'


@pytest.mark.usefixtures('_restore_structlog')
def test_import_time_loggers_pick_up_late_configuration() -> None:
    """Loggers bound at import time honour a level configured afterwards."""
    structlog.reset_defaults()
    logger = get_logger('bound.early')
    assert is_debug_enabled(logger) is True

    with mock.patch.dict(os.environ, clear=True):
        configure_structlog_level()

    assert is_debug_enabled(logger) is False


@pytest.mark.usefixtures('_restore_structlog')
def test_level_survives_a_later_processors_only_configure() -> None:
    """A plugin appending processors after genkit set the level keeps the filter."""
    structlog.reset_defaults()
    with mock.patch.dict(os.environ, clear=True):
        configure_structlog_level()

    processors = list(structlog.get_config()['processors'])
    structlog.configure(processors=[*processors, JSONRenderer()])

    assert is_debug_enabled(get_logger('test')) is False


@pytest.mark.usefixtures('_restore_structlog')
def test_configure_logging_sets_the_structlog_level_outside_dev() -> None:
    """The structlog level applies in every environment."""
    structlog.reset_defaults()
    with mock.patch.dict(os.environ, clear=True):
        configure_logging()

    assert is_debug_enabled(get_logger('test')) is False


@pytest.mark.parametrize(
    ('wrapper', 'expected'),
    [
        (None, True),
        (structlog.make_filtering_bound_logger(logging.NOTSET), True),
        (structlog.make_filtering_bound_logger(logging.DEBUG), True),
        (structlog.make_filtering_bound_logger(logging.INFO), False),
        (structlog.make_filtering_bound_logger(logging.CRITICAL), False),
        (structlog.BoundLogger, True),
    ],
)
@pytest.mark.usefixtures('_restore_structlog')
def test_is_debug_enabled_against_real_wrapper_classes(wrapper: Any, expected: bool) -> None:
    """Every structlog wrapper class answers without raising.

    ``structlog.BoundLogger`` proxies unknown attributes to the wrapped logger, so a
    level check looks callable and then fails; that must not escape.
    """
    structlog.reset_defaults()
    if wrapper is not None:
        structlog.configure(wrapper_class=wrapper)

    assert is_debug_enabled(get_logger('test')) is expected


@pytest.mark.parametrize(
    ('root_level', 'expected'),
    [(logging.DEBUG, True), (logging.WARNING, False)],
)
@pytest.mark.usefixtures('_restore_structlog')
def test_is_debug_enabled_against_stdlib_wrapper(root_level: int, expected: bool) -> None:
    """The stdlib wrapper exposes isEnabledFor and is consulted through it."""
    structlog.reset_defaults()
    structlog.configure(
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    previous = logging.getLogger().level
    logging.getLogger().setLevel(root_level)
    try:
        assert is_debug_enabled(get_logger('test')) is expected
    finally:
        logging.getLogger().setLevel(previous)


@pytest.mark.usefixtures('_restore_structlog')
def test_is_debug_enabled_when_the_level_check_cannot_run() -> None:
    """A wrapper whose level check proxies to a logger lacking it falls back to enabled."""
    structlog.reset_defaults()
    structlog.configure(wrapper_class=structlog.stdlib.BoundLogger)

    assert is_debug_enabled(get_logger('test')) is True
