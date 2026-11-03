# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the logger module."""

import logging
import os
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest
import structlog
from structlog.processors import JSONRenderer
from structlog.testing import capture_logs

from genkit._core._logger import (
    GENKIT_LOG,
    REFLECTION_LOGGERS,
    REFLECTION_THREAD_NAME,
    ReflectionLogFilter,
    configure_logging,
    configure_structlog_level,
    get_logger,
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


def _record(level: int, thread_name: str) -> logging.LogRecord:
    record = logging.LogRecord('uvicorn.access', level, __file__, 1, 'GET /api/__health', None, None)
    record.threadName = thread_name
    return record


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
    with mock.patch.dict(os.environ, {GENKIT_LOG: genkit_log}):
        assert ReflectionLogFilter().filter(_record(level, REFLECTION_THREAD_NAME)) is passes
        assert ReflectionLogFilter().filter(_record(level, 'MainThread')) is True


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


def test_configure_logging_leaves_stdlib_loggers_alone() -> None:
    """configure_logging only sets the structlog level; stdlib loggers are the app's."""
    with mock.patch('logging.getLogger') as mock_get_logger:
        configure_logging()
        mock_get_logger.assert_not_called()


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
