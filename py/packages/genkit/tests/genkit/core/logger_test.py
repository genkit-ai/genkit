# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the logger module."""

import logging
import os
from unittest import mock

from genkit._core._logger import QUIET_LOGGERS, configure_logging, resolve_level


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


def test_configure_logging_mutes_quiet_loggers() -> None:
    """Test that configure_logging sets QUIET_LOGGERS to WARNING in dev environment."""
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


def test_configure_logging_respects_higher_levels() -> None:
    """Test that GENKIT_LOG=error sets QUIET_LOGGERS to ERROR."""
    with (
        mock.patch.dict(os.environ, {'GENKIT_LOG': 'error'}),
        mock.patch('logging.getLogger') as mock_get_logger,
    ):
        mock_get_logger.return_value.level = logging.NOTSET
        configure_logging(shared_tty=True)
        for name in QUIET_LOGGERS:
            mock_get_logger.assert_any_call(name)
        expected_calls = [mock.call(logging.ERROR)] * len(QUIET_LOGGERS)
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
