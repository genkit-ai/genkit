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

    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'invalid'}):
        assert resolve_level() == logging.INFO


def test_configure_logging_mutes_quiet_loggers() -> None:
    """Test that configure_logging sets QUIET_LOGGERS to WARNING by default."""
    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'info'}):
        configure_logging(force=True)
        for name in QUIET_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING


def test_configure_logging_allows_debug() -> None:
    """Test that GENKIT_LOG=debug sets QUIET_LOGGERS to DEBUG."""
    with mock.patch.dict(os.environ, {'GENKIT_LOG': 'debug'}):
        configure_logging(force=True)
        for name in QUIET_LOGGERS:
            assert logging.getLogger(name).level == logging.DEBUG
