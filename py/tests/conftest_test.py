# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for workspace pytest hooks in py/conftest.py."""

import argparse
import bdb
import contextlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest import mock

import pytest
import structlog


def _load_root_conftest() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / 'conftest.py'
    spec = importlib.util.spec_from_file_location('py_workspace_conftest', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root_conftest = _load_root_conftest()


class _FakeConfig:
    def __init__(self, *, numprocesses: str | int = 'auto', dist: str = 'worksteal', capture: str = 'fd') -> None:
        self.option = argparse.Namespace(numprocesses=numprocesses, dist=dist)
        self._capture = capture

    def getoption(self, name: str, default: object = None) -> object:
        if name == 'capture':
            return self._capture
        return default


def test_cmdline_main_keeps_xdist_auto_by_default() -> None:
    """Default captured test runs keep parallel worksteal distribution."""
    cfg = _FakeConfig(numprocesses='auto', dist='worksteal', capture='fd')
    with mock.patch.object(root_conftest, '_debugger_attached', return_value=False):
        root_conftest.pytest_cmdline_main(cast(pytest.Config, cfg))
    assert cfg.option.numprocesses == 'auto'
    assert cfg.option.dist == 'worksteal'


def test_cmdline_main_disables_xdist_when_capture_disabled() -> None:
    """Passing -s / --capture=no drops to in-process so stdin breakpoint() works."""
    cfg = _FakeConfig(numprocesses='auto', dist='worksteal', capture='no')
    with mock.patch.object(root_conftest, '_debugger_attached', return_value=False):
        root_conftest.pytest_cmdline_main(cast(pytest.Config, cfg))
    assert cfg.option.numprocesses == 0
    assert cfg.option.dist == 'no'


def test_cmdline_main_disables_xdist_when_debugger_attached() -> None:
    """An attached IDE/bdb debugger drops automatic xdist to in-process."""
    cfg = _FakeConfig(numprocesses='auto', dist='worksteal', capture='fd')
    with mock.patch.object(root_conftest, '_debugger_attached', return_value=True):
        root_conftest.pytest_cmdline_main(cast(pytest.Config, cfg))
    assert cfg.option.numprocesses == 0
    assert cfg.option.dist == 'no'


def test_cmdline_main_respects_explicit_worker_count() -> None:
    """Explicit -n <N> is preserved even when -s is passed."""
    cfg = _FakeConfig(numprocesses=4, dist='worksteal', capture='no')
    root_conftest.pytest_cmdline_main(cast(pytest.Config, cfg))
    assert cfg.option.numprocesses == 4
    assert cfg.option.dist == 'worksteal'


def test_debugger_attached_distinguishes_bdb_from_coverage() -> None:
    """bdb traces trigger single-process mode while coverage tracers do not."""

    class _FakeCoverageTracer:
        __module__ = 'coverage.tracer'

    debugger = bdb.Bdb()
    no_sysmon = (
        mock.patch('sys.monitoring.get_tool', return_value=None)
        if hasattr(sys, 'monitoring')
        else contextlib.nullcontext()
    )
    with mock.patch.object(sys, 'gettrace', return_value=debugger.trace_dispatch), no_sysmon:
        assert root_conftest._debugger_attached() is True

    no_sysmon_cov = (
        mock.patch('sys.monitoring.get_tool', return_value=None)
        if hasattr(sys, 'monitoring')
        else contextlib.nullcontext()
    )
    with mock.patch.object(sys, 'gettrace', return_value=_FakeCoverageTracer()), no_sysmon_cov:
        assert root_conftest._debugger_attached() is False


def test_pytest_configure_preserves_structlog_wrapper_class() -> None:
    """Swapping ConsoleRenderer to plain_traceback leaves wrapper_class intact."""
    saved = structlog.get_config()
    custom_wrapper = structlog.make_filtering_bound_logger(logging.WARNING)
    try:
        structlog.configure(wrapper_class=custom_wrapper)
        root_conftest.pytest_configure()
        updated = structlog.get_config()
        assert updated['wrapper_class'] is custom_wrapper
        renderers = [p for p in updated['processors'] if isinstance(p, structlog.dev.ConsoleRenderer)]
        assert len(renderers) == 1
        assert renderers[0]._exception_formatter is structlog.dev.plain_traceback
    finally:
        structlog.configure(**saved)
