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
import io
import logging
import os
import shutil
import subprocess  # noqa: S404
import sys
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest import mock

import pytest
import structlog

_PY_ROOT = Path(__file__).resolve().parents[1]


def _load_root_conftest() -> ModuleType:
    path = _PY_ROOT / 'conftest.py'
    spec = importlib.util.spec_from_file_location('py_workspace_conftest', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root_conftest = _load_root_conftest()


class _FakeConfig:
    def __init__(
        self,
        *,
        numprocesses: str | int = 'auto',
        dist: str = 'worksteal',
        capture: str = 'fd',
        usepdb: bool = False,
        trace: bool = False,
        args: list[str] | None = None,
    ) -> None:
        self.option = argparse.Namespace(numprocesses=numprocesses, dist=dist)
        self.args = args or []
        self._options: dict[str, object] = {
            'capture': capture,
            'usepdb': usepdb,
            'trace': trace,
        }

    def getoption(self, name: str, default: object = None) -> object:
        return self._options.get(name, default)


def test_cmdline_main_keeps_xdist_auto_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default captured test runs on the clean workspace keep parallel worksteal distribution."""
    monkeypatch.delenv('PYTHONBREAKPOINT', raising=False)
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


@pytest.mark.parametrize(('usepdb', 'trace'), [(True, False), (False, True)])
def test_cmdline_main_disables_xdist_for_pdb_and_trace_flags(usepdb: bool, trace: bool) -> None:
    """Passing --pdb or --trace drops automatic xdist to in-process."""
    cfg = _FakeConfig(numprocesses='auto', dist='worksteal', capture='fd', usepdb=usepdb, trace=trace)
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


def test_cmdline_main_detects_active_breakpoints_via_ast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AST scan catches breakpoint() and set_trace(), ignores strings/comments, and honors PYTHONBREAKPOINT=0."""
    monkeypatch.delenv('PYTHONBREAKPOINT', raising=False)

    inert = tmp_path / 'inert_test.py'
    inert.write_text(
        '# breakpoint() in a comment\n'
        'DOC = """pdb.set_trace() and breakpoint() in a string"""\n'
        'def test_inert() -> None:\n'
        '    assert DOC\n',
        encoding='utf-8',
    )
    cfg_inert = _FakeConfig(args=[str(inert)])
    with mock.patch.object(root_conftest, '_debugger_attached', return_value=False):
        root_conftest.pytest_cmdline_main(cast(pytest.Config, cfg_inert))
    assert cfg_inert.option.numprocesses == 'auto'

    bp_file = tmp_path / 'bp_test.py'
    bp_file.write_text('def test_bp() -> None:\n    ' + 'breakpoint' + '()\n', encoding='utf-8')
    cfg_bp = _FakeConfig(args=[f'{bp_file}::test_bp'])
    with mock.patch.object(root_conftest, '_debugger_attached', return_value=False):
        root_conftest.pytest_cmdline_main(cast(pytest.Config, cfg_bp))
    assert cfg_bp.option.numprocesses == 0
    assert cfg_bp.option.dist == 'no'

    monkeypatch.setenv('PYTHONBREAKPOINT', '0')
    cfg_disabled = _FakeConfig(args=[str(bp_file)])
    with mock.patch.object(root_conftest, '_debugger_attached', return_value=False):
        root_conftest.pytest_cmdline_main(cast(pytest.Config, cfg_disabled))
    assert cfg_disabled.option.numprocesses == 'auto'

    monkeypatch.delenv('PYTHONBREAKPOINT', raising=False)
    bp_file.unlink()
    pdb_file = tmp_path / 'pdb_test.py'
    pdb_file.write_text('import pdb\ndef test_pdb() -> None:\n    pdb.' + 'set_trace' + '()\n', encoding='utf-8')
    cfg_pdb = _FakeConfig(args=[str(tmp_path)])
    with mock.patch.object(root_conftest, '_debugger_attached', return_value=False):
        root_conftest.pytest_cmdline_main(cast(pytest.Config, cfg_pdb))
    assert cfg_pdb.option.numprocesses == 0
    assert cfg_pdb.option.dist == 'no'


def test_plain_breakpoint_without_s_drops_into_pdb_under_default_addopts(tmp_path: Path) -> None:
    """Running pytest without -s on a test containing breakpoint() enters Pdb cleanly instead of BdbQuit."""
    shutil.copyfile(_PY_ROOT / 'conftest.py', tmp_path / 'conftest.py')
    sample = tmp_path / 'bp_subprocess_test.py'
    sample.write_text(
        'def test_hits_breakpoint() -> None:\n    value = 42\n    ' + 'breakpoint' + '()\n    assert value == 42\n',
        encoding='utf-8',
    )
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONBREAKPOINT'}
    result = subprocess.run(  # noqa: S603
        [sys.executable, '-m', 'pytest', '-c', str(_PY_ROOT / 'pyproject.toml'), str(sample)],
        input=b'c\n',
        capture_output=True,
        cwd=_PY_ROOT,
        env=env,
        timeout=30,
        check=False,
    )
    stdout = result.stdout.decode('utf-8', errors='replace')
    stderr = result.stderr.decode('utf-8', errors='replace')
    assert result.returncode == 0, f'stdout:\n{stdout}\nstderr:\n{stderr}'
    assert '(Pdb)' in stdout
    assert 'BdbQuit' not in stdout
    assert 'BdbQuit' not in stderr


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
    """Swapping ConsoleRenderer to _plain_traceback_with_locals leaves wrapper_class intact."""
    saved = structlog.get_config()
    custom_wrapper = structlog.make_filtering_bound_logger(logging.WARNING)
    try:
        structlog.configure(wrapper_class=custom_wrapper)
        root_conftest.pytest_configure()
        updated = structlog.get_config()
        assert updated['wrapper_class'] is custom_wrapper
        renderers = [p for p in updated['processors'] if isinstance(p, structlog.dev.ConsoleRenderer)]
        assert len(renderers) == 1
        assert renderers[0]._exception_formatter is root_conftest._plain_traceback_with_locals
    finally:
        structlog.configure(**saved)


def test_plain_traceback_with_locals_renders_frame_locals() -> None:
    """Configured test ConsoleRenderer prints frame locals on exc_info so secret-leak tests catch regressions."""
    canary = 'frame-local-redaction-canary-9b1e'
    output = io.StringIO()
    saved = structlog.get_config()
    try:
        structlog.configure(
            processors=[structlog.dev.ConsoleRenderer()],
            logger_factory=structlog.PrintLoggerFactory(output),
            cache_logger_on_first_use=False,
        )
        root_conftest.pytest_configure()

        def _fail_with_secret() -> None:
            env = {'API_KEY': canary}
            raise RuntimeError(f'transport closed ({len(env)} vars)')

        try:
            _fail_with_secret()
        except RuntimeError:
            structlog.get_logger().warning('connection ended', exc_info=True)  # noqa: TID251
    finally:
        structlog.configure(**saved)

    logged = output.getvalue()
    assert 'connection ended' in logged
    assert 'Traceback (most recent call last):' in logged
    assert canary in logged
