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

"""Workspace-wide pytest configuration."""

import ast
import os
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import pytest
import structlog
from structlog.typing import ExcInfo

_WORKSPACE_ROOT = Path(__file__).resolve().parent
_SKIP_DIRS = frozenset({
    '.git',
    '.hg',
    '.mypy_cache',
    '.nox',
    '.pytest_cache',
    '.ruff_cache',
    '.tox',
    '.venv',
    '__pycache__',
    'build',
    'dist',
    'node_modules',
    'site-packages',
    'venv',
})


def _debugger_attached() -> bool:
    """Return True when a Python debugger (not coverage) is attached."""
    monitoring = getattr(sys, 'monitoring', None)
    if monitoring is not None and monitoring.get_tool(monitoring.DEBUGGER_ID) is not None:
        return True
    trace = sys.gettrace()
    if trace is None:
        return False
    module = getattr(trace, '__module__', None) or type(trace).__module__
    return not module.startswith('coverage')


def _source_has_breakpoint(source: bytes) -> bool:
    """Return True when *source* contains an AST call to ``breakpoint`` or ``set_trace``."""
    if b'breakpoint' not in source and b'set_trace' not in source:
        return False
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in {'breakpoint', 'set_trace'}:
            return True
        if isinstance(func, ast.Attribute) and func.attr in {'breakpoint', 'set_trace'}:
            return True
    return False


def _iter_py_files(roots: list[Path]) -> Iterator[str]:
    """Yield deduplicated ``.py`` file paths under *roots*, skipping ignored directories."""
    seen_dirs: set[str] = set()
    seen_files: set[str] = set()
    stack: list[str] = []

    for root in roots:
        resolved = os.path.realpath(root)
        if os.path.isfile(resolved):
            if resolved.endswith('.py') and resolved not in seen_files:
                seen_files.add(resolved)
                yield resolved
        elif os.path.isdir(resolved) and resolved not in seen_dirs:
            seen_dirs.add(resolved)
            stack.append(resolved)

    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.name in _SKIP_DIRS:
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if entry.path not in seen_dirs:
                            seen_dirs.add(entry.path)
                            stack.append(entry.path)
                    elif entry.name.endswith('.py') and entry.is_file(follow_symlinks=False):
                        if entry.path not in seen_files:
                            seen_files.add(entry.path)
                            yield entry.path
        except OSError:
            continue


def _has_active_breakpoint(config: pytest.Config) -> bool:
    """Return True when ``PYTHONBREAKPOINT!=0`` and any workspace or target ``.py`` file calls ``breakpoint()``."""
    if os.environ.get('PYTHONBREAKPOINT') == '0':
        return False

    roots = [_WORKSPACE_ROOT]
    invocation_dir = getattr(getattr(config, 'invocation_params', None), 'dir', None) or _WORKSPACE_ROOT
    for raw_arg in getattr(config, 'args', None) or ():
        if not isinstance(raw_arg, str) or raw_arg.startswith('-'):
            continue
        path_part = raw_arg.split('::', 1)[0]
        if not path_part:
            continue
        candidate = Path(path_part)
        if not candidate.is_absolute():
            candidate = Path(invocation_dir) / candidate
        if candidate.exists():
            roots.append(candidate)

    for file_path in _iter_py_files(roots):
        try:
            with open(file_path, 'rb') as handle:
                raw = handle.read()
        except OSError:
            continue
        if _source_has_breakpoint(raw):
            return True
    return False


def _plain_traceback_with_locals(sio: TextIO, exc_info: ExcInfo) -> None:
    """Format *exc_info* with frame locals via stdlib ``traceback`` (skipping Rich/Pygments)."""
    exc_type, exc_value, exc_tb = exc_info
    formatted = ''.join(
        traceback.TracebackException(exc_type, exc_value, exc_tb, capture_locals=True).format(chain=True)
    )
    if formatted.endswith('\n'):
        formatted = formatted[:-1]
    sio.write('\n' + formatted)


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    """Fall back to single-process runs for interactive debugging flags or active breakpoints."""
    if hasattr(config, 'workerinput'):
        return
    if getattr(config.option, 'numprocesses', None) in {'auto', 'logical'} and (
        config.getoption('capture', None) == 'no'
        or bool(config.getoption('usepdb', False))
        or bool(config.getoption('trace', False))
        or _debugger_attached()
        or (not bool(config.getoption('collectonly', False)) and _has_active_breakpoint(config))
    ):
        config.option.numprocesses = 0
        config.option.dist = 'no'


def pytest_configure() -> None:
    """Use stdlib tracebacks (with frame locals) in structlog during tests to skip Rich/Pygments lexing."""
    processors = [
        structlog.dev.ConsoleRenderer(
            colors=False,
            exception_formatter=_plain_traceback_with_locals,
        )
        if isinstance(processor, structlog.dev.ConsoleRenderer)
        else processor
        for processor in structlog.get_config()['processors']
    ]
    structlog.configure(processors=processors)
