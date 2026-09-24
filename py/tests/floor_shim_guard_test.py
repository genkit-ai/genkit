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

"""Pins the floor shim's guards (py/floor-shim/setup.py).

The shim's whole job happens inside two `sys.exit` guards. If either
disappears, nothing else in CI notices: on old pip the metadata phase
would fail instead of the install phase and pip backtracks to the
pre-transfer squatter package with exit 0, and a wheel build would
produce an artifact that installs silently as an empty package. These
tests execute the real setup.py with spoofed interpreter versions so
deleting or weakening a guard fails loudly here.
"""

import ast
import sys
from pathlib import Path

import pytest

SHIM_SETUP = Path(__file__).resolve().parents[1] / 'floor-shim' / 'setup.py'

OLD_PYTHON = (3, 9, 6, 'final', 0)
NEW_PYTHON = (3, 12, 0, 'final', 0)


def _run_shim_setup(monkeypatch, argv: list[str], version: tuple) -> list:
    """Exec the real setup.py under a spoofed version/argv; return setup() calls."""
    calls: list = []
    monkeypatch.setattr(sys, 'argv', ['setup.py', *argv])
    monkeypatch.setattr(sys, 'version_info', version)
    import setuptools

    monkeypatch.setattr(setuptools, 'setup', lambda **kwargs: calls.append(kwargs))
    exec(compile(SHIM_SETUP.read_text(), str(SHIM_SETUP), 'exec'), {'__name__': '__main__'})  # noqa: S102
    return calls


def test_old_python_install_exits_with_upgrade_message(monkeypatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_shim_setup(monkeypatch, ['install'], OLD_PYTHON)
    assert 'Genkit requires Python 3.10+' in str(excinfo.value)


def test_old_python_bdist_wheel_still_gets_upgrade_message(monkeypatch) -> None:
    # Modern pip installs the sdist via build_wheel; the user must see the
    # upgrade box, not the sdist-only refusal.
    with pytest.raises(SystemExit) as excinfo:
        _run_shim_setup(monkeypatch, ['bdist_wheel'], OLD_PYTHON)
    assert 'Genkit requires Python 3.10+' in str(excinfo.value)


def test_old_python_metadata_commands_succeed(monkeypatch) -> None:
    # Old pip backtracks past metadata failures to the pre-transfer squatter
    # release, so egg_info must succeed and the failure must wait for the
    # install phase.
    for command in ('egg_info', 'dist_info', 'sdist'):
        calls = _run_shim_setup(monkeypatch, [command], OLD_PYTHON)
        assert len(calls) == 1, f'{command} must reach setup(), not exit'


def test_new_python_wheel_build_is_refused(monkeypatch) -> None:
    # sdist-only is self-enforcing: even a plain `uv build` (no --sdist) on a
    # current interpreter must not produce a wheel.
    with pytest.raises(SystemExit) as excinfo:
        _run_shim_setup(monkeypatch, ['bdist_wheel'], NEW_PYTHON)
    assert 'sdist-only' in str(excinfo.value)


def test_new_python_sdist_build_succeeds(monkeypatch) -> None:
    calls = _run_shim_setup(monkeypatch, ['sdist'], NEW_PYTHON)
    assert len(calls) == 1
    assert calls[0]['version'] == '0.3.0'
    assert calls[0]['python_requires'] == '<3.10'


def test_setup_py_annotations_stay_runtime_safe_on_old_pythons() -> None:
    # setup.py executes on whatever ancient interpreter the user has — that
    # is the shim's whole audience. Annotations evaluate at def time, and
    # subscripted generics like `tuple[int, ...]` raise TypeError before 3.9,
    # which would replace the upgrade message with a traceback. Plain builtin
    # names (`tuple`, `str`) are fine everywhere.
    tree = ast.parse(SHIM_SETUP.read_text())
    annotations: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if arg.annotation is not None:
                    annotations.append(arg.annotation)
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
    for annotation in annotations:
        assert isinstance(annotation, ast.Name), (
            f'annotation {ast.unparse(annotation)!r} is not a plain name; '
            'it would evaluate at def time and can crash Python < 3.9'
        )
