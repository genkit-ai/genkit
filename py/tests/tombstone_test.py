# Copyright 2025 Google LLC
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

"""Smoke tests for the deprecation tombstones built by publish_tombstones.py.

The migration's core promise is that a user who kept the old dependency and old
imports (``from genkit.plugins.<x> import ...``) keeps working after upgrade,
just with a ``DeprecationWarning``. These tests reproduce the installed on-disk
layout a tombstone wheel creates next to core ``genkit`` and prove the old
import path still resolves.
"""

import importlib.util
import os
import subprocess  # noqa: S404
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'publish_tombstones.py'
_SPEC = importlib.util.spec_from_file_location('publish_tombstones', _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
tombstones = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tombstones)


def test_shim_survives_module_without_public_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A shim for a private-only module must import cleanly, not raise NameError.

    Leaf modules like ``constants.py`` can expose only underscore-prefixed names.
    That leaves ``__all__`` empty and the copy loop never runs, so the cleanup at
    the end must not assume the loop variable was bound.
    """
    (tmp_path / 'privateonly.py').write_text('_INTERNAL = 1\n')
    monkeypatch.syspath_prepend(str(tmp_path))

    shim_src = tombstones.SHIM.format(
        old_dist='genkit-plugin-x',
        new_dist='genkit-x',
        old_import='x',
        new_import='genkit_x',
        new_module='privateonly',
    )

    namespace: dict[str, object] = {}
    with pytest.warns(DeprecationWarning):
        exec(compile(shim_src, '<shim>', 'exec'), namespace)  # noqa: S102

    assert namespace['__all__'] == []
    assert '_mod' not in namespace
    assert '_name' not in namespace


def test_old_import_path_resolves_next_to_core(tmp_path: Path) -> None:
    """``from genkit.plugins.<x> import ...`` works when a tombstone sits by core.

    A tombstone ships only ``genkit/plugins/<x>/...`` with no ``genkit/__init__.py``
    or ``genkit/plugins/__init__.py``, relying on core ``genkit`` and PEP 420
    namespace resolution for the ``plugins`` layer. This mirrors the merged
    site-packages layout of ``genkit`` + a tombstone and asserts the old import
    still works and warns.
    """
    # write_shim lays files under <root>/src/genkit/plugins/<x>/..., so drop the
    # stub core and new package under the same src/ dir to model one merged tree.
    src = tmp_path / 'src'

    core = src / 'genkit'
    core.mkdir(parents=True)
    (core / '__init__.py').write_text("Genkit = 'CORE'\n")

    new_pkg = src / 'genkit_ollama'
    new_pkg.mkdir(parents=True)
    (new_pkg / '__init__.py').write_text("Ollama = 'NEW_OLLAMA'\n__all__ = ['Ollama']\n")
    (new_pkg / 'constants.py').write_text("DEFAULT_OLLAMA_SERVER_URL = 'http://127.0.0.1:11434'\n")

    for rel in (Path('__init__.py'), Path('constants.py')):
        tombstones.write_shim(
            str(tmp_path),
            old_dist='genkit-plugin-ollama',
            new_dist='genkit-ollama',
            old_import='ollama',
            new_import='genkit_ollama',
            rel_py_path=rel,
        )

    # The whole point: nothing marks plugins as a regular package.
    assert not (core / 'plugins' / '__init__.py').exists()

    check = textwrap.dedent(
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            from genkit.plugins.ollama import Ollama
            from genkit.plugins.ollama.constants import DEFAULT_OLLAMA_SERVER_URL

        import genkit

        assert genkit.Genkit == 'CORE', genkit.Genkit
        assert Ollama == 'NEW_OLLAMA', Ollama
        assert DEFAULT_OLLAMA_SERVER_URL.startswith('http'), DEFAULT_OLLAMA_SERVER_URL
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), 'no DeprecationWarning'
        print('SMOKE_OK')
        """
    )

    env = os.environ.copy()
    env['PYTHONPATH'] = str(src) + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(  # noqa: S603
        [sys.executable, '-c', check],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert 'SMOKE_OK' in result.stdout
