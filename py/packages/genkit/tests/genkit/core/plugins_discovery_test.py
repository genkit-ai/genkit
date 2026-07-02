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

"""Tests for runtime plugin-namespace discovery.

``genkit.plugins`` is a regular package whose ``__path__`` is extended at
runtime by ``extend_plugin_namespace`` so that plugins shipped as separate
distributions (each contributing a bare ``genkit/plugins/<name>`` directory)
become importable as ``genkit.plugins.<name>``. These tests exercise that
discovery against a fake plugin laid out on a temporary ``sys.path`` entry.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import genkit.plugins
from genkit._core._plugins import extend_plugin_namespace

FAKE_PLUGIN = 'fake_discovery_plugin'


def _write_fake_plugin(root: Path) -> Path:
    """Lay out ``<root>/genkit/plugins/<FAKE_PLUGIN>`` as a PEP 420 portion.

    Mirrors how a real plugin distribution ships: no ``genkit/__init__.py`` and
    no ``genkit/plugins/__init__.py``, only the leaf plugin package.
    """
    plugin_dir = root / 'genkit' / 'plugins' / FAKE_PLUGIN
    plugin_dir.mkdir(parents=True)
    (plugin_dir / '__init__.py').write_text('SENTINEL = "discovered"\n')
    return root / 'genkit' / 'plugins'


@pytest.fixture
def isolated_namespace() -> Iterator[None]:
    """Snapshot and restore global discovery state mutated by the tests."""
    original_path = list(genkit.plugins.__path__)
    original_sys_path = list(sys.path)
    try:
        yield
    finally:
        genkit.plugins.__path__[:] = original_path
        sys.path[:] = original_sys_path
        sys.modules.pop(f'genkit.plugins.{FAKE_PLUGIN}', None)
        # Importing the submodule sets it as an attribute on the parent module;
        # popping sys.modules alone leaves that attribute behind.
        if hasattr(genkit.plugins, FAKE_PLUGIN):
            delattr(genkit.plugins, FAKE_PLUGIN)
        importlib.invalidate_caches()


def test_discovers_plugin_dir_on_sys_path(tmp_path: Path, isolated_namespace: None) -> None:
    """A ``genkit/plugins`` directory on ``sys.path`` is grafted in and importable."""
    plugins_dir = _write_fake_plugin(tmp_path)
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()

    extend_plugin_namespace()

    assert str(plugins_dir) in genkit.plugins.__path__

    module = importlib.import_module(f'genkit.plugins.{FAKE_PLUGIN}')
    assert module.SENTINEL == 'discovered'


def test_discovery_is_idempotent(tmp_path: Path, isolated_namespace: None) -> None:
    """Repeated calls do not append duplicate ``__path__`` entries."""
    plugins_dir = _write_fake_plugin(tmp_path)
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()

    extend_plugin_namespace()
    extend_plugin_namespace()

    assert genkit.plugins.__path__.count(str(plugins_dir)) == 1


def test_ignores_sys_path_entry_without_plugins_dir(tmp_path: Path, isolated_namespace: None) -> None:
    """A ``sys.path`` entry lacking ``genkit/plugins`` is left untouched."""
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()

    extend_plugin_namespace()

    assert str(tmp_path / 'genkit' / 'plugins') not in genkit.plugins.__path__
