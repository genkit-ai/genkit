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

"""Workspace metadata guards: published extras and workspace source names."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = pytest.importorskip('tomli')

PY_DIR = Path(__file__).resolve().parents[1]
TEST_ONLY_EXTRAS = frozenset({'dev', 'test', 'tests', 'lint'})


def _load(path: Path) -> dict[str, Any]:
    with path.open('rb') as f:
        return cast(dict[str, Any], tomllib.load(f))


@pytest.mark.parametrize(
    'pyproject', sorted((PY_DIR / 'packages').glob('*/pyproject.toml')), ids=lambda p: p.parent.name
)
def test_published_packages_have_no_test_only_extras(pyproject: Path) -> None:
    """Test tools belong in [dependency-groups]; extras ship in wheel metadata as Provides-Extra."""
    extras = set(_load(pyproject)['project'].get('optional-dependencies', {}))

    assert not extras & TEST_ONLY_EXTRAS


def test_every_sample_is_a_named_workspace_source() -> None:
    """[tool.uv.sources] in py/pyproject.toml names each sample by its project name."""
    sources = set(_load(PY_DIR / 'pyproject.toml')['tool']['uv']['sources'])
    samples = {_load(p)['project']['name'] for p in (PY_DIR / 'samples').glob('*/pyproject.toml')}

    assert samples - sources == set()
    assert {name for name in sources if not name.startswith('genkit')} - samples == set()
