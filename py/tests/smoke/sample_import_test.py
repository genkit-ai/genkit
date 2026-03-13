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

"""Smoke tests for sample imports.

Verifies that:
1. ``samples.shared`` imports cleanly.
2. Every sample's ``main.py`` is syntactically valid Python.
3. Every ``from samples.shared import X`` in a sample resolves to a
   real name exported by ``samples.shared``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Resolve py/samples/ relative to this test file (py/tests/smoke/).
_SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / 'samples'


def _find_sample_main_files() -> list[Path]:
    """Discover all sample main.py files."""
    results: list[Path] = []
    for sample_dir in sorted(_SAMPLES_DIR.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name.startswith(('.', '_')):
            continue
        # Most samples: src/main.py
        main_py = sample_dir / 'src' / 'main.py'
        if main_py.is_file():
            results.append(main_py)
            continue
        # framework-evaluator-demo: evaluator_demo/main.py
        for sub in sample_dir.iterdir():
            if sub.is_dir() and (sub / 'main.py').is_file():
                results.append(sub / 'main.py')
                break
    return results


def _extract_shared_imports(source: str) -> list[str]:
    """Extract names imported from ``samples.shared`` via AST."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('samples.shared'):
            for alias in node.names:
                names.append(alias.name)
    return names


# ── Tests ────────────────────────────────────────────────────────────


def test_samples_shared_imports() -> None:
    """samples.shared itself imports without errors."""
    import samples.shared  # noqa: F401

    assert hasattr(samples.shared, '__all__')
    assert len(samples.shared.__all__) > 0


def test_samples_shared_has_expected_exports() -> None:
    """samples.shared exports the key types and functions."""
    import samples.shared

    expected = [
        'setup_sample',
        'GreetingInput',
        'CharacterInput',
        'WeatherInput',
        'get_weather',
        'calculate',
        'convert_currency',
    ]
    for name in expected:
        assert hasattr(samples.shared, name), f'samples.shared missing export: {name}'


_SAMPLE_MAINS = _find_sample_main_files()


@pytest.mark.parametrize(
    'main_py',
    _SAMPLE_MAINS,
    ids=[str(p.relative_to(_SAMPLES_DIR)) for p in _SAMPLE_MAINS],
)
def test_sample_syntax_valid(main_py: Path) -> None:
    """Each sample main.py is syntactically valid Python."""
    source = main_py.read_text(encoding='utf-8')
    try:
        ast.parse(source, filename=str(main_py))
    except SyntaxError as exc:
        pytest.fail(f'Syntax error in {main_py}: {exc}')


@pytest.mark.parametrize(
    'main_py',
    _SAMPLE_MAINS,
    ids=[str(p.relative_to(_SAMPLES_DIR)) for p in _SAMPLE_MAINS],
)
def test_sample_shared_imports_resolve(main_py: Path) -> None:
    """All ``from samples.shared import X`` names exist in samples.shared."""
    import samples.shared

    source = main_py.read_text(encoding='utf-8')
    imported_names = _extract_shared_imports(source)
    if not imported_names:
        pytest.skip('No samples.shared imports')

    missing = [n for n in imported_names if not hasattr(samples.shared, n)]
    # Also check submodules (e.g. samples.shared.logging.setup_sample)
    submodule_names = {n for n in imported_names if '.' not in n}
    missing = [n for n in submodule_names if not hasattr(samples.shared, n)]

    assert not missing, (
        f'{main_py.relative_to(_SAMPLES_DIR)}: imports {missing} from samples.shared but they do not exist'
    )
