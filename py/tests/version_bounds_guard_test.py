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

"""Guards intra-repo version bounds, bump_version transitions, and PEP 561 py.typed markers."""

from __future__ import annotations

import importlib.util
import os
import subprocess  # noqa: S404
from pathlib import Path
from types import ModuleType

import pytest

PY_DIR = Path(__file__).resolve().parents[1]
BUMP_SCRIPT = PY_DIR / 'bin' / 'bump_version'
CONSISTENCY_SCRIPT = PY_DIR / 'scripts' / 'check_consistency.py'


def _load_consistency_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('check_consistency', CONSISTENCY_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


consistency = _load_consistency_module()


def _write_pkg(
    py_root: Path,
    pkg_name: str,
    version: str,
    *,
    dependencies: list[str],
    optional_dependencies: dict[str, list[str]] | None = None,
    classifiers: list[str] | None = None,
    create_py_typed: bool = True,
) -> Path:
    pkg_dir = py_root / 'packages' / pkg_name
    mod_dir = pkg_dir / 'src' / pkg_name.replace('-', '_')
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / '__init__.py').write_text('', encoding='utf-8')
    if create_py_typed:
        (mod_dir / 'py.typed').write_bytes(b'')
    (pkg_dir / 'README.md').write_text(f'# {pkg_name}\n', encoding='utf-8')
    (pkg_dir / 'LICENSE').write_text('Apache-2.0\n', encoding='utf-8')

    cls_list = classifiers if classifiers is not None else ['Typing :: Typed']
    cls_toml = ',\n  '.join(f'"{c}"' for c in cls_list)
    deps_toml = ',\n  '.join(f'"{d}"' for d in dependencies)

    opt_lines = ''
    if optional_dependencies:
        opt_lines = '\n[project.optional-dependencies]\n'
        for extra, specs in optional_dependencies.items():
            specs_toml = ', '.join(f'"{s}"' for s in specs)
            opt_lines += f'{extra} = [{specs_toml}]\n'

    toml = (
        '[project]\n'
        f'name = "{pkg_name}"\n'
        f'version = "{version}"\n'
        'requires-python = ">=3.10"\n'
        f'classifiers = [\n  {cls_toml}\n]\n'
        f'dependencies = [\n  {deps_toml}\n]\n'
        f'{opt_lines}'
    )
    (pkg_dir / 'pyproject.toml').write_text(toml, encoding='utf-8')
    return pkg_dir


def _make_workspace_fixture(py_root: Path, version: str = '0.12.0', *, include_evaluators: bool = True) -> None:
    cap = consistency.expected_major_cap(version)
    extras: dict[str, list[str]] = {
        'openai': [f'genkit-openai>={version},<{cap}'],
        'vertex-ai': [f'genkit-vertexai[anthropic,openai]>={version},<{cap}'],
        'vertexai': [f'genkit-vertexai[anthropic,openai]>={version},<{cap}'],
    }
    if include_evaluators:
        extras['evaluators'] = [f'genkit-evaluators>={version},<1']

    _write_pkg(
        py_root,
        'genkit',
        version,
        dependencies=['pydantic>=2.11'],
        optional_dependencies=extras,
    )
    _write_pkg(
        py_root,
        'genkit-openai',
        version,
        dependencies=[f'genkit>={version},<{cap}', 'openai>=1.60.0,<3'],
    )
    _write_pkg(
        py_root,
        'genkit-anthropic',
        version,
        dependencies=[f'genkit>={version},<{cap}', 'anthropic>=0.96.0'],
    )
    _write_pkg(
        py_root,
        'genkit-vertexai',
        version,
        dependencies=[f'genkit>={version},<{cap}'],
        optional_dependencies={
            'anthropic': [f'genkit-anthropic=={version}'],
            'openai': [f'genkit-openai=={version}'],
        },
    )
    if include_evaluators:
        _write_pkg(
            py_root,
            'genkit-evaluators',
            version,
            dependencies=[f'genkit>={version},<1', 'jsonata-python>=0.6.0'],
        )


def _run_bump(py_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, 'GENKIT_PY_DIR': str(py_root)}
    return subprocess.run(  # noqa: S603
        [str(BUMP_SCRIPT), '--no-lock', *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def _workspace_errors(py_root: Path) -> list[str]:
    _, pkg_results, sample_results = consistency.check_workspace(str(py_root))
    errors: list[str] = []
    for pkg_name, _, pkg_errs in pkg_results:
        errors.extend(f'{pkg_name}: {err}' for err in pkg_errs)
    for sample_name, sample_errs in sample_results:
        errors.extend(f'{sample_name}: {err}' for err in sample_errs)
    return errors


def test_real_workspace_passes_consistency_and_extras_contract() -> None:
    assert consistency.main(str(PY_DIR)) == 0

    genkit_project = consistency.load_pyproject(str(PY_DIR / 'packages' / 'genkit' / 'pyproject.toml'))['project']
    core_version = genkit_project['version']
    opt_deps = genkit_project['optional-dependencies']

    # Both vertex-ai and vertexai aliases resolve to the same bounded target.
    assert opt_deps['vertex-ai'] == [f'genkit-vertexai[anthropic,openai]>={core_version},<2']
    assert opt_deps['vertexai'] == [f'genkit-vertexai[anthropic,openai]>={core_version},<2']
    assert opt_deps['evaluators'] == [f'genkit-evaluators>={core_version},<1']

    # Every publishable plugin is reachable from at least one genkit[...] extra.
    referenced_plugins = {spec.split('[', 1)[0].split('>=', 1)[0] for specs in opt_deps.values() for spec in specs}
    all_plugins = {
        p.name
        for p in (PY_DIR / 'packages').iterdir()
        if p.is_dir() and p.name != 'genkit' and (p / 'pyproject.toml').exists()
    }
    assert referenced_plugins == all_plugins

    # genkit-openai bounds openai>=1.60.0,<3.
    openai_project = consistency.load_pyproject(str(PY_DIR / 'packages' / 'genkit-openai' / 'pyproject.toml'))[
        'project'
    ]
    assert 'openai>=1.60.0,<3' in openai_project['dependencies']


def test_bump_transition_0_12_0_to_0_13_0_rolls_floors_and_preserves_lt1(tmp_path: Path) -> None:
    _make_workspace_fixture(tmp_path, '0.12.0', include_evaluators=True)
    assert _workspace_errors(tmp_path) == []

    _run_bump(tmp_path, '0.13.0')
    assert _workspace_errors(tmp_path) == []

    genkit_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit' / 'pyproject.toml'))['project']
    assert genkit_proj['version'] == '0.13.0'
    assert genkit_proj['optional-dependencies']['openai'] == ['genkit-openai>=0.13.0,<2']
    assert genkit_proj['optional-dependencies']['vertexai'] == ['genkit-vertexai[anthropic,openai]>=0.13.0,<2']
    assert genkit_proj['optional-dependencies']['evaluators'] == ['genkit-evaluators>=0.13.0,<1']

    openai_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit-openai' / 'pyproject.toml'))['project']
    assert openai_proj['dependencies'] == ['genkit>=0.13.0,<2', 'openai>=1.60.0,<3']

    evals_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit-evaluators' / 'pyproject.toml'))[
        'project'
    ]
    assert evals_proj['dependencies'] == ['genkit>=0.13.0,<1', 'jsonata-python>=0.6.0']

    vertex_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit-vertexai' / 'pyproject.toml'))[
        'project'
    ]
    assert vertex_proj['optional-dependencies']['openai'] == ['genkit-openai==0.13.0']


def test_bump_transition_0_13_0_to_1_0_0_keeps_lt2_cap(tmp_path: Path) -> None:
    _make_workspace_fixture(tmp_path, '0.13.0', include_evaluators=False)
    assert _workspace_errors(tmp_path) == []

    _run_bump(tmp_path, '1.0.0')
    assert _workspace_errors(tmp_path) == []

    genkit_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit' / 'pyproject.toml'))['project']
    assert genkit_proj['version'] == '1.0.0'
    assert genkit_proj['optional-dependencies']['openai'] == ['genkit-openai>=1.0.0,<2']
    assert genkit_proj['optional-dependencies']['vertex-ai'] == ['genkit-vertexai[anthropic,openai]>=1.0.0,<2']
    assert genkit_proj['optional-dependencies']['vertexai'] == ['genkit-vertexai[anthropic,openai]>=1.0.0,<2']

    openai_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit-openai' / 'pyproject.toml'))['project']
    assert openai_proj['dependencies'] == ['genkit>=1.0.0,<2', 'openai>=1.60.0,<3']

    vertex_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit-vertexai' / 'pyproject.toml'))[
        'project'
    ]
    assert vertex_proj['optional-dependencies']['anthropic'] == ['genkit-anthropic==1.0.0']


@pytest.mark.parametrize('start_version', ['1.0.0', '1.2.0'])
def test_bump_transition_1_x_to_2_0_0_advances_cap_to_lt3(tmp_path: Path, start_version: str) -> None:
    _make_workspace_fixture(tmp_path, start_version, include_evaluators=False)
    assert _workspace_errors(tmp_path) == []

    _run_bump(tmp_path, '2.0.0')
    assert _workspace_errors(tmp_path) == []

    genkit_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit' / 'pyproject.toml'))['project']
    assert genkit_proj['version'] == '2.0.0'
    assert genkit_proj['optional-dependencies']['openai'] == ['genkit-openai>=2.0.0,<3']
    assert genkit_proj['optional-dependencies']['vertexai'] == ['genkit-vertexai[anthropic,openai]>=2.0.0,<3']

    openai_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit-openai' / 'pyproject.toml'))['project']
    assert openai_proj['dependencies'] == ['genkit>=2.0.0,<3', 'openai>=1.60.0,<3']

    vertex_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit-vertexai' / 'pyproject.toml'))[
        'project'
    ]
    assert vertex_proj['optional-dependencies']['openai'] == ['genkit-openai==2.0.0']


def test_bump_transition_multi_digit_major_9_to_10(tmp_path: Path) -> None:
    _make_workspace_fixture(tmp_path, '9.4.0', include_evaluators=False)
    assert _workspace_errors(tmp_path) == []

    _run_bump(tmp_path, '10.0.0')
    assert _workspace_errors(tmp_path) == []

    openai_proj = consistency.load_pyproject(str(tmp_path / 'packages' / 'genkit-openai' / 'pyproject.toml'))['project']
    assert openai_proj['dependencies'] == ['genkit>=10.0.0,<11', 'openai>=1.60.0,<3']


@pytest.mark.parametrize(
    ('bad_dep', 'expected_substring'),
    [
        ('genkit', "must declare floor and major cap 'genkit>=0.12.0,<2'"),
        ('genkit>=0.12.0', "must declare floor and major cap 'genkit>=0.12.0,<2'"),
        ('genkit>=0.11.0,<2', "expected '>=0.12.0,<2'"),
        ('genkit>=0.12.0,<1', "expected '>=0.12.0,<2'"),
        ('genkit>=0.12.0,<3', "expected '>=0.12.0,<2'"),
    ],
)
def test_consistency_rejects_unbounded_or_misbounded_plugin_deps(
    tmp_path: Path,
    bad_dep: str,
    expected_substring: str,
) -> None:
    _make_workspace_fixture(tmp_path, '0.12.0', include_evaluators=False)
    _write_pkg(tmp_path, 'genkit-openai', '0.12.0', dependencies=[bad_dep])

    errors = _workspace_errors(tmp_path)
    assert any(expected_substring in err for err in errors), errors


def test_consistency_rejects_missing_py_typed_and_classifier(tmp_path: Path) -> None:
    _make_workspace_fixture(tmp_path, '0.12.0', include_evaluators=False)
    _write_pkg(
        tmp_path,
        'genkit-otel',
        '0.12.0',
        dependencies=['genkit>=0.12.0,<2'],
        classifiers=['Development Status :: 4 - Beta'],
        create_py_typed=False,
    )

    errors = _workspace_errors(tmp_path)
    assert any('missing PEP 561 marker src/genkit_otel/py.typed' in err for err in errors), errors
    assert any("missing 'Typing :: Typed' classifier" in err for err in errors), errors


def test_consistency_rejects_unretired_zero_x_package_at_1_0_0(tmp_path: Path) -> None:
    _make_workspace_fixture(tmp_path, '1.0.0', include_evaluators=True)

    errors = _workspace_errors(tmp_path)
    assert any('0.x-only package (<1), which must be retired before 1.0.0' in err for err in errors), errors


def test_fallback_toml_parser_matches_tomllib_on_workspace_packages() -> None:
    for pkg_toml in sorted((PY_DIR / 'packages').glob('*/pyproject.toml')):
        parsed = consistency.load_pyproject(str(pkg_toml))['project']
        fallback = consistency._fallback_parse_toml(str(pkg_toml))['project']
        for key in ('name', 'version', 'requires-python', 'classifiers', 'dependencies', 'optional-dependencies'):
            assert fallback.get(key) == parsed.get(key), f'Mismatch on {key} in {pkg_toml}'


def test_consistency_rejects_unbounded_extra_and_stale_exact_pin(tmp_path: Path) -> None:
    _make_workspace_fixture(tmp_path, '0.12.0', include_evaluators=False)
    _write_pkg(
        tmp_path,
        'genkit',
        '0.12.0',
        dependencies=['pydantic>=2.11'],
        optional_dependencies={'openai': ['genkit-openai']},
    )
    _write_pkg(
        tmp_path,
        'genkit-vertexai',
        '0.12.0',
        dependencies=['genkit>=0.12.0,<2'],
        optional_dependencies={'openai': ['genkit-openai==0.11.0']},
    )

    errors = _workspace_errors(tmp_path)
    assert any("optional-dependencies.openai: 'genkit-openai' must declare floor" in err for err in errors), errors
    assert any(
        "optional-dependencies.openai: pins genkit-openai==0.11.0 (expected '0.12.0')" in err for err in errors
    ), errors


def test_bump_version_dry_run_leaves_files_untouched(tmp_path: Path) -> None:
    _make_workspace_fixture(tmp_path, '0.12.0', include_evaluators=True)
    before = (tmp_path / 'packages' / 'genkit-openai' / 'pyproject.toml').read_text(encoding='utf-8')

    proc = _run_bump(tmp_path, '--dry-run', '0.13.0')
    after = (tmp_path / 'packages' / 'genkit-openai' / 'pyproject.toml').read_text(encoding='utf-8')
    assert before == after
    assert 'genkit>=0.13.0,<2' in proc.stdout
    assert 'genkit>=0.13.0,<1' in proc.stdout
