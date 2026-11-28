#!/usr/bin/env python3
# ruff: noqa: T201
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

"""Workspace consistency checks for the Genkit Python SDK."""

from __future__ import annotations

import ast
import os
import re
import sys
from typing import cast

# Color formatting
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_PYTHON = '>=3.10'
ZERO_X_ONLY_PACKAGES: frozenset[str] = frozenset({'genkit-evaluators'})

_INTRA_REPO_NAME_RE = re.compile(r'^\s*(genkit(?:-[a-z0-9-]+)?)\b')
_EXACT_PIN_RE = re.compile(r'^(genkit(?:-[a-z0-9-]+)?)(?:\[[a-z0-9,_-]+\])?==([^,;\s]+)$')
_CAPPED_RANGE_RE = re.compile(r'^(genkit(?:-[a-z0-9-]+)?)(?:\[[a-z0-9,_-]+\])?>=([^,;\s]+),<([^,;\s]+)$')
_CORE_DEP_RE = re.compile(r'^genkit(?![a-z0-9-])')


def _strip_toml_comment(line: str) -> str:
    """Strip trailing TOML # comments outside quoted strings."""
    in_quote: str | None = None
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == '\\' and in_quote == '"':
            escaped = True
            continue
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif ch == '#' and in_quote is None:
            return line[:idx]
    return line


def _fallback_parse_toml(filepath: str) -> dict[str, object]:
    """Minimal TOML reader for [project] and [project.optional-dependencies] on Python 3.10 without tomli."""
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    project: dict[str, object] = {}
    opt_deps: dict[str, list[str]] = {}
    section = ''
    idx = 0
    while idx < len(lines):
        raw = _strip_toml_comment(lines[idx]).strip()
        idx += 1
        if not raw:
            continue
        if raw.startswith('[') and raw.endswith(']'):
            section = raw[1:-1].strip()
            continue
        if section not in ('project', 'project.optional-dependencies') or '=' not in raw:
            continue
        key, rhs = raw.split('=', 1)
        key = key.strip().strip('"').strip("'")
        rhs = rhs.strip()
        while rhs.startswith('[') and not rhs.endswith(']') and idx < len(lines):
            nxt = _strip_toml_comment(lines[idx]).strip()
            idx += 1
            if nxt:
                rhs = f'{rhs} {nxt}'
        try:
            val = ast.literal_eval(rhs)
        except (SyntaxError, ValueError):
            continue
        if section == 'project':
            project[key] = val
        elif isinstance(val, list):
            opt_deps[key] = [str(item) for item in val]

    if opt_deps:
        project['optional-dependencies'] = opt_deps
    return {'project': project}


def load_pyproject(filepath: str) -> dict[str, object]:
    """Load a pyproject.toml file into a dictionary."""
    if not os.path.exists(filepath):
        return {}
    if sys.version_info >= (3, 11):
        import tomllib

        with open(filepath, 'rb') as f:
            return cast(dict[str, object], tomllib.load(f))
    try:
        import tomli  # type: ignore[import-not-found]

        with open(filepath, 'rb') as f:
            return cast(dict[str, object], tomli.load(f))
    except ModuleNotFoundError:
        return _fallback_parse_toml(filepath)


def expected_major_cap(core_version: str, *, zero_x_only: bool = False) -> str:
    """Return the expected upper major cap for a given core release version."""
    if zero_x_only:
        return '1'
    major = int(core_version.split('.', 1)[0])
    return str(max(2, major + 1))


def _check_intra_repo_spec(
    pkg_name: str,
    section: str,
    spec: str,
    core_version: str,
) -> str | None:
    """Validate an intra-repo genkit/genkit-* dependency specifier, returning an error message if invalid."""
    m_name = _INTRA_REPO_NAME_RE.match(spec)
    if not m_name:
        return None

    dep_name = m_name.group(1)

    # Exact intra-repo pins (e.g. genkit-vertexai extras importing private sibling modules).
    if section.startswith('optional-dependencies.'):
        m_pin = _EXACT_PIN_RE.match(spec)
        if m_pin:
            pinned = m_pin.group(2)
            if pinned != core_version:
                return f"{section}: pins {dep_name}=={pinned} (expected '{core_version}')"
            return None

    core_major = int(core_version.split('.', 1)[0])
    zero_x_only = (pkg_name in ZERO_X_ONLY_PACKAGES) or (dep_name in ZERO_X_ONLY_PACKAGES)
    if zero_x_only and core_major >= 1:
        return f"{section}: '{spec}' targets 0.x-only package (<1), which must be retired before {core_version}"

    exp_cap = expected_major_cap(core_version, zero_x_only=zero_x_only)
    m_range = _CAPPED_RANGE_RE.match(spec)
    if not m_range:
        return f"{section}: '{spec}' must declare floor and major cap '{dep_name}>={core_version},<{exp_cap}'"

    _, floor, cap = m_range.groups()
    if floor != core_version or cap != exp_cap:
        return f"{section}: '{spec}' has bounds '>={floor},<{cap}' (expected '>={core_version},<{exp_cap}')"
    return None


def check_package(pkg_path: str, core_version: str) -> tuple[str, str, list[str]]:
    """Validate a single publishable package directory under py/packages/*."""
    toml_path = os.path.join(pkg_path, 'pyproject.toml')
    data = load_pyproject(toml_path)
    project = cast(dict[str, object], data.get('project', {}))

    pkg_name = str(project.get('name', '') or os.path.basename(pkg_path))
    pkg_version = str(project.get('version', ''))
    pkg_python = str(project.get('requires-python', ''))
    classifiers = [str(c) for c in cast(list[object], project.get('classifiers', []))]
    dependencies = [str(d) for d in cast(list[object], project.get('dependencies', []))]
    raw_opt_deps = cast(dict[str, list[object]], project.get('optional-dependencies', {}))

    pkg_errors: list[str] = []
    if pkg_version != core_version:
        pkg_errors.append(f"version '{pkg_version}' (expected '{core_version}')")
    if pkg_python != EXPECTED_PYTHON:
        pkg_errors.append(f"requires-python '{pkg_python}' (expected '{EXPECTED_PYTHON}')")
    if not os.path.exists(os.path.join(pkg_path, 'README.md')):
        pkg_errors.append('missing README.md')
    if not os.path.exists(os.path.join(pkg_path, 'LICENSE')):
        pkg_errors.append('missing LICENSE')

    module_name = pkg_name.replace('-', '_')
    py_typed_path = os.path.join(pkg_path, 'src', module_name, 'py.typed')
    if not os.path.isfile(py_typed_path):
        pkg_errors.append(f'missing PEP 561 marker src/{module_name}/py.typed')
    if 'Typing :: Typed' not in classifiers:
        pkg_errors.append("missing 'Typing :: Typed' classifier in pyproject.toml")

    if pkg_name != 'genkit' and not any(_CORE_DEP_RE.match(dep) for dep in dependencies):
        exp_cap = expected_major_cap(core_version, zero_x_only=pkg_name in ZERO_X_ONLY_PACKAGES)
        pkg_errors.append(f"dependencies: missing core 'genkit>={core_version},<{exp_cap}' dependency")

    for spec in dependencies:
        err = _check_intra_repo_spec(pkg_name, 'dependencies', spec, core_version)
        if err:
            pkg_errors.append(err)

    for extra_name in sorted(raw_opt_deps):
        for obj in raw_opt_deps[extra_name]:
            err = _check_intra_repo_spec(pkg_name, f'optional-dependencies.{extra_name}', str(obj), core_version)
            if err:
                pkg_errors.append(err)

    return pkg_name, pkg_version, pkg_errors


def check_workspace(py_dir: str) -> tuple[str, list[tuple[str, str, list[str]]], list[tuple[str, list[str]]]]:
    """Run all consistency checks against a Genkit Python workspace root."""
    core_toml = os.path.join(py_dir, 'packages', 'genkit', 'pyproject.toml')
    core_project = cast(dict[str, object], load_pyproject(core_toml).get('project', {}))
    core_version = str(core_project.get('version', ''))
    if not core_version:
        raise ValueError(f'Could not resolve core genkit version in {core_toml}')

    pkg_results: list[tuple[str, str, list[str]]] = []
    packages_dir = os.path.join(py_dir, 'packages')
    if os.path.isdir(packages_dir):
        for pkg in sorted(os.listdir(packages_dir)):
            pkg_path = os.path.join(packages_dir, pkg)
            toml_path = os.path.join(pkg_path, 'pyproject.toml')
            if not os.path.isdir(pkg_path) or not os.path.exists(toml_path):
                continue
            pkg_results.append(check_package(pkg_path, core_version))

    sample_results: list[tuple[str, list[str]]] = []
    samples_dir = os.path.join(py_dir, 'samples')
    if os.path.isdir(samples_dir):
        for sample in sorted(os.listdir(samples_dir)):
            sample_path = os.path.join(samples_dir, sample)
            toml_path = os.path.join(sample_path, 'pyproject.toml')
            if not os.path.isdir(sample_path) or not os.path.exists(toml_path):
                continue
            sample_project = cast(dict[str, object], load_pyproject(toml_path).get('project', {}))
            sample_name = str(sample_project.get('name', '') or sample)
            sample_python = str(sample_project.get('requires-python', ''))
            sample_errors: list[str] = []
            if sample_python != EXPECTED_PYTHON:
                sample_errors.append(f"requires-python '{sample_python}' (expected '{EXPECTED_PYTHON}')")
            sample_results.append((sample_name, sample_errors))

    return core_version, pkg_results, sample_results


def main(py_dir: str = PY_DIR) -> int:
    """Print workspace consistency report and return exit status code."""
    print(f'{BLUE}=== Genkit Python Consistency Check ==={NC}\n')

    try:
        core_version, pkg_results, sample_results = check_workspace(py_dir)
    except ValueError as exc:
        print(f'{RED}ERROR{NC}: {exc}')
        return 1

    print(f'Core genkit version (source of truth): {GREEN}{core_version}{NC}\n')

    errors = 0
    print(f'\n{YELLOW}Checking Packages (Publishable)...{NC}')
    for pkg_name, pkg_version, pkg_errors in pkg_results:
        if pkg_errors:
            for err in pkg_errors:
                print(f'  {RED}✗{NC} {pkg_name}: {err}')
                errors += 1
        else:
            print(f'  {GREEN}✓{NC} {pkg_name} ({pkg_version})')

    print(f'\n{YELLOW}Checking Samples (Non-Publishable)...{NC}')
    for sample_name, sample_errors in sample_results:
        if sample_errors:
            for err in sample_errors:
                print(f'  {RED}✗{NC} {sample_name}: {err}')
                errors += 1
        else:
            print(f'  {GREEN}✓{NC} {sample_name} (Local Demo)')

    print(f'\n{BLUE}=== Summary ==={NC}')
    if errors > 0:
        print(f'{RED}FAILED{NC}: {errors} consistency errors found.')
        return 1

    print(f'{GREEN}PASSED{NC}: All packages and samples are consistent!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
