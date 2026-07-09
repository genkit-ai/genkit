#!/usr/bin/env python3
# ruff: noqa
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

import os
import re
import sys

# Color formatting
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_PYTHON = '>=3.10'


def get_toml_value(filepath: str, key: str) -> str:
    """Extract a key value from a TOML file."""
    if not os.path.exists(filepath):
        return ''
    if sys.version_info >= (3, 11):
        try:
            import tomllib

            with open(filepath, 'rb') as f:
                data = tomllib.load(f)
                if key == 'version':
                    return data.get('project', {}).get('version') or data.get('version', '')
                elif key == 'requires-python':
                    return data.get('project', {}).get('requires-python') or data.get('requires-python', '')
                return str(data.get('project', {}).get(key) or data.get(key, ''))
        except Exception:
            pass

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(rf'^\s*{key}\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else ''


def main() -> None:
    print(f'{BLUE}=== Genkit Python Consistency Check ==={NC}\n')

    # 1. Get core version (source of truth)
    core_toml = os.path.join(PY_DIR, 'packages', 'genkit', 'pyproject.toml')
    core_version = get_toml_value(core_toml, 'version')
    if not core_version:
        print(f'{RED}ERROR{NC}: Could not resolve core genkit version in {core_toml}')
        sys.exit(1)

    print(f'Core genkit version (source of truth): {GREEN}{core_version}{NC}\n')

    errors = 0

    # 2. Check publishable packages under packages/*
    packages_dir = os.path.join(PY_DIR, 'packages')
    print(f'\n{YELLOW}Checking Packages (Publishable)...{NC}')
    for pkg in sorted(os.listdir(packages_dir)):
        pkg_path = os.path.join(packages_dir, pkg)
        toml_path = os.path.join(pkg_path, 'pyproject.toml')
        if not os.path.isdir(pkg_path) or not os.path.exists(toml_path):
            continue

        pkg_name = get_toml_value(toml_path, 'name')
        pkg_version = get_toml_value(toml_path, 'version')
        pkg_python = get_toml_value(toml_path, 'requires-python')

        pkg_errors = 0
        if pkg_version != core_version:
            print(f"  {RED}✗{NC} {pkg_name}: version '{pkg_version}' (expected '{core_version}')")
            errors += 1
            pkg_errors += 1
        if pkg_python != EXPECTED_PYTHON:
            print(f"  {RED}✗{NC} {pkg_name}: requires-python '{pkg_python}' (expected '{EXPECTED_PYTHON}')")
            errors += 1
            pkg_errors += 1
        if not os.path.exists(os.path.join(pkg_path, 'README.md')):
            print(f'  {RED}✗{NC} {pkg_name}: missing README.md')
            errors += 1
            pkg_errors += 1
        if not os.path.exists(os.path.join(pkg_path, 'LICENSE')):
            print(f'  {RED}✗{NC} {pkg_name}: missing LICENSE')
            errors += 1
            pkg_errors += 1

        if pkg_errors == 0:
            print(f'  {GREEN}✓{NC} {pkg_name} ({pkg_version})')

    # 3. Check samples under samples/* (Non-Publishable)
    samples_dir = os.path.join(PY_DIR, 'samples')
    print(f'\n{YELLOW}Checking Samples (Non-Publishable)...{NC}')
    for sample in sorted(os.listdir(samples_dir)):
        sample_path = os.path.join(samples_dir, sample)
        toml_path = os.path.join(sample_path, 'pyproject.toml')
        if not os.path.isdir(sample_path) or not os.path.exists(toml_path):
            continue

        sample_name = get_toml_value(toml_path, 'name')
        sample_python = get_toml_value(toml_path, 'requires-python')

        if sample_python != EXPECTED_PYTHON:
            print(f"  {RED}✗{NC} {sample_name}: requires-python '{sample_python}' (expected '{EXPECTED_PYTHON}')")
            errors += 1
        else:
            print(f'  {GREEN}✓{NC} {sample_name} (Local Demo)')

    print(f'\n{BLUE}=== Summary ==={NC}')
    if errors > 0:
        print(f'{RED}FAILED{NC}: {errors} consistency errors found.')
        sys.exit(1)
    else:
        print(f'{GREEN}PASSED{NC}: All packages and samples are consistent!')
        sys.exit(0)


if __name__ == '__main__':
    main()
