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

"""Python-floor shim for the `genkit` package on PyPI.

Real genkit releases require Python >= 3.10. On older interpreters
(like the Python 3.9 that macOS ships), pip either resolves nothing or
resolves an unrelated pre-transfer package that holds low version
numbers on PyPI. This sdist declares `Requires-Python: <3.10`, so
current interpreters can never select it, while old interpreters pick
it as their best candidate and get a clear how-to-upgrade message
instead of a resolver error.

Publish as sdist ONLY (`uv build --sdist`). A wheel would install
"successfully" as an empty package on old Pythons, which is the silent
failure this shim exists to prevent. Version 0.3.0 was never published
(the history goes 0.3.0.dev2 -> 0.3.1), so the shim fills that hole:
above the pre-transfer squatter releases (<= 0.2.0), below every real
genkit release (>= 0.3.1). It never needs to change.
"""

import sys

MIN_PYTHON = (3, 10)

# Fail at the build/install phase, not the metadata phase: old pip (e.g. the
# 21.2.4 bundled with macOS system Python) treats a metadata-build failure as
# a discardable candidate and backtracks to the pre-transfer 0.2.0 — the
# exact silent wrong-install this shim exists to prevent. An install-phase
# failure is fatal on every pip generation, so the user actually sees the
# message and nothing gets installed.
_METADATA_ONLY_COMMANDS = {'egg_info', 'dist_info', 'sdist'}


def _format_error_message(min_ver: tuple[int, ...], cur_ver: tuple[int, ...]) -> str:
    min_str = '.'.join(map(str, min_ver))
    cur_str = '.'.join(map(str, cur_ver[:3]))

    if sys.platform == 'win32':
        platform_hint = f'Windows requires Python {min_str}+ to run Genkit.'
        recipes = (
            '  # Option 1 (Recommended): uv (manages Python versions automatically)\n'
            '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"\n'
            '  uv init && uv add genkit\n'
            '\n'
            f'  # Option 2: Download Python (>= {min_str}) from https://www.python.org/downloads/\n'
            '  python -m venv .venv && .venv\\Scripts\\pip install genkit'
        )
    elif sys.platform == 'darwin':
        platform_hint = (
            'macOS ships an older Python at /usr/bin/python3.\n'
            f'To use Genkit, install Python {min_str}+ using one of the options below:'
        )
        recipes = (
            '  # Option 1 (Recommended): uv (manages Python for you)\n'
            '  curl -LsSf https://astral.sh/uv/install.sh | sh\n'
            '  uv init && uv add genkit\n'
            '\n'
            '  # Option 2: Homebrew\n'
            '  brew install python\n'
            '  python3 -m venv .venv && source .venv/bin/activate\n'
            '  pip install genkit\n'
            '\n'
            '  # Option 3: Download installer from https://www.python.org/downloads/'
        )
    else:
        platform_hint = (
            f'Your system Python is older than the required Python {min_str}+.\n'
            f'To use Genkit, install Python {min_str}+ using one of the options below:'
        )
        recipes = (
            '  # Option 1 (Recommended): uv (standalone, no root required)\n'
            '  curl -LsSf https://astral.sh/uv/install.sh | sh\n'
            '  uv init && uv add genkit\n'
            '\n'
            '  # Option 2: System Package Manager\n'
            '  sudo apt update && sudo apt install python3 python3-venv\n'
            '  python3 -m venv .venv && source .venv/bin/activate\n'
            '  pip install genkit\n'
            '\n'
            '  # Option 3: Download installer from https://www.python.org/downloads/'
        )

    return (
        '\n'
        '========================================================================\n'
        f'Genkit requires Python {min_str}+, but you are running Python {cur_str}.\n'
        '\n'
        f'{platform_hint}\n'
        '\n'
        f'{recipes}\n'
        '\n'
        'Docs: https://genkit.dev/docs/python/get-started/\n'
        '========================================================================\n'
    )


if sys.version_info < MIN_PYTHON and not _METADATA_ONLY_COMMANDS.intersection(sys.argv[1:]):
    sys.exit(_format_error_message(MIN_PYTHON, sys.version_info))

from setuptools import setup  # noqa: E402

setup(
    name='genkit',
    version='0.3.0',
    description=(
        'Genkit requires Python >= 3.10. This placeholder release exists only '
        'to give older interpreters a clear upgrade message.'
    ),
    long_description=(
        'Genkit is an open-source framework for building AI-powered '
        'applications, from Google. It requires Python 3.10 or newer. '
        'This placeholder release exists only so that installs on older '
        'Python versions fail with a clear message instead of a resolver '
        'error. See https://genkit.dev/docs/python/get-started/'
    ),
    long_description_content_type='text/plain',
    url='https://genkit.dev',
    author='Google',
    license='Apache-2.0',
    python_requires='<3.10',
    py_modules=[],
)
