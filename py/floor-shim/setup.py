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
failure this shim exists to prevent. Version 0.2.1 sorts above the
pre-transfer squatter releases (<= 0.2.0) and below every real genkit
release (>= 0.3.x); it never needs to change.
"""

import sys

MIN_PYTHON = (3, 10)

if sys.version_info < MIN_PYTHON:
    sys.exit(
        '\n'
        '========================================================================\n'
        'Genkit requires Python {min}+, but you are running Python {cur}.\n'
        '\n'
        'macOS ships an older Python at /usr/bin/python3. To use Genkit,\n'
        'install a current Python first. Any one of these works:\n'
        '\n'
        '  # Homebrew\n'
        '  brew install python@3.13\n'
        '  python3.13 -m venv .venv && source .venv/bin/activate\n'
        '  pip install genkit\n'
        '\n'
        '  # uv (recommended - manages Python for you)\n'
        '  curl -LsSf https://astral.sh/uv/install.sh | sh\n'
        '  uv init && uv add genkit\n'
        '\n'
        '  # Or download an installer from https://www.python.org/downloads/\n'
        '\n'
        'Docs: https://genkit.dev/docs/python/get-started/\n'
        '========================================================================\n'.format(
            min='.'.join(map(str, MIN_PYTHON)),
            cur='.'.join(map(str, sys.version_info[:3])),
        )
    )

from setuptools import setup  # noqa: E402

setup(
    name='genkit',
    version='0.2.1',
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
