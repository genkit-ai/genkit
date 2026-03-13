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

"""``conform check-plugin`` — verify that every model plugin has conformance files.

Mirrors check 21 from ``py/bin/check_consistency`` so the Python tool can
be called from both the shell script and directly.
"""

from __future__ import annotations

import sys

from conform.config import ConformConfig
from conform.plugins import discover_model_plugins, entry_point, spec_file

# ANSI color codes — match the style used by check_consistency.
_RED = '\033[0;31m'
_GREEN = '\033[0;32m'
_NC = '\033[0m'


def check_model_conformance(config: ConformConfig) -> int:
    """Check that every model plugin has a conformance spec and entry point.

    Prints colored output to stdout (same format as ``check_consistency``)
    and returns the number of errors found (0 = success).
    """
    errors = 0
    model_plugins = discover_model_plugins(config)

    for plugin in model_plugins:
        spec = spec_file(plugin, config)
        if not spec.exists():
            sys.stdout.write(
                f'  {_RED}MISSING{_NC}: plugins/{plugin} is a model provider but has no conformance spec\n'
            )
            sys.stdout.write(f'           Expected: {spec}\n')
            errors += 1

        ep = entry_point(plugin, config)
        if not ep.exists():
            sys.stdout.write(
                f'  {_RED}MISSING{_NC}: plugins/{plugin} is a model provider but has no conformance entry point\n'
            )
            sys.stdout.write(f'           Expected: {ep}\n')
            errors += 1

    if errors == 0:
        sys.stdout.write(f'  {_GREEN}✓{_NC} All model plugins have conformance test specs and entry points\n')

    return errors
