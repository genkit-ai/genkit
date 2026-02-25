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

"""Path constants for the ``conform`` tool.

Only paths intrinsic to the tool itself are defined here.  All
repository-specific paths (specs directory, plugins directory, working
directory) are supplied via ``pyproject.toml`` configuration or CLI
flags, making the tool fully portable across repositories.
"""

from __future__ import annotations

from pathlib import Path

# tools/conform/src/conform/paths.py  →  src/  →  tools/conform/
PACKAGE_DIR: Path = Path(__file__).resolve().parent
TOOL_DIR: Path = PACKAGE_DIR.parent.parent
