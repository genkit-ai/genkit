# Copyright 2025 Google LLC
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

"""Path utilities for Genkit trace paths. Format: /{name,t:type,s:subtype}."""

import re
from urllib.parse import quote

_PATH_SEGMENT_RE = re.compile(r'\{([^,}]+),[^}]+\}')


def build_path(
    name: str,
    parent_path: str,
    type_str: str,
    subtype: str | None = None,
) -> str:
    """Build hierarchical path: /{name,t:type,s:subtype}."""
    segment = quote(name, safe='')
    if type_str:
        segment = f'{segment},t:{type_str}'
    if subtype:
        segment = f'{segment},s:{subtype}'
    return f'{parent_path}/{{{segment}}}'


def _has_subtype(segment_inner: str) -> bool:
    parts = segment_inner.split(',')[1:]  # skip name, check annotations only
    return any(p.strip().startswith('s:') for p in parts)


def decorate_path_with_subtype(path: str, subtype: str) -> str:
    """Add subtype to leaf node. Idempotent if subtype already present."""
    if not path or not subtype:
        return path
    start = path.rfind('{')
    if start == -1:
        return path
    end = path.find('}', start)
    if end == -1:
        return path
    inner = path[start + 1 : end]
    if _has_subtype(inner):
        return path
    return f'{path[: start + 1]}{inner},s:{subtype}{path[end:]}'


def to_display_path(qualified_path: str) -> str:
    """Convert /{a,t:flow}/{b,t:step} to 'a > b'."""
    if not qualified_path:
        return ''
    return ' > '.join(_PATH_SEGMENT_RE.findall(qualified_path))
