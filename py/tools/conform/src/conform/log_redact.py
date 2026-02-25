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

"""Structlog processor that truncates data URIs in log output.

Data URIs (``data:image/png;base64,iVBOR...``) can be 20,000+ characters
long and make debug logs unreadable when multimodal inputs are used.
This module provides a structlog processor and a pure helper function
to recursively truncate them.

Key Concepts (ELI5)::

    ┌──────────────────────┬──────────────────────────────────────────────┐
    │ Concept              │ ELI5 Explanation                             │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ Data URI             │ An image/audio file embedded directly in     │
    │                      │ text as base64.  Can be 20,000+ characters.  │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ Structlog processor  │ A function in a pipeline that transforms     │
    │                      │ every log message before it is printed.      │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ Truncation           │ Replacing the huge base64 payload with a     │
    │                      │ short placeholder like ``...<19850 bytes>``. │
    └──────────────────────┴──────────────────────────────────────────────┘

Data Flow::

    log.debug("request", data={"url": "data:image/png;base64,iVBOR..."})
        │
        ▼
    redact_data_uris_processor  ← inserted into structlog chain
        │
        ▼
    log.debug("request", data={"url": "data:image/png;base64,...<19850 bytes>"})

Usage::

    import structlog
    from conform.log_redact import redact_data_uris_processor

    structlog.configure(
        processors=[
            ...,
            redact_data_uris_processor,
            ...,
        ],
    )
"""

from __future__ import annotations

import re
from typing import Any, Union

# Matches data URIs: data:<mediatype>;base64,<data>
# Captures the prefix (up to and including "base64,") and the base64 payload.
_DATA_URI_RE = re.compile(
    r'(data:[a-zA-Z0-9][a-zA-Z0-9!#$&\-^_.+]*'
    r'/[a-zA-Z0-9][a-zA-Z0-9!#$&\-^_.+]*'
    r'(?:;[a-zA-Z0-9][a-zA-Z0-9!#$&\-^_.+]*=[a-zA-Z0-9][a-zA-Z0-9!#$&\-^_.+]*)*'
    r';base64,)'
    r'([A-Za-z0-9+/=]{100,})'
)

# Truncate if the base64 payload exceeds this many characters.
_TRUNCATE_THRESHOLD = 64


def truncate_data_uri(value: str) -> str:
    """Truncate base64 data URIs in a string.

    Pure function — no side effects.

    Example::

        >>> truncate_data_uri("data:image/png;base64,iVBOR..." + "A" * 20000)
        'data:image/png;base64,...<20004 bytes>'
    """

    def _replace(m: re.Match) -> str:
        prefix = m.group(1)
        payload = m.group(2)
        if len(payload) > _TRUNCATE_THRESHOLD:
            return f'{prefix}...<{len(payload)} bytes>'
        return m.group(0)

    return _DATA_URI_RE.sub(_replace, value)


# Type alias for JSON-like structures that redact_data_uris handles.
_JsonLike = Union[str, dict, list, tuple, int, float, bool, None]  # noqa: UP007


def redact_data_uris(obj: _JsonLike) -> _JsonLike:
    """Recursively truncate data URIs in dicts, lists, and strings.

    Pure function — returns a new structure with truncated URIs.
    The original object is not modified.
    """
    if isinstance(obj, str):
        return truncate_data_uri(obj)
    if isinstance(obj, dict):
        return {k: redact_data_uris(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_data_uris(item) for item in obj)
    return obj


def redact_data_uris_processor(
    logger: object,
    method_name: str,
    event_dict: dict[str, Any],  # noqa: ANN401 - structlog processor protocol requires Any
) -> dict[str, Any]:  # noqa: ANN401 - structlog processor protocol requires Any
    """Structlog processor that truncates data URIs in all event values."""
    return {k: redact_data_uris(v) for k, v in event_dict.items()}
