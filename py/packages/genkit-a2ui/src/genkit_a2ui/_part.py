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

"""A2UI data-part helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from genkit import Part

from ._types import A2UI_MIME_TYPE, Envelope


def a2ui_part(envelopes: list[Envelope]) -> Part:
    """Wraps envelopes in an A2UI data part.

    The list sits under the A2UI extension spec's own ``messages`` wrapper
    (``{"messages": [...]}``), the same key the Dart, JS and Go plugins use.
    """
    return Part.from_data({'messages': envelopes}, metadata={'mimeType': A2UI_MIME_TYPE})


def has_a2ui_mime(*, part: Part) -> bool:
    if part.data is None:
        return False
    metadata = part.metadata or {}
    return metadata.get('mimeType') == A2UI_MIME_TYPE


def _messages_of(part: Part) -> list[Any] | None:
    """The message list in an A2UI part's data.

    ``messages``, or ``envelopes`` for a part written before the rename (still
    read, so persisted sessions and older clients keep working). A
    ``messages`` key that is present but ``None`` falls back too.
    """
    data = part.data
    if not isinstance(data, dict):
        return None
    raw = data.get('messages')
    if raw is None:
        raw = data.get('envelopes')
    return raw if isinstance(raw, list) else None


def is_a2ui_part(part: Part) -> bool:
    return has_a2ui_mime(part=part) and _messages_of(part) is not None


def envelopes_from_parts(parts: Sequence[Part] | None) -> list[Envelope]:
    if not parts:
        return []
    out: list[Envelope] = []
    for part in parts:
        if not has_a2ui_mime(part=part):
            continue
        for item in _messages_of(part) or []:
            if isinstance(item, dict):
                out.append(item)
    return out
