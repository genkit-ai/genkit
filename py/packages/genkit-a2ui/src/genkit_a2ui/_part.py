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

from collections.abc import Sequence

from genkit._core._model import Part, as_part
from genkit._core._typing import PartData

from ._types import A2UI_MIME_TYPE, Envelope


def a2ui_part(envelopes: list[Envelope]) -> Part:
    return Part.from_data({'envelopes': envelopes}, metadata={'mimeType': A2UI_MIME_TYPE})


def has_a2ui_mime(*, part: Part | PartData) -> bool:
    p = as_part(part)
    if p.data is None:
        return False
    metadata = p.metadata or {}
    return metadata.get('mimeType') == A2UI_MIME_TYPE


def is_a2ui_part(part: Part | PartData) -> bool:
    if not has_a2ui_mime(part=part):
        return False
    p = as_part(part)
    data = p.data
    return isinstance(data, dict) and 'envelopes' in data


def envelopes_from_parts(parts: Sequence[Part | PartData] | None) -> list[Envelope]:
    if not parts:
        return []
    out: list[Envelope] = []
    for raw in parts:
        part = as_part(raw)
        if not is_a2ui_part(part):
            continue
        data = part.data
        if not isinstance(data, dict):
            continue
        raw_envelopes = data.get('envelopes')
        if not isinstance(raw_envelopes, list):
            continue
        for item in raw_envelopes:
            if isinstance(item, dict):
                out.append(item)
    return out
