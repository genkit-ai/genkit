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

"""Incremental extractor for ```a2ui fenced blocks."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from genkit._core._logger import get_logger

from ._catalog import A2uiCatalog
from ._types import DEFAULT_VERSION, SURFACE_ID_PLACEHOLDER, SURFACE_KEYS, Envelope, ValidateMode

logger = get_logger(__name__)

OPEN_FENCE_RE = re.compile(r'(?i)```a2ui[ \t]*\r?\n')
# Hold back a trailing `, ``, or incomplete ```a2ui so a fence split across
# chunks is not leaked as prose.
PARTIAL_OPEN_FENCE_RE = re.compile(r'(?i)(?:`|``|```(?:a(?:2(?:u(?:i[ \t]*\r?)?)?)?)?)$')
CLOSE_FENCE_RE = re.compile(r'^[ \t]*```', re.MULTILINE)
LEADING_NEWLINE_RE = re.compile(r'^[ \t]*\r?\n')


@dataclass
class Segment:
    """Prose, or a finished A2UI block (`envelopes` set)."""

    prose: str = ''
    envelopes: list[Envelope] = field(default_factory=list)


@dataclass
class ClosedBlock:
    """A finished fence, raw prose to keep, or `need_more` if the closing ``` has not arrived."""

    need_more: bool = False
    envelopes: list[Envelope] | None = None
    prose: str = ''


class A2uiParseError(ValueError):
    """Raised in strict mode when a fence is malformed or names an unknown component."""


class StreamParser:
    def __init__(
        self,
        *,
        catalog: A2uiCatalog | None,
        validate: ValidateMode,
        version: str,
        surface_id: Callable[[], str],
    ) -> None:
        self.catalog = catalog
        self.validate = validate or 'warn'
        self.version = version or DEFAULT_VERSION
        self.surface_id = surface_id
        self.buffer = ''
        self.in_block = False
        self.current_surface_id = ''
        self.block_scan = 0
        self.known_components = {c.name for c in catalog.components} if catalog is not None else set()

    def push(self, *, text: str) -> list[Segment]:
        self.buffer += text
        return self.drain(final=False)

    def flush(self) -> list[Segment]:
        return self.drain(final=True)

    def drain(self, *, final: bool) -> list[Segment]:
        segments: list[Segment] = []
        prose_buf = ''

        def flush_prose() -> None:
            nonlocal prose_buf
            if prose_buf:
                segments.append(Segment(prose=prose_buf))
                prose_buf = ''

        while True:
            if not self.in_block:
                prefix = self.take_open_fence()
                if prefix is None:
                    if final:
                        prose_buf += self.buffer
                        self.buffer = ''
                    else:
                        prose_buf += self.take_safe_prose()
                    break
                prose_buf += prefix
                continue

            taken = self.take_closed_block(final=final)
            if taken.need_more:
                break
            if taken.envelopes:
                flush_prose()
                segments.append(Segment(envelopes=taken.envelopes))
            elif taken.prose:
                prose_buf += taken.prose
        flush_prose()
        return segments

    def take_open_fence(self) -> str | None:
        match = OPEN_FENCE_RE.search(self.buffer)
        if match is None:
            return None
        prefix = self.buffer[: match.start()]
        self.buffer = self.buffer[match.end() :]
        self.in_block = True
        self.block_scan = 0
        self.current_surface_id = self.surface_id()
        return prefix

    def take_safe_prose(self) -> str:
        keep = 0
        partial = PARTIAL_OPEN_FENCE_RE.search(self.buffer)
        if partial is not None:
            keep = len(self.buffer) - partial.start()
        safe_len = len(self.buffer) - keep
        if safe_len <= 0:
            return ''
        taken = self.buffer[:safe_len]
        self.buffer = self.buffer[safe_len:]
        return taken

    def take_closed_block(self, *, final: bool) -> ClosedBlock:
        match = CLOSE_FENCE_RE.search(self.buffer[self.block_scan :])
        if match is None:
            if not final:
                nl = self.buffer.rfind('\n')
                if nl + 1 > self.block_scan:
                    self.block_scan = nl + 1
                return ClosedBlock(need_more=True)
            raw = self.buffer
            result = self.finalize_block(raw=raw)
            self.buffer = ''
            self.in_block = False
            self.block_scan = 0
            return closed_block(result=result, closed=False)

        match_start = self.block_scan + match.start()
        match_end = self.block_scan + match.end()
        block_text = self.buffer[:match_start]
        self.buffer = LEADING_NEWLINE_RE.sub('', self.buffer[match_end:], count=1)
        self.in_block = False
        self.block_scan = 0
        return closed_block(result=self.finalize_block(raw=block_text), closed=True)

    def reject(self, *, message: str) -> None:
        full = f'A2UI: {message}'
        if self.validate == 'off':
            return
        if self.validate == 'strict':
            raise A2uiParseError(full)
        logger.warning('%s (dropping block/envelope)', full)

    def finalize_block(self, *, raw: str) -> list[Envelope] | str | None:
        surface_id = self.current_surface_id or self.surface_id()
        self.current_surface_id = ''

        text = raw.strip()
        if not text:
            return None
        try:
            parsed: object = json.loads(text)
        except json.JSONDecodeError as exc:
            self.reject(message=f'failed to parse envelope block as JSON: {exc}')
            if self.validate == 'strict':
                return None
            # warn / off: the caller still has the model text, even if it is not a card.
            return raw

        raw_envelopes = parsed if isinstance(parsed, list) else [parsed]
        out: list[Envelope] = []
        for env in raw_envelopes:
            normalized = self.normalize_envelope(env=env, surface_id=surface_id)
            if normalized is not None:
                out.append(normalized)
        if not out:
            return None

        if any('createSurface' in e for e in out):
            # New card. Stamp the minted id so we don't reuse one the model
            # copied from a prior surface in history.
            for e in out:
                force_surface_id(envelope=e, surface_id=surface_id)
            return self.require_root(envelopes=out)

        existing_id = next((envelope_surface_id(envelope=e) for e in out if envelope_surface_id(envelope=e)), '')
        if existing_id and existing_id != surface_id:
            # Update to a surface the model already knows. Don't invent a
            # createSurface — that would wipe the card the click landed on.
            return out

        # New card that omitted createSurface. Add one so the client has a
        # surface to paint before the updates land.
        if self.require_root(envelopes=out) is None:
            return None
        catalog_id = self.catalog.id if self.catalog is not None else ''
        create: Envelope = {
            'version': self.version,
            'createSurface': {'surfaceId': surface_id, 'catalogId': catalog_id},
        }
        return [create, *out]

    def require_root(self, *, envelopes: list[Envelope]) -> list[Envelope] | None:
        if self.validate == 'off':
            return envelopes
        msg = validate_root(envelopes=envelopes)
        if msg:
            self.reject(message=msg)
            return None
        return envelopes

    def normalize_envelope(self, *, env: object, surface_id: str) -> Envelope | None:
        if not isinstance(env, dict):
            self.reject(message='envelope must be an object.')
            return None
        payload = cast(dict[str, Any], env)
        raw_version = payload.get('version')
        version = raw_version if isinstance(raw_version, str) and raw_version else self.version

        for key in SURFACE_KEYS:
            body = payload.get(key)
            if not isinstance(body, dict):
                continue
            fill_placeholder_id(body=body, surface_id=surface_id)
            if key == 'updateComponents' and self.validate != 'off':
                err = self.validate_components(components=body.get('components'))
                if err:
                    self.reject(message=err)
                    return None
            out = dict(payload)
            out['version'] = version
            return out

        keys = ', '.join(str(k) for k in payload)
        self.reject(message=f'unknown envelope type (keys: {keys}).')
        return None

    def validate_components(self, *, components: object) -> str:
        if self.catalog is None:
            return ''
        if not isinstance(components, list):
            return 'updateComponents.components must be an array.'
        for item in components:
            row = cast(dict[str, Any], item) if isinstance(item, dict) else None
            name = row.get('component') if row is not None else None
            if not isinstance(name, str):
                return 'every component needs a "component" type name.'
            if name not in self.known_components:
                return f'component {name!r} is not in catalog {self.catalog.id!r}.'
        return ''


def closed_block(*, result: list[Envelope] | str | None, closed: bool) -> ClosedBlock:
    if isinstance(result, list):
        return ClosedBlock(envelopes=result)
    if isinstance(result, str):
        fence = f'```a2ui\n{result}'
        if closed:
            fence += '\n```'
        return ClosedBlock(prose=fence)
    return ClosedBlock()


def fill_placeholder_id(*, body: dict[str, object], surface_id: str) -> None:
    sid = body.get('surfaceId')
    if not isinstance(sid, str) or sid == '' or sid == SURFACE_ID_PLACEHOLDER:
        body['surfaceId'] = surface_id


def force_surface_id(*, envelope: Envelope, surface_id: str) -> None:
    for key in SURFACE_KEYS:
        payload = envelope.get(key)
        if isinstance(payload, dict):
            payload['surfaceId'] = surface_id
            return


def envelope_surface_id(*, envelope: Envelope) -> str:
    for key in SURFACE_KEYS:
        payload = envelope.get(key)
        if isinstance(payload, dict) and isinstance(payload.get('surfaceId'), str) and payload['surfaceId']:
            return payload['surfaceId']
    return ''


def validate_root(*, envelopes: list[Envelope]) -> str:
    saw_list = False
    for e in envelopes:
        uc = e.get('updateComponents')
        if not isinstance(uc, dict):
            continue
        arr = uc.get('components')
        if not isinstance(arr, list):
            continue
        saw_list = True
        for item in arr:
            if isinstance(item, dict) and item.get('id') == 'root':
                return ''
    if not saw_list:
        return ''
    return 'component list must contain a component id "root".'
