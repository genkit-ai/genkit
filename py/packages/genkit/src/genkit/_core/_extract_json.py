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

"""Utility functions for extracting JSON data from text and markdown."""

import json
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import json5
from partial_json_parser import loads

CHAR_NON_BREAKING_SPACE = '\u00a0'
_SCAN_TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*(?:"|\\?)|[{}[\]\'/]', re.DOTALL)


def parse_partial_json(json_string: str) -> Any:  # noqa: ANN401
    """Parse a partially complete JSON string."""
    return loads(json_string)


def extract_json(text: str, throw_on_bad_json: bool = True) -> Any:  # noqa: ANN401
    """Extract complete JSON from text with lenient parsing (trailing commas, code fences, surrounding prose).

    A finished reply cut off mid-object isn't JSON. A ``{name}`` placeholder
    in the prose is skipped so a later object can still be found.
    """
    return scan_json(text, throw_on_bad_json=throw_on_bad_json, allow_partial=False)


def extract_partial_json(text: str) -> Any:  # noqa: ANN401
    """Extract JSON from streaming text. Never raises.

    Returns complete JSON, a repaired partial object, or None.
    """
    return scan_json(text, throw_on_bad_json=False, allow_partial=True)


def is_complete_json(text: str) -> bool:
    """Return True when ``text`` contains a complete JSON value without repair."""
    if not text.strip():
        return False
    try:
        scan_json(text, throw_on_bad_json=True, allow_partial=False)
    except ValueError:
        return False
    return True


def scan_json(text: str, *, throw_on_bad_json: bool, allow_partial: bool) -> Any:  # noqa: ANN401
    """Scan ``text`` for a JSON value. Used by extract_json and extract_partial_json."""
    stripped = text.strip()
    if not stripped:
        return None

    with suppress(ValueError):
        return json.loads(stripped)

    opening_char: str | None = None
    closing_char: str | None = None
    start_pos: int | None = None
    nesting_count = 0
    saw_json5_syntax = False
    tried_stripped_json5 = False

    for match in _SCAN_TOKEN_RE.finditer(text):
        tok = match.group(0)
        if tok[0] == '"':
            continue

        if opening_char is not None and tok in ("'", '/'):
            saw_json5_syntax = True
            continue

        if not opening_char and tok in '{[':
            opening_char = tok
            closing_char = '}' if tok == '{' else ']'
            start_pos = match.start()
            nesting_count = 1
        elif tok == opening_char:
            nesting_count += 1
        elif tok == closing_char:
            nesting_count -= 1
            if not nesting_count:
                candidate = text[start_pos or 0 : match.end()]
                with suppress(ValueError):
                    return json.loads(candidate)
                try:
                    return json5.loads(candidate)
                except ValueError:
                    if saw_json5_syntax and not tried_stripped_json5:
                        tried_stripped_json5 = True
                        with suppress(ValueError):
                            return json5.loads(stripped)
                    # {name} or a set literal is a balanced group that isn't
                    # JSON; keep looking so surrounding prose doesn't hide the
                    # object they asked for.
                    opening_char = None
                    closing_char = None
                    start_pos = None
                    nesting_count = 0

    if not tried_stripped_json5 and (stripped[0] not in '{[' or saw_json5_syntax):
        with suppress(ValueError):
            return json5.loads(stripped)

    if allow_partial and start_pos is not None and nesting_count > 0:
        try:
            return parse_partial_json(text[start_pos:])
        except Exception as e:
            if throw_on_bad_json:
                raise ValueError(f'Invalid JSON extracted from model output: {text}') from e
            return None

    if throw_on_bad_json:
        raise ValueError(f'Invalid JSON extracted from model output: {text}')
    return None


@dataclass
class ExtractItemsResult:
    """Result of extracting JSON items from text."""

    items: list
    cursor: int


def extract_json_array_from_text(text: str, cursor: int = 0) -> ExtractItemsResult:
    """Extract complete JSON objects from the first array found in text."""
    items: list = []
    current_cursor = cursor

    if cursor == 0:
        array_start = text.find('[')
        if array_start == -1:
            return ExtractItemsResult(items=[], cursor=len(text))
        current_cursor = array_start + 1

    object_start = -1
    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(current_cursor, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == '{':
            if brace_count == 0:
                object_start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and object_start != -1:
                try:
                    items.append(json5.loads(text[object_start : i + 1]))
                    current_cursor = i + 1
                    object_start = -1
                except Exception:  # noqa: S110
                    pass
        elif char == ']' and brace_count == 0:
            break

    return ExtractItemsResult(items=items, cursor=current_cursor)
