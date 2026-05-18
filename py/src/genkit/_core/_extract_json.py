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

from dataclasses import dataclass
from typing import Any

import json5
from partial_json_parser import loads

CHAR_NON_BREAKING_SPACE = '\u00a0'


def parse_partial_json(json_string: str) -> Any:  # noqa: ANN401
    """Parse a partially complete JSON string."""
    return loads(json_string)


def extract_json(text: str, throw_on_bad_json: bool = True) -> Any:  # noqa: ANN401
    """Extract JSON from text with lenient parsing (handles trailing commas, partial JSON, etc.)."""
    if not text.strip():
        return None

    opening_char: str | None = None
    closing_char: str | None = None
    start_pos: int | None = None
    nesting_count = 0
    in_string = False
    escape_next = False

    for i in range(len(text)):
        char = text[i].replace(CHAR_NON_BREAKING_SPACE, ' ')

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

        if not opening_char and char in '{[':
            opening_char = char
            closing_char = '}' if char == '{' else ']'
            start_pos = i
            nesting_count += 1
        elif char == opening_char:
            nesting_count += 1
        elif char == closing_char:
            nesting_count -= 1
            if not nesting_count:
                return json5.loads(text[start_pos or 0 : i + 1])

    # Handle incomplete JSON structure
    if start_pos is not None and nesting_count > 0:
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
