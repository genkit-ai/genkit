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

from typing import Any

import json5
from partial_json_parser import loads

CHAR_NON_BREAKING_SPACE = '\u00a0'


def parse_partial_json(json_string: str) -> Any:  # noqa: ANN401
    """Parses a partially complete JSON string and returns the parsed object.

    This function attempts to parse the given JSON string, even if it is not
    a complete or valid JSON document.

    Args:
        json_string: The string to parse as JSON.

    Returns:
        The parsed JSON object.

    Raises:
        AssertionError: If the string cannot be parsed as JSON.
    """
    # TODO(#4350): add handling for malformed JSON cases.
    return loads(json_string)


def extract_json(text: str, throw_on_bad_json: bool = True) -> Any:  # noqa: ANN401
    """Extracts JSON from a string with lenient parsing.

    This function attempts to extract a valid JSON object or array from a
    string, even if the string contains extraneous characters or minor
    formatting issues. It uses a combination of basic parsing and
    `json5` and `partial-json` libraries to maximize the chance of
    successful extraction.

    Args:
        text: The string to extract JSON from.
        throw_on_bad_json: If True, raises a ValueError if no valid JSON
            can be extracted. If False, returns None in such cases.

    Returns:
        The extracted JSON object (dict or list), or None if no valid
        JSON is found and `throw_on_bad_json` is False.

    Raises:
        ValueError: If `throw_on_bad_json` is True and no valid JSON
            can be extracted, or if parsing an incomplete structure fails.

    Examples:
        >>> extract_json('  { "key" : "value" }  ')
        {'key': 'value'}

        >>> extract_json('{"key": "value",}')  # Trailing comma
        {'key': 'value'}

        >>> extract_json('some text {"key": "value"} more text')
        {'key': 'value'}

        >>> extract_json('invalid json', throw_on_bad_json=False)
        None
    """
    if text.strip() == '':
        return None

    # Explicit type annotations for loop variables to help type checkers
    opening_char: str | None = None
    closing_char: str | None = None
    start_pos: int | None = None
    nesting_count: int = 0
    in_string: bool = False
    escape_next: bool = False

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

        if not opening_char and (char == '{' or char == '['):
            # Look for opening character
            opening_char = char
            closing_char = '}' if char == '{' else ']'
            start_pos = i
            nesting_count += 1
        elif char == opening_char:
            # Increment nesting for matching opening character
            nesting_count += 1
        elif char == closing_char:
            # Decrement nesting for matching closing character
            nesting_count -= 1
            if not nesting_count:
                # Reached end of target element
                return json5.loads(text[start_pos or 0 : i + 1])
    if start_pos is not None and nesting_count > 0:
        # If an incomplete JSON structure is detected
        try:
            # Parse the incomplete JSON structure using partial-json for lenient parsing
            return parse_partial_json(text[start_pos:])
        except Exception as e:
            # If parsing fails, throw an error
            if throw_on_bad_json:
                raise ValueError(f'Invalid JSON extracted from model output: {text}') from e
            return None

    if throw_on_bad_json:
        raise ValueError(f'Invalid JSON extracted from model output: {text}')
    return None


class ExtractItemsResult:
    """Holds the result of extracting items from a text array.

    Attributes:
        items: A list of the extracted JSON objects.
        cursor: The index in the original text immediately after the last
                processed character.
    """

    def __init__(self, items: list[Any], cursor: int) -> None:
        """Initialize an ExtractItemsResult.

        Args:
            items: A list of the extracted JSON objects.
            cursor: The updated cursor position.
        """
        self.items: list[Any] = items
        self.cursor: int = cursor


def extract_items(text: str, cursor: int = 0) -> ExtractItemsResult:
    """Extracts complete JSON objects from the first array found in the text.

    This function searches for the first JSON array within the input string,
    starting from an optional cursor position. It extracts complete JSON
    objects from this array and returns them along with an updated cursor
    position, indicating how much of the string has been processed.

    Args:
        text: The string to extract items from.
        cursor: The starting position for searching the array (default: 0).
            Useful for processing large strings in chunks.

    Returns:
        An `ExtractItemsResult` object containing:
          - `items`: A list of extracted JSON objects (dictionaries).
          - `cursor`: The updated cursor position, which is the index
            immediately after the last processed character. If no array is
            found, the cursor will be the length of the text.

    Examples:
        >>> text = '[{"a": 1}, {"b": 2}, {"c": 3}]'
        >>> result = extract_items(text)
        >>> result.items
        [{'a': 1}, {'b': 2}, {'c': 3}]
        >>> result.cursor
        29

        >>> text = '  [ {"x": 10},  {"y": 20} ]  '
        >>> result = extract_items(text)
        >>> result.items
        [{'x': 10}, {'y': 20}]
        >>> result.cursor
        25

        >>> text = 'some text [ {"p": 100} , {"q": 200} ] more text'
        >>> result = extract_items(text, cursor=10)
        >>> result.items
        [{'p': 100}, {'q': 200}]
        >>> result.cursor
        35

        >>> text = 'no array here'
        >>> result = extract_items(text)
        >>> result.items
        []
        >>> result.cursor
        13
    """
    items = []
    current_cursor = cursor

    # Find the first array start if we haven't already processed any text
    if cursor == 0:
        array_start = text.find('[')
        if array_start == -1:
            return ExtractItemsResult(items=[], cursor=len(text))
        current_cursor = array_start + 1

    object_start = -1
    brace_count = 0
    in_string = False
    escape_next = False

    # Process the text from the cursor position
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
                    obj = json5.loads(text[object_start : i + 1])
                    items.append(obj)
                    current_cursor = i + 1
                    object_start = -1
                except Exception:  # noqa: S110 - intentionally silent, parsing partial JSON
                    # If parsing fails, continue trying next position
                    pass
        elif char == ']' and brace_count == 0:
            # End of array
            break

    return ExtractItemsResult(items=items, cursor=current_cursor)
