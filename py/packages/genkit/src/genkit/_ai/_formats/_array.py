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

"""Implementation of Array output format."""

import json
from typing import Any, cast

from genkit._ai._formats._schema import resolve_json_schema_refs
from genkit._ai._formats._types import FormatDef, Formatter, FormatterConfig
from genkit._ai._model import (
    Message,
    ModelResponseChunk,
)
from genkit._core._compat import override
from genkit._core._error import GenkitError
from genkit._core._extract_json import extract_json_array_from_text


class ArrayFormat(FormatDef):
    """Defines an Array format for use with AI models.

    This format instructs the model to output a JSON array of items matching a specified schema.
    It is useful for generating lists of structured objects.

    The formatter automatically handles:
    1.  Validating that the schema is of type `array`.
    2.  Injecting instructions into the prompt to output a JSON array.
    3.  Parsing the response (both full messages and streaming chunks) using `extract_json_array_from_text`
        to recover valid JSON objects from potentially incomplete or noisy output.

    Usage:
        ai.generate(
            output=OutputConfig(
                format='array',
                schema={
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {'name': {'type': 'string'}}
                    }
                }
            )
        )
    """

    def __init__(self) -> None:
        """Initializes the ArrayFormat.

        Configures the format with:
        - name: 'array'
        - content_type: 'application/json'
        - constrained: True
        """
        super().__init__(
            'array',
            FormatterConfig(
                content_type='application/json',
                constrained=True,
            ),
        )

    @override
    def handle(self, schema: dict[str, object] | None) -> Formatter[Any, Any]:
        """Creates a Formatter for handling JSON array data.

        Args:
            schema: The JSON schema for the array. Must be of type 'array'.

        Returns:
            A Formatter configured to parse JSON arrays.

        Raises:
            GenkitError: If the schema is missing or not of type 'array'.
        """
        resolved_schema = cast(dict[str, object], resolve_json_schema_refs(schema, schema)) if schema else None

        if resolved_schema and resolved_schema.get('type') != 'array':
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message="Must supply an 'array' schema type when using the 'items' parser format.",
            )

        def message_parser(msg: Message) -> list[object]:
            """Parses a complete message into a list of items."""
            result = extract_json_array_from_text(msg.text, 0)
            return result.items

        def chunk_parser(chunk: ModelResponseChunk) -> list[object]:
            """Parses a streaming chunk into a list of items."""
            # Calculate the length of text from previous chunks
            previous_text_len = len(chunk.accumulated_text) - len(chunk.text)

            # Find cursor position from previous text
            cursor = 0
            if previous_text_len > 0:
                cursor = extract_json_array_from_text(chunk.accumulated_text[:previous_text_len]).cursor

            result = extract_json_array_from_text(chunk.accumulated_text, cursor)
            return result.items

        instructions = None
        if resolved_schema:
            instructions = f"""Output should be a JSON array conforming to the following schema:

```
{json.dumps(resolved_schema, indent=2)}
```
"""
        return Formatter(
            chunk_parser=chunk_parser,
            message_parser=message_parser,
            instructions=instructions,
        )
