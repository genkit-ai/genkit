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

"""Implementation of JSON output format."""

import json
from typing import Any

from genkit._ai._formats._types import FormatDef, Formatter, FormatterConfig
from genkit._ai._model import (
    Message,
    ModelResponseChunk,
)
from genkit._core._compat import override
from genkit._core._extract_json import extract_json


class JsonFormat(FormatDef):
    """Defines a JSON format for use with AI models.

    This format instructs the model to output a valid JSON object. It is the default format
    when a JSON schema is provided in the configuration.

    The formatter handles:
    1.  Injecting instructions with the JSON schema.
    2.  Parsing the response using `extract_json` to handle potentially noisy output (e.g. markdown code blocks).

    Usage:
        ai.generate(
            output=OutputConfig(
                format='json',
                schema={'type': 'object', 'properties': {'foo': {'type': 'string'}}}
            )
        )
    """

    def __init__(self) -> None:
        """Initializes a JsonFormat instance.

        Sets up the format definition with configurations suitable for JSON,
        including content type, constraints, and default instructions.
        """
        super().__init__(
            'json',
            FormatterConfig(
                format='json',
                content_type='application/json',
                constrained=True,
                default_instructions=False,
            ),
        )

    @override
    def handle(self, schema: dict[str, object] | None) -> Formatter[Any, Any]:
        """Creates a Formatter for handling JSON data based on an optional schema.

        Args:
            schema: An optional dictionary representing the JSON schema.
                    If provided, the formatter will ensure that the output
                    conforms to this schema.

        Returns:
            A Formatter instance configured for JSON handling, including
            parsers for messages and chunks, and instructions derived from
            the provided schema.
        """

        def message_parser(msg: Message) -> object:
            """Extracts JSON from a Message object.

            Concatenates the text content of all parts in the message and
            attempts to extract a JSON object from the resulting string.

            Args:
                msg: The Message object to parse.

            Returns:
                A JSON object extracted from the message content.
            """
            return extract_json(msg.text)

        def chunk_parser(chunk: ModelResponseChunk) -> object:
            """Extracts JSON from a ModelResponseChunk object.

            Extracts a JSON object from the accumulated text in the given chunk.
            Returns None if no valid JSON is found yet (common during streaming
            when receiving preamble text).

            Args:
                chunk: The ModelResponseChunk object to parse.

            Returns:
                A JSON object extracted from the chunk's accumulated text,
                or None if no valid JSON is found.
            """
            return extract_json(chunk.accumulated_text, throw_on_bad_json=False)

        instructions: str | None = None

        if schema:
            instructions = f"""\
Output should be in JSON format and conform to the following schema:

```
{json.dumps(schema, indent=2)}
```
"""

        return Formatter(
            chunk_parser=chunk_parser,
            message_parser=message_parser,
            instructions=instructions,
        )
