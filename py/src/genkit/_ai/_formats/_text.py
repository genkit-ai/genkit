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

"""Implementation of text output format."""

from typing import Any

from genkit._ai._formats._types import FormatDef, Formatter, FormatterConfig
from genkit._ai._model import (
    Message,
    ModelResponseChunk,
)
from genkit._core._compat import override


class TextFormat(FormatDef):
    """Defines a text format for use with AI models.

    This is the simplest format, returning the raw text content from the model's response.
    It does not enforce any schema or structural constraints.

    Usage:
        ai.generate(
            output=OutputConfig(format='text')
        )
    """

    def __init__(self) -> None:
        """Initializes a TextFormat instance.

        Configures the format with:
        - name: 'text'
        - content_type: 'text/plain'
        """
        super().__init__(
            'text',
            FormatterConfig(
                content_type='text/plain',
            ),
        )

    @override
    def handle(self, schema: dict[str, object] | None) -> Formatter[Any, Any]:
        """Creates a Formatter for handling text data.

        Args:
            schema: Optional schema (ignored for text).

        Returns:
            A Formatter instance configured for text handling.
        """

        def message_parser(msg: Message) -> str:
            """Extracts text from a Message object.

            Args:
                msg: The Message object.

            Returns:
                The raw text content of the message.
            """
            return msg.text

        def chunk_parser(chunk: ModelResponseChunk) -> str:
            """Extracts text from a ModelResponseChunk object.

            Args:
                chunk: The ModelResponseChunk object.

            Returns:
                The text content from the current chunk only.
            """
            return chunk.text

        return Formatter(
            chunk_parser=chunk_parser,
            message_parser=message_parser,
            instructions=None,
        )
