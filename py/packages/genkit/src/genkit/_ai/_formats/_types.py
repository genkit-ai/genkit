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

"""Output format types: Formatter, FormatterConfig, FormatHandler."""

import abc
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from genkit._ai._model import (
    Message,
    ModelResponseChunk,
)
from genkit._core._base import GenkitModel


class FormatterConfig(GenkitModel):
    """Request settings a Formatter applies to every generate call that uses it.

    `format` and `content_type` replace the request's values; `constrained` is a default
    the request can override. `default_instructions=False` skips the handler's instructions.
    """

    format: str | None = None
    content_type: str | None = None
    constrained: bool | None = None
    default_instructions: bool | str | None = None


OutputT = TypeVar('OutputT')
ChunkT = TypeVar('ChunkT')


class FormatHandler(Generic[OutputT, ChunkT]):
    """Per-request parsers and prompt instructions returned by `Formatter.handle()`.

    `message_parser` builds `response.output`; `chunk_parser` builds `chunk.output`.
    """

    def __init__(
        self,
        message_parser: Callable[[Message], OutputT],
        chunk_parser: Callable[[ModelResponseChunk], ChunkT],
        instructions: str | None,
    ) -> None:
        """Initializes a FormatHandler.

        Args:
            message_parser: Parses the final model message into the output value.
            chunk_parser: Parses a streamed chunk into the chunk output value.
            instructions: Text appended to the prompt, or None for no instructions.
        """
        self.instructions: str | None = instructions
        self.__message_parser = message_parser
        self.__chunk_parser = chunk_parser

    def parse_message(self, message: Message) -> OutputT:
        """Parses the final model message."""
        return self.__message_parser(message)

    def parse_chunk(self, chunk: ModelResponseChunk) -> ChunkT:
        """Parses one streamed chunk."""
        return self.__chunk_parser(chunk)


class Formatter:
    """A named output format. Subclass it, implement `handle()`, register with `ai.define_format()`.

    `handle()` runs once per generate call with the request's JSON schema (or None).

    Example:
        ```python
        import csv
        import io

        from genkit import Genkit
        from genkit.formats import FormatHandler, Formatter, FormatterConfig
        from genkit_google_genai import GoogleAI


        class CsvFormat(Formatter):
            def __init__(self) -> None:
                super().__init__('csv', FormatterConfig(content_type='text/csv'))

            def handle(self, schema: dict[str, object] | None) -> FormatHandler:
                return FormatHandler(
                    message_parser=lambda msg: list(csv.DictReader(io.StringIO(msg.text))),
                    chunk_parser=lambda chunk: chunk.text,
                    instructions='Reply only with CSV. First row is the header.',
                )


        # 1. Register the format
        ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')
        ai.define_format(CsvFormat())

        # 2. Request it by name
        response = await ai.generate(prompt='List 2 dishes with price.', output_format='csv')

        # 3. response.output is what message_parser returned
        print(response.output)
        # => [{'dish': 'Ramen', 'price': '14.50'}, {'dish': 'Pho', 'price': '13.00'}]
        ```
    """

    def __init__(self, name: str, config: FormatterConfig) -> None:
        """Initializes a Formatter.

        Args:
            name: Name passed as `output_format=` to select this format.
            config: Request settings applied when this format is selected.
        """
        self.name: str = name
        self.config: FormatterConfig = config

    @abc.abstractmethod
    def handle(self, schema: dict[str, object] | None) -> FormatHandler[Any, Any]:
        """Returns the FormatHandler for one generate call.

        Args:
            schema: The request's output JSON schema, or None.
        """
        pass

    def __call__(self, schema: dict[str, object] | None) -> FormatHandler[Any, Any]:
        """Same as `handle(schema)`."""
        return self.handle(schema)
