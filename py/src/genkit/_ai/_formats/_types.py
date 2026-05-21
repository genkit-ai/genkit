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

"""Format definition classes."""

import abc
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from genkit._ai._model import (
    Message,
    ModelResponseChunk,
)
from genkit._core._base import GenkitModel


class FormatterConfig(GenkitModel):
    """SDK configuration for output formatters (format, content_type, etc.).

    Used by format definitions (json, array, enum, etc.) - not the schema type.
    """

    format: str | None = None
    content_type: str | None = None
    constrained: bool | None = None
    default_instructions: bool | str | None = None


OutputT = TypeVar('OutputT')
ChunkT = TypeVar('ChunkT')


class Formatter(Generic[OutputT, ChunkT]):
    """Base class representing a formatter for model outputs.

    Formatters are responsible for parsing raw model messages and chunks
    into structured data (types OutputT and ChunkT respectively) and potentially
    providing instructions to the model on how to format its output.
    """

    def __init__(
        self,
        message_parser: Callable[[Message], OutputT],
        chunk_parser: Callable[[ModelResponseChunk], ChunkT],
        instructions: str | None,
    ) -> None:
        """Initializes a Formatter.

        Args:
            message_parser: A callable that parses a Message into type OutputT.
            chunk_parser: A callable that parses a ModelResponseChunk into type ChunkT.
            instructions: Optional instructions for the formatter.
        """
        self.instructions: str | None = instructions
        self.__message_parser = message_parser
        self.__chunk_parser = chunk_parser

    def parse_message(self, message: Message) -> OutputT:
        """Parses a message.

        Args:
            message: The message to parse.

        Returns:
            The parsed message.
        """
        return self.__message_parser(message)

    def parse_chunk(self, chunk: ModelResponseChunk) -> ChunkT:
        """Parses a chunk.

        Args:
            chunk: The chunk to parse.

        Returns:
            The parsed chunk.
        """
        return self.__chunk_parser(chunk)


class FormatDef:
    """Represents the definition of a specific output format.

    This class holds the name and configuration for a format and provides
    a method (`handle`) to create a specific Formatter instance based on
    an optional schema.
    """

    def __init__(self, name: str, config: FormatterConfig) -> None:
        """Initializes a FormatDef.

        Args:
            name: The name of the format.
            config: The configuration for the format.
        """
        self.name: str = name
        self.config: FormatterConfig = config

    @abc.abstractmethod
    def handle(self, schema: dict[str, object] | None) -> Formatter[Any, Any]:
        """Handles the format.

        Args:
            schema: Optional schema for the format.

        Returns:
            A Formatter instance.
        """
        pass

    def __call__(self, schema: dict[str, object] | None) -> Formatter[Any, Any]:
        """Calls the handle method.

        Args:
            schema: Optional schema for the format.

        Returns:
            A Formatter instance.
        """
        return self.handle(schema)
