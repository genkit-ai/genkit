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

"""Model type definitions for the Genkit framework.

This module defines the type interfaces for AI models in the Genkit framework.
These types ensure consistent interaction with different AI models and provide
type safety when working with model inputs and outputs.

Example:
    def my_model(request: GenerateRequest) -> GenerateResponse:
        # Model implementation
        return GenerateResponse(...)

    model_fn: ModelFn = my_model
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from functools import cached_property
from typing import Any, Generic, cast

from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import TypeVar

from genkit.core.action import ActionMetadata, ActionRunContext
from genkit.core.action.types import ActionKind
from genkit.core.extract import extract_json
from genkit.core.schema import to_json_schema
from genkit.core.typing import (
    Candidate,
    DocumentPart,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    GenerationUsage,
    Media,
    MediaModel,
    Message,
    ModelInfo,
    Part,
    Text,
    ToolRequestPart,
)

# type ModelFn = Callable[[GenerateRequest], GenerateResponse]
ModelFn = Callable[[GenerateRequest, ActionRunContext], GenerateResponse]

# type ModelMiddlewareNext = Callable[[GenerateRequest, ActionRunContext], Awaitable[GenerateResponse]]
ModelMiddlewareNext = Callable[[GenerateRequest, ActionRunContext], Awaitable[GenerateResponse]]
# type ModelMiddleware = Callable[
#     [GenerateRequest, ActionRunContext, ModelMiddlewareNext],
#     Awaitable[GenerateResponse],
# ]
ModelMiddleware = Callable[
    [GenerateRequest, ActionRunContext, ModelMiddlewareNext],
    Awaitable[GenerateResponse],
]

# TypeVar for generic output type in GenerateResponseWrapper
OutputT = TypeVar('OutputT', default=object)


class ModelReference(BaseModel):
    """Reference to a model with configuration."""

    name: str
    config_schema: object | None = None
    info: ModelInfo | None = None
    version: str | None = None
    config: dict[str, object] | None = None


class MessageWrapper(Message):
    """A wrapper around the base Message type providing utility methods.

    This class extends the standard `Message` by adding convenient cached properties
    like `text` (for concatenated text content) and `tool_requests`.
    It stores the original message in the `_original_message` attribute.
    """

    def __init__(
        self,
        message: Message,
    ) -> None:
        """Initializes the MessageWrapper.

        Args:
            message: The original Message object to wrap.
        """
        super().__init__(
            role=message.role,
            content=message.content,
            metadata=message.metadata,
        )
        self._original_message: Message = message

    @cached_property
    def text(self) -> str:
        """Returns all text parts of the current chunk joined into a single string.

        Returns:
            str: The combined text content from the current chunk.
        """
        return text_from_message(self)

    @cached_property
    def tool_requests(self) -> list[ToolRequestPart]:
        """Returns all tool request parts of the response as a list.

        Returns:
            list[ToolRequestPart]: list of tool requests present in this response.
        """
        return [p.root for p in self.content if isinstance(p.root, ToolRequestPart)]

    @cached_property
    def interrupts(self) -> list[ToolRequestPart]:
        """Returns all interrupted tool request parts of the message as a list.

        Returns:
            list[ToolRequestPart]: list of interrupted tool requests.
        """
        return [p for p in self.tool_requests if p.metadata and p.metadata.root.get('interrupt')]


class GenerateResponseWrapper(GenerateResponse, Generic[OutputT]):
    """A wrapper around GenerateResponse providing utility methods.

    Extends the base `GenerateResponse` with cached properties (`text`, `output`,
    `messages`, `tool_requests`) and methods for validation (`assert_valid`,
    `assert_valid_schema`). It also handles optional message/chunk parsing.

    When used with `Output[T]`, the `output` property is typed as `T`.
    """

    # _message_parser is a private attribute that Pydantic will ignore
    _message_parser: Callable[[MessageWrapper], object] | None = PrivateAttr(None)
    # _schema_type stores the Pydantic class for runtime validation
    _schema_type: type[BaseModel] | None = PrivateAttr(None)
    # Override the parent's message field with our wrapper type (intentional Liskov violation)
    # pyrefly: ignore[bad-override] - Intentional covariant override for wrapper functionality
    message: MessageWrapper | None = None  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        response: GenerateResponse,
        request: GenerateRequest,
        message_parser: Callable[[MessageWrapper], object] | None = None,
        schema_type: type[BaseModel] | None = None,
    ) -> None:
        """Initializes a GenerateResponseWrapper instance.

        Args:
            response: The original GenerateResponse object.
            request: The GenerateRequest object associated with the response.
            message_parser: An optional function to parse the output from the message.
            schema_type: Optional Pydantic model class for runtime validation.
        """
        # Wrap the message if it's not already a MessageWrapper
        wrapped_message: MessageWrapper | None = None
        if response.message is not None:
            wrapped_message = (
                MessageWrapper(response.message)
                if not isinstance(response.message, MessageWrapper)
                else response.message
            )

        super().__init__(
            message=wrapped_message,
            finish_reason=response.finish_reason,
            finish_message=response.finish_message,
            latency_ms=response.latency_ms,
            usage=response.usage if response.usage is not None else GenerationUsage(),
            custom=response.custom if response.custom is not None else {},
            request=request,
            candidates=response.candidates,
            operation=response.operation,
        )
        # Set subclass-specific fields after parent initialization
        self._message_parser = message_parser
        self._schema_type = schema_type

    def assert_valid(self) -> None:
        """Validates the basic structure of the response.

        Note: This method is currently a placeholder (TODO).

        Raises:
            AssertionError: If the response structure is considered invalid.
        """
        # TODO(#4343): implement
        pass

    def assert_valid_schema(self) -> None:
        """Validates that the response message conforms to any specified output schema.

        Note: This method is currently a placeholder (TODO).

        Raises:
            AssertionError: If the response message does not conform to the schema.
        """
        # TODO(#4343): implement
        pass

    @cached_property
    def text(self) -> str:
        """Returns all text parts of the response joined into a single string.

        Returns:
            str: The combined text content from the response.
        """
        if self.message is None:
            return ''
        return self.message.text

    @cached_property
    def output(self) -> OutputT:
        """Parses out JSON data from the text parts of the response.

        When used with `Output[T]`, returns the parsed output typed as `T`.
        If a schema_type was provided and the parsed output is a dict,
        validates and returns a proper Pydantic model instance.

        Returns:
            The parsed JSON data from the response, typed according to the schema.
        """
        if self._message_parser and self.message is not None:
            parsed = self._message_parser(self.message)
        else:
            parsed = extract_json(self.text)

        # If we have a schema type and the parsed output is a dict, validate and
        # return a proper Pydantic instance. Skip if parsed is already the correct
        # type or if it's not a dict (e.g., custom formats may return strings).
        if self._schema_type is not None and parsed is not None and isinstance(parsed, dict):
            return cast(OutputT, self._schema_type.model_validate(parsed))

        return cast(OutputT, parsed)

    @cached_property
    def messages(self) -> list[Message]:
        """Returns all messages of the response, including request messages as a list.

        Returns:
            list[Message]: list of messages.
        """
        if self.message is None:
            return list(self.request.messages) if self.request else []
        return [
            *(self.request.messages if self.request else []),
            self.message._original_message,  # pyright: ignore[reportPrivateUsage]
        ]

    @cached_property
    def tool_requests(self) -> list[ToolRequestPart]:
        """Returns all tool request parts of the response as a list.

        Returns:
            list[ToolRequestPart]: list of tool requests present in this response.
        """
        if self.message is None:
            return []
        return self.message.tool_requests

    @cached_property
    def interrupts(self) -> list[ToolRequestPart]:
        """Returns all interrupted tool request parts of the response as a list.

        Returns:
            list[ToolRequestPart]: list of interrupted tool requests.
        """
        if self.message is None:
            return []
        return self.message.interrupts


class GenerateResponseChunkWrapper(GenerateResponseChunk):
    """A wrapper around GenerateResponseChunk providing utility methods.

    Extends the base `GenerateResponseChunk` with cached properties for accessing
    the text content of the current chunk (`text`), the accumulated text from all
    previous chunks including the current one (`accumulated_text`), and parsed
    output from the accumulated text (`output`). It also stores previous chunks.
    """

    # Field(exclude=True) means these fields are not included in serialization
    previous_chunks: list[GenerateResponseChunk] = Field(default_factory=list, exclude=True)
    chunk_parser: Callable[[GenerateResponseChunkWrapper], object] | None = Field(None, exclude=True)

    def __init__(
        self,
        chunk: GenerateResponseChunk,
        previous_chunks: list[GenerateResponseChunk],
        index: int,
        chunk_parser: Callable[[GenerateResponseChunkWrapper], object] | None = None,
    ) -> None:
        """Initializes the GenerateResponseChunkWrapper.

        Args:
            chunk: The raw GenerateResponseChunk to wrap.
            previous_chunks: A list of preceding chunks in the stream.
            index: The index of this chunk in the sequence of messages/chunks.
            chunk_parser: An optional function to parse the output from the chunk.
        """
        super().__init__(
            role=chunk.role,
            index=index,
            content=chunk.content,
            custom=chunk.custom,
            aggregated=chunk.aggregated,
        )
        # Set subclass-specific fields after parent initialization
        self.previous_chunks = previous_chunks
        self.chunk_parser = chunk_parser

    @cached_property
    def text(self) -> str:
        """Returns all text parts of the current chunk joined into a single string.

        Returns:
            str: The combined text content from the current chunk.
        """
        parts: list[str] = []
        for p in self.content:
            text_val = p.root.text
            if text_val is not None:
                # Handle Text RootModel (access .root) or plain str
                if isinstance(text_val, Text):
                    parts.append(str(text_val.root) if text_val.root is not None else '')
                else:
                    parts.append(str(text_val))
        return ''.join(parts)

    @cached_property
    def accumulated_text(self) -> str:
        """Returns all text parts from previous chunks plus the latest chunk.

        Returns:
            str: The combined text content from all chunks seen so far.
        """
        parts: list[str] = []
        if self.previous_chunks:
            for chunk in self.previous_chunks:
                for p in chunk.content:
                    text_val = p.root.text
                    if text_val:
                        # Handle Text RootModel (access .root) or plain str
                        if isinstance(text_val, Text):
                            parts.append(str(text_val.root) if text_val.root is not None else '')
                        else:
                            parts.append(str(text_val))
        return ''.join(parts) + self.text

    @cached_property
    def output(self) -> object:
        """Parses out JSON data from the accumulated text parts of the response.

        Returns:
            Any: The parsed JSON data from the accumulated chunks.
        """
        if self.chunk_parser:
            return self.chunk_parser(self)
        return extract_json(self.accumulated_text)


class PartCounts(BaseModel):
    """Stores counts of different types of media parts.

    Attributes:
        characters: Total number of characters in text parts.
        images: Total number of image parts.
        videos: Total number of video parts.
        audio: Total number of audio parts.
    """

    characters: int = 0
    images: int = 0
    videos: int = 0
    audio: int = 0


def text_from_message(msg: Message) -> str:
    """Extracts and concatenates text content from all parts of a Message.

    Args:
        msg: The Message object.

    Returns:
        A single string containing all text found in the message parts.
    """
    return text_from_content(msg.content)


def text_from_content(content: Sequence[Part | DocumentPart]) -> str:
    """Extracts and concatenates text content from a list of Parts or DocumentParts.

    Args:
        content: A sequence of Part or DocumentPart objects.

    Returns:
        A single string containing all text found in the parts.
    """
    return ''.join(str(p.root.text) for p in content if hasattr(p.root, 'text') and p.root.text is not None)


def get_basic_usage_stats(input_: list[Message], response: Message | list[Candidate]) -> GenerationUsage:
    """Calculates basic usage statistics based on input and output messages/candidates.

    Counts characters, images, videos, and audio files for both input and output.

    Args:
        input_: A list of input Message objects.
        response: Either a single output Message object or a list of Candidate objects.

    Returns:
        A GenerationUsage object populated with the calculated counts.
    """
    request_parts = []

    for msg in input_:
        request_parts.extend(msg.content)

    response_parts = []
    if isinstance(response, list):
        for candidate in response:
            response_parts.extend(candidate.message.content)
    else:
        response_parts = response.content

    input_counts = get_part_counts(parts=request_parts)
    output_counts = get_part_counts(parts=response_parts)

    return GenerationUsage(
        input_characters=input_counts.characters,
        input_images=input_counts.images,
        input_videos=input_counts.videos,
        input_audio_files=input_counts.audio,
        output_characters=output_counts.characters,
        output_images=output_counts.images,
        output_videos=output_counts.videos,
        output_audio_files=output_counts.audio,
    )


def get_part_counts(parts: list[Part]) -> PartCounts:
    """Counts the occurrences of different media types within a list of Parts.

    Iterates through the parts, summing character lengths and counting image,
    video, and audio parts based on content type or data URL prefix.

    Args:
        parts: A list of Part objects to analyze.

    Returns:
        A PartCounts object containing the aggregated counts.
    """
    part_counts = PartCounts()

    for part in parts:
        text_val = part.root.text
        if text_val:
            # Handle Text RootModel (access .root) or plain str
            if isinstance(text_val, Text):
                part_counts.characters += len(str(text_val.root)) if text_val.root else 0
            else:
                part_counts.characters += len(str(text_val))

        media = part.root.media

        if media:
            # Handle Media BaseModel vs MediaModel RootModel
            if isinstance(media, Media):
                content_type = media.content_type or ''
                url = media.url or ''
            elif isinstance(media, MediaModel) and hasattr(media.root, 'content_type'):
                content_type = getattr(media.root, 'content_type', '') or ''
                url = getattr(media.root, 'url', '') or ''
            else:
                content_type = ''
                url = ''
            is_image = content_type.startswith('image') or url.startswith('data:image')
            is_video = content_type.startswith('video') or url.startswith('data:video')
            is_audio = content_type.startswith('audio') or url.startswith('data:audio')

            part_counts.images += 1 if is_image else 0
            part_counts.videos += 1 if is_video else 0
            part_counts.audio += 1 if is_audio else 0

    return part_counts


def model_action_metadata(
    name: str,
    info: dict[str, object] | None = None,
    config_schema: type | dict[str, Any] | None = None,
) -> ActionMetadata:
    """Generates an ActionMetadata for models."""
    info = info if info is not None else {}
    return ActionMetadata(
        kind=cast(ActionKind, ActionKind.MODEL),
        name=name,
        input_json_schema=to_json_schema(GenerateRequest),
        output_json_schema=to_json_schema(GenerateResponse),
        metadata={'model': {**info, 'customOptions': to_json_schema(config_schema) if config_schema else None}},
    )


def model_ref(
    name: str,
    namespace: str | None = None,
    info: ModelInfo | None = None,
    version: str | None = None,
    config: dict[str, object] | None = None,
) -> ModelReference:
    """The factory function equivalent to export function modelRef(...).

    Args:
        name: The model name.
        namespace: Optional namespace to prefix the name.
        info: Optional model info.
        version: Optional model version.
        config: Optional model configuration.

    Returns:
        A ModelReference instance.
    """
    # Logic: if (options.namespace && !name.startsWith(options.namespace + '/'))
    final_name = f'{namespace}/{name}' if namespace and not name.startswith(f'{namespace}/') else name

    return ModelReference(name=final_name, info=info, version=version, config=config)
