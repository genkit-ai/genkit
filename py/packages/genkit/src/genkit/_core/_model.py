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

"""Model veneer types for the Genkit framework.

This module contains the hand-written wrapper classes that provide convenient
properties and methods on top of the generated wire types.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from functools import cached_property
from typing import Any, ClassVar, Generic, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_serializer
from pydantic.alias_generators import to_camel
from typing_extensions import TypeVar

from genkit._core._base import GenkitModel
from genkit._core._extract_json import extract_json
from genkit._core._typing import (
    Candidate,
    DocumentData,
    DocumentPart,
    FinishReason,
    GenerateActionOptionsData,
    GenerateActionOutputConfig,
    GenerateResponseChunk,
    GenerationCommonConfig,
    GenerationUsage,
    Media,
    MediaModel,
    MediaPart,
    MessageData,
    MiddlewareRef,
    Operation,
    Part,
    Resume,
    Role,
    Text,
    TextPart,
    ToolChoice,
    ToolDefinition,
    ToolRequestPart,
)

ModelConfig = GenerationCommonConfig  # public name for GenerationCommonConfig
ModelUsage = GenerationUsage  # public name for GenerationUsage

# TypeVars for generic types
OutputT = TypeVar('OutputT', default=object)
ConfigT = TypeVar('ConfigT', bound=ModelConfig, default=ModelConfig)


class ModelRef(BaseModel):
    """Reference to a model with configuration."""

    name: str
    config_schema: object | None = None
    info: object | None = None
    version: str | None = None
    config: dict[str, object] | None = None


class Message(MessageData):
    """Message wrapper with utility properties for text and tool requests."""

    def __init__(
        self,
        message: MessageData | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize from MessageData or keyword arguments."""
        if message is not None:
            if isinstance(message, dict):
                role = message.get('role')
                if role is None:
                    raise ValueError('Message role is required')
                super().__init__(
                    role=role,
                    content=message.get('content', []),
                    metadata=message.get('metadata'),
                )
            else:
                super().__init__(
                    role=message.role,
                    content=message.content,
                    metadata=message.metadata,
                )
        else:
            super().__init__(**kwargs)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        """Compare messages by role, content, and metadata."""
        if isinstance(other, MessageData):
            return self.role == other.role and self.content == other.content and self.metadata == other.metadata
        return super().__eq__(other)

    def __hash__(self) -> int:
        """Return identity-based hash."""
        return hash(id(self))

    @cached_property
    def text(self) -> str:
        """All text parts concatenated into a single string."""
        return text_from_message(self)

    @cached_property
    def tool_requests(self) -> list[ToolRequestPart]:
        """All tool request parts in this message."""
        return [p.root for p in self.content if isinstance(p.root, ToolRequestPart)]

    @cached_property
    def interrupts(self) -> list[ToolRequestPart]:
        """Tool requests marked as interrupted."""
        return [p for p in self.tool_requests if p.metadata and p.metadata.get('interrupt')]


class GenerateActionOptions(GenerateActionOptionsData):
    """Generate options with messages as list[Message] for type-safe use with ai.generate()."""

    messages: list[Message]

    @field_validator('messages', mode='before')
    @classmethod
    def _wrap_messages(cls, v: list[MessageData]) -> list[Message]:
        return [m if isinstance(m, Message) else Message(m) for m in v]


_TEXT_DATA_TYPE: str = 'text'


class Document(DocumentData):
    """Multi-part document that can be embedded, indexed, or retrieved."""

    def __init__(
        self,
        content: list[DocumentPart],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with content parts and optional metadata."""
        doc_content = deepcopy(content)
        doc_metadata = deepcopy(metadata)
        super().__init__(content=doc_content, metadata=doc_metadata)

    @staticmethod
    def from_text(text: str, metadata: dict[str, Any] | None = None) -> Document:
        """Create a document from a text string."""
        return Document(content=[DocumentPart(root=TextPart(text=text))], metadata=metadata)

    @staticmethod
    def from_media(
        url: str,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Create a document from a media URL."""
        return Document(
            content=[DocumentPart(root=MediaPart(media=Media(url=url, content_type=content_type)))],
            metadata=metadata,
        )

    @staticmethod
    def from_data(
        data: str,
        data_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Create a document from data, inferring text vs media from data_type."""
        if data_type == _TEXT_DATA_TYPE:
            return Document.from_text(data, metadata)
        return Document.from_media(data, data_type, metadata)

    @cached_property
    def text(self) -> str:
        """Concatenate all text parts."""
        texts = []
        for p in self.content:
            part = p.root if hasattr(p, 'root') else p
            text_val = getattr(part, 'text', None)
            if isinstance(text_val, str):
                texts.append(text_val)
        return ''.join(texts)

    @cached_property
    def media(self) -> list[Media]:
        """Get all media parts."""
        return [
            part.root.media for part in self.content if isinstance(part.root, MediaPart) and part.root.media is not None
        ]

    @cached_property
    def data(self) -> str:
        """Primary data: text if available, otherwise first media URL."""
        if self.text:
            return self.text
        if self.media:
            return self.media[0].url
        return ''

    @cached_property
    def data_type(self) -> str | None:
        """Type of primary data: 'text' or first media's content type."""
        if self.text:
            return _TEXT_DATA_TYPE
        if self.media and self.media[0].content_type:
            return self.media[0].content_type
        return None


class ModelRequest(GenkitModel, Generic[ConfigT]):
    """Hand-written model request with flat output fields and veneer types.

    Output config is inlined as flat fields (output_format, output_schema, etc.)
    instead of a nested OutputConfig object. Messages and docs use veneer types
    (Message, Document) for convenience methods like .text.

    Example:
        class GeminiConfig(ModelConfig):
            safety_settings: dict[str, str] | None = None

        def gemini_model(request: ModelRequest[GeminiConfig]) -> ModelResponse:
            temp = request.config.temperature  # inherited from ModelConfig
            for msg in request.messages:
                print(msg.text)  # Message veneer property
            if request.output_format == 'json':
                schema = request.output_schema
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(alias_generator=to_camel, extra='allow', populate_by_name=True)
    # Veneer types for IDE/typing (validators wrap MessageData->Message, DocumentData->Document)
    messages: list[Message]  # pyright: ignore[reportIncompatibleVariableOverride]
    docs: list[Document] | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    config: ConfigT | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = Field(default=None)
    # Flat output fields (no nested OutputConfig)
    output_format: str | None = None
    output_schema: dict[str, Any] | None = None
    output_constrained: bool | None = None
    output_content_type: str | None = None

    @field_validator('messages', mode='before')
    @classmethod
    def _wrap_messages(cls, v: list[MessageData]) -> list[Message]:
        """Wrap MessageData in Message veneer for convenience methods."""
        # pyrefly: ignore[bad-return]
        return [m if isinstance(m, Message) else Message(m) for m in v]

    @field_validator('docs', mode='before')
    @classmethod
    def _wrap_docs(cls, v: list[DocumentData] | None) -> list[Document] | None:
        """Wrap DocumentData in Document veneer for convenience methods."""
        if v is None:
            return None
        # pyrefly: ignore[bad-return]
        return [d if isinstance(d, Document) else Document(d.content, d.metadata) for d in v]

    @model_serializer(mode='wrap')
    def _serialize_for_spec(self, serializer: Callable[..., dict[str, Any]]) -> dict[str, Any]:
        """Serialize to spec wire format with nested output (matches JS/Go)."""
        data = serializer(self)
        # Build nested output from flat fields - spec expects output key always present
        output: dict[str, Any] = {}
        if self.output_format is not None:
            output['format'] = self.output_format
        if self.output_schema is not None:
            output['schema'] = self.output_schema
        if self.output_constrained is not None:
            output['constrained'] = self.output_constrained
        if self.output_content_type is not None:
            output['contentType'] = self.output_content_type
        # Remove flat fields, add nested output
        data.pop('outputFormat', None)
        data.pop('outputSchema', None)
        data.pop('outputConstrained', None)
        data.pop('outputContentType', None)
        data['output'] = output
        return data


class ModelResponse(GenkitModel, Generic[OutputT]):
    """Model response with utilities for text extraction, output parsing, and validation."""

    # _message_parser and _schema_type are set by the framework after construction
    # when output format parsing or schema validation is needed.
    _message_parser: Callable[[Message], object] | None = PrivateAttr(None)
    _schema_type: type[BaseModel] | None = PrivateAttr(None)
    # Wire fields (must be declared for extra='forbid' to accept wire responses)
    message: Message | None = None
    finish_reason: FinishReason | None = None
    finish_message: str | None = None
    latency_ms: float | None = None
    usage: GenerationUsage | None = None
    custom: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    request: ModelRequest | None = None
    operation: Operation | None = None
    candidates: list[Candidate] | None = None

    def model_post_init(self, __context: object) -> None:
        """Initialize default usage and custom dict if not provided."""
        if self.usage is None:
            self.usage = GenerationUsage()
        if self.custom is None:
            self.custom = {}

    def assert_valid(self) -> None:
        """Validate response structure. (TODO: not yet implemented)."""
        # TODO(#4343): implement
        pass

    def assert_valid_schema(self) -> None:
        """Validate response conforms to output schema. (TODO: not yet implemented)."""
        # TODO(#4343): implement
        pass

    def __eq__(self, other: object) -> bool:
        """Compare responses by message and finish_reason."""
        if isinstance(other, ModelResponse):
            return self.message == other.message and self.finish_reason == other.finish_reason
        return super().__eq__(other)

    def __hash__(self) -> int:
        """Return identity-based hash."""
        return hash(id(self))

    @cached_property
    def text(self) -> str:
        """All text parts concatenated into a single string."""
        if self.message is None:
            return ''
        return self.message.text

    @cached_property
    def output(self) -> OutputT:
        """Parsed JSON output from the response text, validated against schema if set."""
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
        """All messages including request history and the response message."""
        if self.message is None:
            return [Message(m) for m in self.request.messages] if self.request else []
        return [
            *(Message(m) for m in (self.request.messages if self.request else [])),
            self.message,
        ]

    @cached_property
    def tool_requests(self) -> list[ToolRequestPart]:
        """All tool request parts in the response message."""
        if self.message is None:
            return []
        return self.message.tool_requests

    @cached_property
    def media(self) -> list[Media]:
        """All media parts in the response message."""
        if self.message is None:
            return []
        return [
            part.root.media
            for part in self.message.content
            if isinstance(part.root, MediaPart) and part.root.media is not None
        ]

    @cached_property
    def interrupts(self) -> list[ToolRequestPart]:
        """Tool requests marked as interrupted."""
        if self.message is None:
            return []
        return self.message.interrupts


class ModelResponseChunk(GenerateResponseChunk, Generic[OutputT]):
    """Streaming chunk with text, accumulated text, and output parsing."""

    # Field(exclude=True) means these fields are not included in serialization
    previous_chunks: list[ModelResponseChunk[Any]] = Field(default_factory=list, exclude=True)
    chunk_parser: Callable[[ModelResponseChunk[Any]], object] | None = Field(None, exclude=True)

    def __init__(
        self,
        chunk: ModelResponseChunk[Any] | None = None,
        previous_chunks: list[ModelResponseChunk[Any]] | None = None,
        index: int | float | None = None,
        chunk_parser: Callable[[ModelResponseChunk[Any]], object] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize from a chunk or keyword arguments."""
        if chunk is not None:
            # Framework wrapping mode
            super().__init__(
                role=chunk.role,
                index=index,
                content=chunk.content,
                custom=chunk.custom,
                aggregated=chunk.aggregated,
            )
        else:
            # No source chunk — caller passes fields (role, content, etc.) as kwargs directly
            super().__init__(**kwargs)
        self.previous_chunks = previous_chunks or []
        self.chunk_parser = chunk_parser

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if isinstance(other, ModelResponseChunk):
            return self.role == other.role and self.content == other.content
        return super().__eq__(other)

    def __hash__(self) -> int:
        """Return hash."""
        return hash(id(self))

    @cached_property
    def text(self) -> str:
        """Text content of this chunk."""
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
        """Text from all previous chunks plus this chunk."""
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
    def output(self) -> OutputT:
        """Parsed JSON output from accumulated text."""
        if self.chunk_parser:
            return cast(OutputT, self.chunk_parser(self))
        return cast(OutputT, extract_json(self.accumulated_text))


def text_from_message(msg: Message) -> str:
    """Concatenate text from all parts of a message."""
    return text_from_content(msg.content)


def text_from_content(content: Sequence[Part | DocumentPart]) -> str:
    """Concatenate text from a list of parts."""
    return ''.join(str(p.root.text) for p in content if hasattr(p.root, 'text') and p.root.text is not None)


def get_basic_usage_stats(input_: list[Message], response: Message) -> GenerationUsage:
    """Calculate usage stats (characters, media counts) from messages."""
    request_parts: list[Part] = []
    for msg in input_:
        request_parts.extend(msg.content)

    response_parts = response.content

    def count_parts(parts: list[Part]) -> tuple[int, int, int, int]:
        """Count characters, images, videos, audio in parts."""
        characters = 0
        images = 0
        videos = 0
        audio = 0

        for part in parts:
            text_val = part.root.text
            if text_val:
                if isinstance(text_val, Text):
                    characters += len(str(text_val.root)) if text_val.root else 0
                else:
                    characters += len(str(text_val))

            media = part.root.media
            if media:
                if isinstance(media, Media):
                    content_type = media.content_type or ''
                    url = media.url or ''
                elif isinstance(media, MediaModel) and hasattr(media.root, 'content_type'):
                    content_type = getattr(media.root, 'content_type', '') or ''
                    url = getattr(media.root, 'url', '') or ''
                else:
                    content_type = ''
                    url = ''

                if content_type.startswith('image') or url.startswith('data:image'):
                    images += 1
                elif content_type.startswith('video') or url.startswith('data:video'):
                    videos += 1
                elif content_type.startswith('audio') or url.startswith('data:audio'):
                    audio += 1

        return characters, images, videos, audio

    in_chars, in_imgs, in_vids, in_audio = count_parts(request_parts)
    out_chars, out_imgs, out_vids, out_audio = count_parts(response_parts)

    return GenerationUsage(
        input_characters=in_chars,
        input_images=in_imgs,
        input_videos=in_vids,
        input_audio_files=in_audio,
        output_characters=out_chars,
        output_images=out_imgs,
        output_videos=out_vids,
        output_audio_files=out_audio,
    )


# Rebuild schema after all types (including Message) are fully defined.
# _types_namespace provides forward-ref resolution for GenerateActionOptionsData fields.
GenerateActionOptions.model_rebuild(
    _types_namespace={
        'GenerateActionOutputConfig': GenerateActionOutputConfig,
        'MiddlewareRef': MiddlewareRef,
        'Resume': Resume,
        'Role': Role,
    }
)

# Type aliases for model middleware (Any is intentional - middleware is type-agnostic)
# Middleware can have two signatures:
#   Simple (3 params): (req, ctx, next) -> response
#   Streaming (4 params): (req, ctx, on_chunk, next) -> response
# The framework detects which signature is used based on parameter count.
ModelMiddleware = Callable[..., Awaitable[ModelResponse[Any]]]
