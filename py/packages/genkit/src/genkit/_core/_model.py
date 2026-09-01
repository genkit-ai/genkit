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

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from typing import Any, ClassVar, Generic, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator
from pydantic.alias_generators import to_camel
from typing_extensions import TypedDict, TypeVar

from genkit._core._base import GenkitModel
from genkit._core._error import GenkitError
from genkit._core._extract_json import extract_json
from genkit._core._schema import parse_schema
from genkit._core._typing import (
    Candidate,
    DocumentData,
    DocumentPart,
    FinishReason,
    GenerateActionOptionsData,
    GenerateActionOutputConfig,
    GenerationCommonConfig,
    GenerationUsage,
    Media,
    MediaModel,
    MediaPart,
    MessageData,
    MiddlewareRef,
    ModelInfo,
    ModelResponseChunk as ModelResponseChunkSchema,
    Operation,
    OutputConfig as OutputConfigData,
    Part,
    Resume,
    Role,
    Text,
    TextPart,
    ToolChoice,
    ToolDefinition,
    ToolRequestPart,
)

# Runtime schema for common generate knobs. ModelConfigDict is the
# hand-copied autocomplete list — keep the keys matching so a new knob
# shows up in the IDE the same day it becomes legal.
ModelConfig = GenerationCommonConfig
ModelUsage = GenerationUsage  # public name for GenerationUsage

# The model's own reason stays on the response. A leftover that failed
# schema on a normal stop becomes ERROR instead.
_KEEP_MODEL_FINISH_REASONS = frozenset({
    FinishReason.BLOCKED,
    FinishReason.ABORTED,
    FinishReason.INTERRUPTED,
    FinishReason.OTHER,
})


class ModelConfigDict(TypedDict, extra_items=Any, total=False):
    """Common knobs for dict-literal autocomplete on ``config={...}``.

    ``None`` clears a ModelRef default. Extra keys (provider-specific) stay
    in the bag and are forwarded.

    Keys match ``GenerationCommonConfig`` / ``ModelConfig``. If a common
    knob is added there and not here, autocomplete quietly drops it.
    """

    version: str | None
    temperature: float | None
    max_output_tokens: float | None
    top_k: float | None
    top_p: float | None
    stop_sequences: Sequence[str] | None
    api_key: str | None


# TypeVars for generic types
OutputT = TypeVar('OutputT', default=object)
ConfigT = TypeVar('ConfigT', bound=ModelConfig, default=ModelConfig)
# Bound to BaseModel so ModelRef is always parameterized with a concrete Pydantic config schema.
# Covariant so ModelRef[GeminiConfig] is assignable to ModelRef[BaseModel] or ModelRef[Any].
ModelRefConfigT = TypeVar('ModelRefConfigT', bound=BaseModel, covariant=True)
# Unbounded so ModelRequest can carry plugin config schemas, plain dicts, or
# ModelConfig subclasses without forcing everything through GenerationCommonConfig.
# Invariant: config is writable, so ModelRequest[GeminiConfig] is not a
# ModelRequest[ModelConfig] you can assign a ModelConfig into.
ModelRequestConfigT = TypeVar('ModelRequestConfigT')


def declared_config_type(cls: type) -> type | None:
    """The config class on ``ModelRequest[ThatClass]``, or None if unparametrized."""
    meta = getattr(cls, '__pydantic_generic_metadata__', None)
    if not meta:
        return None
    args = meta.get('args') or ()
    if not args:
        return None
    arg = args[0]
    if isinstance(arg, TypeVar) or arg is Any:
        return None
    return arg


def config_type_path(cls: type) -> str:
    """The public import a plugin author would use, else the defining module.

    Walks parent packages from the top and uses the first one that re-exports
    this class under the same name (``genkit_openai.OpenAIConfig``, not
    ``genkit_openai.typing.OpenAIConfig``). Nested / test-local classes keep
    the defining path.
    """
    impl = f'{cls.__module__}.{cls.__qualname__}'
    if '<locals>' in cls.__qualname__ or '.' in cls.__qualname__:
        return impl
    parts = cls.__module__.split('.')
    name = cls.__name__
    for i in range(1, len(parts) + 1):
        mod_name = '.'.join(parts[:i])
        try:
            mod = import_module(mod_name)
        except ImportError:
            continue
        if getattr(mod, name, None) is not cls:
            continue
        public = getattr(mod, '__all__', None)
        if public is not None and name not in public:
            continue
        return f'{mod_name}.{name}'
    return impl


@dataclass(frozen=True, kw_only=True)
class ModelRef(Generic[ModelRefConfigT]):
    """Handle for a model tied to a config schema.

    Fields cannot be rebound. config and info are copied at construction so later
    mutations of the caller's objects don't change the ref; the copies themselves
    stay ordinary mutable Pydantic models.
    """

    name: str
    config_schema: type[ModelRefConfigT]
    info: ModelInfo | None = None
    version: str | None = None
    config: ModelRefConfigT | None = None

    # Explicitly opt out of hashing: Pydantic configs are unhashable, so an
    # auto-generated __hash__ would fail once set.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # If config_schema is not a BaseModel subclass, raise an error.
        schema = self.config_schema
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            got = (
                f'{schema.__module__}.{schema.__name__}'
                if isinstance(schema, type)
                else f'{type(schema).__module__}.{type(schema).__name__}'
            )
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{self.name}: config_schema must be a BaseModel subclass, got {got}',
            )
        if self.config is not None and not isinstance(self.config, schema):
            expected = config_type_path(schema)
            actual = config_type_path(type(self.config))
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{self.name}: config must be an instance of {expected}, got {actual}',
            )
        # If info is present, validate that it is a ModelInfo and raise an error if not.
        if self.info is not None and not isinstance(self.info, ModelInfo):
            actual = f'{type(self.info).__module__}.{type(self.info).__name__}'
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(f'{self.name}: info must be an instance of {ModelInfo.__module__}.ModelInfo, got {actual}'),
            )
        # Callers often keep the config/info they passed in. Copy so later
        # mutations of those objects don't change the ref's defaults.
        if self.config is not None:
            object.__setattr__(self, 'config', self.config.model_copy(deep=True))
        if self.info is not None:
            object.__setattr__(self, 'info', self.info.model_copy(deep=True))


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

    @property
    def text(self) -> str:
        """All text parts concatenated into a single string."""
        return text_from_message(self)

    @property
    def tool_requests(self) -> list[ToolRequestPart]:
        """All tool request parts in this message."""
        return [p.root for p in self.content if isinstance(p.root, ToolRequestPart)]

    @property
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
        """All media parts."""
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


class OutputConfig(OutputConfigData):
    """Output settings for a model request.

    Construct with ``json_schema=``; the serialized key on the wire is
    ``schema``. This is the class to import and construct from hand-written
    code.
    """


class ModelRequest(GenkitModel, Generic[ModelRequestConfigT]):
    """Hand-written model request with veneer types and flat output accessors.

    Output settings live nested as ``output: OutputConfig`` so dump/validate
    round-trips the wire shape, while flat properties (``output_format`` etc.)
    stay the plugin-author convenience surface. Messages and docs use veneer
    types (Message, Document) for helpers like ``.text``.

    Example:
        class GeminiConfig(ModelConfig):
            safety_settings: dict[str, str] | None = None

        def gemini_model(request: ModelRequest[GeminiConfig]) -> ModelResponse:
            temp = request.config.temperature  # inherited from ModelConfig
            for msg in request.messages:
                print(msg.text)  # Message veneer property
            if request.output_format == 'json':
                schema = request.output_schema

    Note:
        Pass output settings as ``output=OutputConfig(...)``. The flat
        names (``output_format`` etc.) are convenience properties you read
        and write after construction — they are not constructor arguments,
        so passing them there leaves output unset.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(alias_generator=to_camel, extra='allow', populate_by_name=True)
    # Veneer types for IDE/typing (validators wrap MessageData->Message, DocumentData->Document)
    messages: list[Message]  # pyright: ignore[reportIncompatibleVariableOverride]
    docs: list[Document] | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    config: ModelRequestConfigT | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = Field(default=None)
    # Wire-shaped output storage; flat access via the properties below.
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator('config', mode='before')
    @classmethod
    def _check_config_type(cls, v: object) -> object:
        """A mapping is the bag the plugin schema coerces.

        A Pydantic instance is only legal if it is that schema. OpenAIConfig
        on a Gemini request is a caller mistake — pass a mapping instead.
        """
        if v is None:
            return v
        if isinstance(v, Mapping) and not isinstance(v, BaseModel):
            return v
        if isinstance(v, BaseModel):
            expected = declared_config_type(cls)
            if isinstance(expected, type) and issubclass(expected, BaseModel) and not isinstance(v, expected):
                raise ValueError(
                    f'config must be {config_type_path(expected)} or a mapping, got {config_type_path(type(v))}'
                )
            if expected is dict:
                raise ValueError(f'config must be a mapping, got {type(v).__name__}')
            return v
        raise ValueError(f'config must be a BaseModel or mapping, got {type(v).__name__}')

    @field_validator('messages', mode='before')
    @classmethod
    def _wrap_messages(cls, v: list[MessageData]) -> list[Message]:
        """Wrap MessageData in Message veneer for convenience methods."""
        # pyrefly: ignore[bad-return]
        return [m if isinstance(m, Message) else Message(m) for m in v]

    @field_validator('docs', mode='before')
    @classmethod
    def _wrap_docs(cls, v: list[DocumentData] | None) -> list[Document] | None:
        """Wrap DocumentData in Document veneer for convenience methods.

        A dumped request sends docs as dicts. Messages already take a mapping;
        this wrap has to as well or a bad config plus docs= never reaches the
        GenkitError for the config.
        """
        if v is None:
            return None
        wrapped: list[Document] = []
        for d in v:
            if isinstance(d, Document):
                wrapped.append(d)
            elif isinstance(d, dict):
                wrapped.append(Document(d.get('content') or [], d.get('metadata')))
            else:
                wrapped.append(Document(d.content, d.metadata))
        return wrapped

    # Flat accessors: the plugin-author convenience surface over nested output.

    @property
    def output_format(self) -> str | None:
        """Output format (e.g. 'json'); reads ``output.format``."""
        return self.output.format

    @output_format.setter
    def output_format(self, v: str | None) -> None:
        self.output.format = v

    @property
    def output_schema(self) -> dict[str, Any] | None:
        """Output JSON schema; reads ``output.json_schema``."""
        return self.output.json_schema

    @output_schema.setter
    def output_schema(self, v: dict[str, Any] | None) -> None:
        self.output.json_schema = v

    @property
    def output_constrained(self) -> bool | None:
        """Whether constrained decoding is requested; reads ``output.constrained``."""
        return self.output.constrained

    @output_constrained.setter
    def output_constrained(self, v: bool | None) -> None:
        self.output.constrained = v

    @property
    def output_content_type(self) -> str | None:
        """Output content type; reads ``output.content_type``."""
        return self.output.content_type

    @output_content_type.setter
    def output_content_type(self, v: str | None) -> None:
        self.output.content_type = v


def operation_snapshot(*, operation: Operation | None) -> tuple[object, object, object, object]:
    """Job id plus the fields that change when a check lands."""
    if operation is None:
        return (None, None, None, None)
    return (operation.id, operation.done, operation.error, operation.output)


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
        """No-op. A blocked or empty reply is still a response the caller can read."""

    def assert_valid_schema(self) -> None:
        """Mark this response as unusable structured output without throwing.

        A leftover echo or a wrong-shape JSON is not a Recipe. generate()
        still returns so the leftover stays on ``.text``; we set
        ``finish_reason=error`` and ``.output`` is None.
        A blocked/aborted/interrupted/other finish keeps the model's reason.
        """
        schema = self.request.output_schema if self.request is not None else None
        if schema is None and self._schema_type is None:
            return
        if self.finish_reason in _KEEP_MODEL_FINISH_REASONS:
            return

        try:
            parsed = self._raw_parsed_output()
        except ValueError:
            preview = (self.text or '')[:200]
            self.finish_reason = FinishReason.FAILED
            self.finish_message = f'Model output was not valid JSON for the requested schema: {preview}'
            return

        # A custom format's parser can return a string on purpose (enum,
        # text). Still check it against the schema — MAYBE is not one of
        # POSITIVE/NEGATIVE/NEUTRAL.
        if self._message_parser is not None and not isinstance(parsed, (dict, list)):
            if schema is not None:
                try:
                    parse_schema(data=parsed, json_schema=schema)
                except GenkitError as error:
                    if error.original_message.startswith('Invalid output_schema'):
                        raise
                    self.finish_reason = FinishReason.FAILED
                    self.finish_message = error.original_message
            return

        if schema is not None:
            try:
                parse_schema(data=parsed, json_schema=schema)
            except GenkitError as error:
                if error.original_message.startswith('Invalid output_schema'):
                    raise
                self.finish_reason = FinishReason.FAILED
                self.finish_message = error.original_message
                return
        if self._schema_type is None:
            return
        try:
            _ = self._schema_type.model_validate(parsed)
        except ValidationError:
            self.finish_reason = FinishReason.FAILED
            self.finish_message = 'Model output did not match the requested schema.'

    def _raw_parsed_output(self) -> object:
        if self._message_parser and self.message is not None:
            return self._message_parser(self.message)
        return extract_json(self.text)

    def __eq__(self, other: object) -> bool:
        """Compare responses by message, finish_reason, and poll snapshot.

        Same job id with a later done/error/output is a later check, not
        the same response. Timing on the handle is not part of the job.
        """
        if isinstance(other, ModelResponse):
            return (
                self.message == other.message
                and self.finish_reason == other.finish_reason
                and operation_snapshot(operation=self.operation) == operation_snapshot(operation=other.operation)
            )
        return super().__eq__(other)

    def __hash__(self) -> int:
        """Return identity-based hash."""
        return hash(id(self))

    @property
    def text(self) -> str:
        """All text parts concatenated into a single string."""
        if self.message is None:
            return ''
        return self.message.text

    @property
    def output(self) -> OutputT:
        """Parsed structured output, or None when the reply is not that shape.

        generate() does not throw on a leftover string. If you asked for a
        schema and this is not it, read ``finish_reason`` / ``.text`` instead.
        """
        schema = self.request.output_schema if self.request is not None else None
        wants_schema = schema is not None or self._schema_type is not None
        if self.finish_reason in (FinishReason.BLOCKED, FinishReason.FAILED):
            return cast(OutputT, None)
        if wants_schema and self.finish_reason in _KEEP_MODEL_FINISH_REASONS:
            return cast(OutputT, None)

        try:
            parsed = self._raw_parsed_output()
        except ValueError:
            if wants_schema:
                return cast(OutputT, None)
            raise

        if self._message_parser is not None and not isinstance(parsed, (dict, list)):
            if schema is not None:
                try:
                    parse_schema(data=parsed, json_schema=schema)
                except GenkitError:
                    return cast(OutputT, None)
            return cast(OutputT, parsed)

        if schema is not None:
            try:
                parse_schema(data=parsed, json_schema=schema)
            except GenkitError:
                return cast(OutputT, None)
        if self._schema_type is not None and parsed is not None:
            try:
                return cast(OutputT, self._schema_type.model_validate(parsed))
            except ValidationError:
                return cast(OutputT, None)
        return cast(OutputT, parsed)

    @property
    def messages(self) -> list[Message]:
        """All messages including request history and the response message.

        Recomputed each read so attaching ``request`` later still shows up.
        """
        if self.message is None:
            return [Message(m) for m in self.request.messages] if self.request else []
        return [
            *(Message(m) for m in (self.request.messages if self.request else [])),
            self.message,
        ]

    @property
    def tool_requests(self) -> list[ToolRequestPart]:
        """All tool request parts in the response message.

        Recomputed each read so a later message still shows up.
        """
        if self.message is None:
            return []
        return self.message.tool_requests

    @property
    def media(self) -> list[Media]:
        """All media parts in the response message."""
        if self.message is None:
            return []
        return [
            part.root.media
            for part in self.message.content
            if isinstance(part.root, MediaPart) and part.root.media is not None
        ]

    @property
    def interrupts(self) -> list[ToolRequestPart]:
        """Tool requests marked as interrupted."""
        if self.message is None:
            return []
        return self.message.interrupts


class ModelResponseChunk(ModelResponseChunkSchema, Generic[OutputT]):
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

    @property
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

    @property
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

    @property
    def output(self) -> OutputT:
        """Parsed JSON output from accumulated text."""
        if self.chunk_parser:
            return cast(OutputT, self.chunk_parser(self))
        return cast(OutputT, extract_json(self.accumulated_text))


def text_from_message(msg: Message) -> str:
    """Concatenate text from all parts of a message."""
    return text_from_content(msg.content)


def text_from_content(content: Sequence[Part | DocumentPart]) -> str:
    """Concatenate text parts.

    Thoughts ride on ``ReasoningPart``, so they stay out of ``.text`` —
    that's the visible reply, not the model's scratch work.
    """
    texts: list[str] = []
    for p in content:
        root = p.root
        if isinstance(root, TextPart) and root.text is not None:
            texts.append(str(root.text))
    return ''.join(texts)


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
