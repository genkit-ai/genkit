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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    RootModel,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel
from typing_extensions import TypedDict, TypeVar

from genkit._core import _typing as typing_mod
from genkit._core._base import GenkitModel, dump_keeping_unknown
from genkit._core._error import GenkitError
from genkit._core._extract_json import extract_json
from genkit._core._partial import construct_partial
from genkit._core._schema import parse_schema
from genkit._core._typing import (
    AgentFinishReason,
    Artifact as ArtifactData,
    DocumentData,
    FinishReason,
    GenerateActionOutputConfig,
    GenerationCommonConfig,
    GenerationUsage,
    GenkitRuntimeError,
    JsonPatch,
    Media,
    MessageData,
    MiddlewareRef,
    ModelInfo,
    Operation,
    OutputConfig as OutputConfigData,
    PartData,
    Resource,
    Resume as ResumeData,
    Role,
    SnapshotStatus,
    ToolChoice,
    ToolDefinition,
    ToolRequest,
    ToolResponse,
    TurnEnd,
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


# Exclusive kinds. camelCase and snake_case are the same kind so a merged
# dump of one tool call is not two kinds. custom is the vendor hatch — it
# may ride on another kind, or be the kind when it's the only payload (a
# signed thought is still one reasoning part). Metadata rides too. Empty
# or two exclusive kinds is a validation error so a Message never carries
# an ambiguous part the model would have to guess at.
PART_KIND_KEYS = frozenset({
    'text',
    'media',
    'toolRequest',
    'tool_request',
    'toolResponse',
    'tool_response',
    'reasoning',
    'resource',
    'data',
})
PART_KIND_FIELDS = (
    'text',
    'media',
    'tool_request',
    'tool_response',
    'reasoning',
    'resource',
    'data',
)
PART_KIND_ALIASES = {
    'tool_request': 'toolRequest',
    'tool_response': 'toolResponse',
}
EXACTLY_ONE_KIND = (
    'a part must have exactly one of text, media, toolRequest, toolResponse, reasoning, resource, data, or custom'
)


def present_part_kinds(raw: dict[str, object]) -> list[str]:
    seen: set[str] = set()
    for key in PART_KIND_KEYS:
        if raw.get(key) is not None:
            seen.add(PART_KIND_ALIASES.get(key, key))
    return list(seen)


def _require_exactly_one_kind(raw: Mapping[str, object]) -> None:
    kinds = present_part_kinds(dict(raw))
    if len(kinds) > 1:
        raise ValueError(EXACTLY_ONE_KIND)
    if len(kinds) == 1:
        return
    if raw.get('custom') is not None:
        return
    raise ValueError(EXACTLY_ONE_KIND)


class Part(GenkitModel):
    """A single piece of content in a message or document."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel,
        extra='forbid',
        populate_by_name=True,
        validate_assignment=True,
    )

    text: str | None = None
    media: Media | None = None
    tool_request: ToolRequest | None = None
    tool_response: ToolResponse | None = None
    data: Any | None = Field(default=None)
    metadata: dict[str, Any] | None = None
    custom: dict[str, Any] | None = None
    reasoning: str | None = None
    resource: Resource | None = None

    @model_validator(mode='before')
    @classmethod
    def _exactly_one_kind(cls, value: object) -> object:
        if isinstance(value, Mapping) and 'root' in value:
            raise ValueError('Part(root=...) is gone; use Part.from_text or Part(text=...)')
        if isinstance(value, Part):
            value = dump_keeping_unknown(value)
        if isinstance(value, PartData):
            value = value.root
        if isinstance(value, BaseModel):
            value = dump_keeping_unknown(value)
        if not isinstance(value, dict):
            return value
        raw = cast(dict[str, object], value)
        _require_exactly_one_kind(raw)
        return value

    @model_validator(mode='after')
    def _exactly_one_kind_after(self) -> Part:
        kinds = [name for name in PART_KIND_FIELDS if getattr(self, name) is not None]
        if len(kinds) > 1:
            raise ValueError(EXACTLY_ONE_KIND)
        if len(kinds) == 1:
            return self
        if self.custom is not None:
            return self
        raise ValueError(EXACTLY_ONE_KIND)

    @classmethod
    def from_text(cls, text: str, metadata: dict[str, Any] | None = None) -> Part:
        return cls(text=text, metadata=metadata)

    @classmethod
    def from_media(
        cls,
        url: str,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Part:
        return cls(media=Media(url=url, content_type=content_type), metadata=metadata)

    @classmethod
    def from_tool_request(
        cls,
        name: str,
        input: Any | None = None,  # noqa: ANN401
        ref: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Part:
        return cls(tool_request=ToolRequest(name=name, input=input, ref=ref), metadata=metadata)

    @classmethod
    def from_tool_response(
        cls,
        name: str,
        output: Any | None = None,  # noqa: ANN401
        ref: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Part:
        return cls(tool_response=ToolResponse(name=name, output=output, ref=ref), metadata=metadata)

    @classmethod
    def from_data(cls, data: Any, metadata: dict[str, Any] | None = None) -> Part:  # noqa: ANN401
        return cls(data=data, metadata=metadata)

    @classmethod
    def from_custom(cls, custom: dict[str, Any], metadata: dict[str, Any] | None = None) -> Part:
        return cls(custom=custom, metadata=metadata)

    @classmethod
    def from_reasoning(cls, reasoning: str, metadata: dict[str, Any] | None = None) -> Part:
        return cls(reasoning=reasoning, metadata=metadata)


def as_part(value: object) -> Part:
    if isinstance(value, Part):
        return Part.model_validate(dump_keeping_unknown(value))
    if isinstance(value, PartData):
        return Part.model_validate(value.root)
    if isinstance(value, BaseModel):
        return Part.model_validate(dump_keeping_unknown(value))
    return Part.model_validate(value)


def as_message(value: object) -> Message:
    if isinstance(value, Message):
        return Message(
            role=value.role,
            content=[as_part(p) for p in value.content],
            metadata=value.metadata,
        )
    if isinstance(value, MessageData):
        return Message.model_validate(dump_keeping_unknown(value))
    return Message.model_validate(value)


def as_document(value: object) -> Document:
    if isinstance(value, Document):
        return Document(
            content=[as_part(p) for p in value.content],
            metadata=value.metadata,
        )
    if isinstance(value, DocumentData):
        return Document.model_validate(dump_keeping_unknown(value))
    return Document.model_validate(value)


def as_artifact(value: object) -> Artifact:
    if isinstance(value, Artifact):
        return Artifact(
            name=value.name,
            parts=[as_part(p) for p in value.parts],
            metadata=value.metadata,
        )
    if isinstance(value, ArtifactData):
        return Artifact.model_validate(dump_keeping_unknown(value))
    return Artifact.model_validate(value)


def as_output_config(value: object) -> OutputConfig:
    if isinstance(value, OutputConfig):
        return value
    if isinstance(value, OutputConfigData):
        return OutputConfig.model_validate(dump_keeping_unknown(value))
    return OutputConfig.model_validate(value)


def as_resume_respond(value: object) -> Part:
    part = as_part(value)
    if part.tool_response is None:
        raise ValueError('resume_respond needs a tool response part')
    return part


def as_resume_restart(value: object) -> Part:
    part = as_part(value)
    if part.tool_request is None:
        raise ValueError('resume_restart needs a tool request part')
    return part


class Resume(GenkitModel):
    """Resume payload whose respond/restart lists accept Part."""

    respond: list[Part] | None = None
    restart: list[Part] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator('respond', mode='before')
    @classmethod
    def _wrap_respond(cls, v: object) -> object:
        if v is None or not isinstance(v, list):
            return v
        return [as_resume_respond(p) for p in v]

    @field_validator('restart', mode='before')
    @classmethod
    def _wrap_restart(cls, v: object) -> object:
        if v is None or not isinstance(v, list):
            return v
        return [as_resume_restart(p) for p in v]


def as_resume(value: object) -> Resume:
    if isinstance(value, Resume):
        return Resume(
            respond=value.respond,
            restart=value.restart,
            metadata=value.metadata,
        )
    if isinstance(value, ResumeData):
        return Resume.model_validate(dump_keeping_unknown(value))
    return Resume.model_validate(value)


def _normalize_resume_parts(value: Part | list[Part] | None) -> list[Part] | None:
    if value is None:
        return None
    return list(value) if isinstance(value, list) else [value]


def resume_options_to_resume(
    *,
    resume_respond: Part | list[Part] | None = None,
    resume_restart: Part | list[Part] | None = None,
    resume_metadata: dict[str, Any] | None = None,
) -> Resume | None:
    """Build Resume from flat keyword options (``generate`` / prompts)."""
    respond = _normalize_resume_parts(resume_respond)
    restart = _normalize_resume_parts(resume_restart)
    if respond is None and restart is None and resume_metadata is None:
        return None
    return Resume(respond=respond, restart=restart, metadata=resume_metadata)


class Message(GenkitModel):
    """A single turn in a conversation."""

    role: Role | str
    content: list[Part]
    metadata: dict[str, Any] | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        if args:
            raise TypeError('Message takes keyword fields; use as_message to unwrap a wire message')
        super().__init__(**kwargs)

    @field_validator('content', mode='before')
    @classmethod
    def _wrap_parts(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_part(p) for p in v]

    @property
    def text(self) -> str:
        """All text parts concatenated into a single string."""
        return text_from_message(self)

    @property
    def tool_requests(self) -> list[Part]:
        """All tool request parts in this message."""
        return [p for p in self.content if p.tool_request is not None]

    @property
    def interrupts(self) -> list[Part]:
        """Tool requests marked as interrupted."""
        return [p for p in self.tool_requests if p.metadata and p.metadata.get('interrupt')]


class Candidate(GenkitModel):
    """One sampled reply from a generate call."""

    index: float
    message: Message
    usage: GenerationUsage | None = None
    finish_reason: FinishReason
    finish_message: str | None = None
    custom: Any | None = Field(default=None)

    @field_validator('message', mode='before')
    @classmethod
    def _wrap_message(cls, v: object) -> object:
        return as_message(v)


def as_candidate(value: object) -> Candidate:
    if isinstance(value, Candidate):
        return Candidate(
            index=value.index,
            message=value.message,
            usage=value.usage,
            finish_reason=value.finish_reason,
            finish_message=value.finish_message,
            custom=value.custom,
        )
    return Candidate.model_validate(value)


class GenerateActionOptions(GenkitModel):
    """Generate options with messages as list[Message] for type-safe use with ai.generate()."""

    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    docs: list[Document] | None = None
    tools: list[str] | None = None
    resources: list[str] | None = None
    tool_choice: ToolChoice | None = None
    config: Any | None = Field(default=None)
    output: GenerateActionOutputConfig | None = None
    resume: Resume | None = None
    return_tool_requests: bool | None = None
    max_turns: float | None = None
    step_name: str | None = None
    use: list[MiddlewareRef] | None = None

    @field_validator('messages', mode='before')
    @classmethod
    def _wrap_messages(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_message(m) for m in v]

    @field_validator('docs', mode='before')
    @classmethod
    def _wrap_docs(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_document(d) for d in v]

    @field_validator('resume', mode='before')
    @classmethod
    def _wrap_resume(cls, v: object) -> object:
        if v is None:
            return v
        return as_resume(v)


_TEXT_DATA_TYPE: str = 'text'


class Document(GenkitModel):
    """Multi-part document that can be embedded, indexed, or retrieved."""

    content: list[Part]
    metadata: dict[str, Any] | None = None

    @field_validator('content', mode='before')
    @classmethod
    def _wrap_parts(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_part(p) for p in v]

    def __init__(
        self,
        content: Sequence[Part],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with content parts and optional metadata."""
        if isinstance(content, (Document, DocumentData)) or (
            isinstance(content, BaseModel) and not isinstance(content, Sequence)
        ):
            raise TypeError('Document(other) is gone; pass content= or use as_document')
        payload: dict[str, Any] = {'content': deepcopy(content), 'metadata': deepcopy(metadata)}
        BaseModel.__init__(self, **cast(Any, payload))

    @staticmethod
    def from_text(text: str, metadata: dict[str, Any] | None = None) -> Document:
        """Create a document from a text string."""
        return Document(content=[Part.from_text(text)], metadata=metadata)

    @staticmethod
    def from_media(
        url: str,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Create a document from a media URL."""
        return Document(content=[Part.from_media(url, content_type)], metadata=metadata)

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
            if isinstance(p.text, str):
                texts.append(p.text)
        return ''.join(texts)

    @cached_property
    def media(self) -> list[Media]:
        """All media parts."""
        return [part.media for part in self.content if part.media is not None]

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


class Artifact(GenkitModel):
    """Named session file whose parts are the public Part type."""

    name: str | None = None
    parts: list[Part]
    metadata: dict[str, Any] | None = None

    @field_validator('parts', mode='before')
    @classmethod
    def _wrap_parts(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_part(p) for p in v]


class EmbedRequest(GenkitModel):
    """Embed request whose documents are the public Document type."""

    input: list[Document]
    options: Any | None = Field(default=None)

    @field_validator('input', mode='before')
    @classmethod
    def _wrap_docs(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_document(d) for d in v]


class SessionState(GenkitModel):
    """Session state whose conversation uses Message and Artifact."""

    session_id: str | None = None
    messages: list[Message] | None = None
    custom: Any | None = Field(default=None)
    artifacts: list[Artifact] | None = None

    @field_validator('messages', mode='before')
    @classmethod
    def _wrap_messages(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_message(m) for m in v]

    @field_validator('artifacts', mode='before')
    @classmethod
    def _wrap_artifacts(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_artifact(a) for a in v]


def as_session_state(value: object) -> SessionState:
    if isinstance(value, SessionState):
        return SessionState(
            session_id=value.session_id,
            messages=value.messages,
            custom=value.custom,
            artifacts=value.artifacts,
        )
    return SessionState.model_validate(value)


class SessionSnapshot(GenkitModel):
    """Snapshot whose state uses the public SessionState type."""

    snapshot_id: str
    session_id: str | None = None
    parent_id: str | None = None
    created_at: str
    updated_at: str | None = None
    heartbeat_at: str | None = None
    status: SnapshotStatus | None = None
    finish_reason: AgentFinishReason | None = None
    error: GenkitRuntimeError | None = None
    state: SessionState | None = None

    @field_validator('state', mode='before')
    @classmethod
    def _wrap_state(cls, v: object) -> object:
        if v is None:
            return v
        return as_session_state(v)


class AgentInit(GenkitModel):
    """Init payload whose state uses the public SessionState type."""

    session_id: str | None = None
    snapshot_id: str | None = None
    state: SessionState | None = None

    @field_validator('state', mode='before')
    @classmethod
    def _wrap_state(cls, v: object) -> object:
        if v is None:
            return v
        return as_session_state(v)


class AgentInput(GenkitModel):
    """Turn input whose message is the public Message type."""

    detach: bool | None = None
    message: Message | None = None
    resume: Resume | None = None

    @field_validator('message', mode='before')
    @classmethod
    def _wrap_message(cls, v: object) -> object:
        if v is None:
            return v
        return as_message(v)

    @field_validator('resume', mode='before')
    @classmethod
    def _wrap_resume(cls, v: object) -> object:
        if v is None:
            return v
        return as_resume(v)


class AgentOutput(GenkitModel):
    """Turn output whose message and artifacts are the public types."""

    session_id: str | None = None
    snapshot_id: str | None = None
    state: SessionState | None = None
    message: Message | None = None
    artifacts: list[Artifact] | None = None
    finish_reason: AgentFinishReason | None = None
    error: GenkitRuntimeError | None = None

    @field_validator('message', mode='before')
    @classmethod
    def _wrap_message(cls, v: object) -> object:
        if v is None:
            return v
        return as_message(v)

    @field_validator('artifacts', mode='before')
    @classmethod
    def _wrap_artifacts(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_artifact(a) for a in v]

    @field_validator('state', mode='before')
    @classmethod
    def _wrap_state(cls, v: object) -> object:
        if v is None:
            return v
        return as_session_state(v)


class AgentResult(GenkitModel):
    """Agent result whose message and artifacts are the public types."""

    message: Message | None = None
    artifacts: list[Artifact] | None = None
    finish_reason: AgentFinishReason | None = None

    @field_validator('message', mode='before')
    @classmethod
    def _wrap_message(cls, v: object) -> object:
        if v is None:
            return v
        return as_message(v)

    @field_validator('artifacts', mode='before')
    @classmethod
    def _wrap_artifacts(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_artifact(a) for a in v]


class OutputConfig(GenkitModel):
    """Output settings for a model request.

    Construct with ``json_schema=``; the serialized key on the wire is
    ``schema``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, extra='forbid', populate_by_name=True, protected_namespaces=()
    )
    format: str | None = None
    json_schema: dict[str, Any] | None = Field(default=None, validation_alias='schema', serialization_alias='schema')
    constrained: bool | None = None
    content_type: str | None = None


class ModelRequest(GenkitModel, Generic[ModelRequestConfigT]):
    """Hand-written model request with veneer types and flat output accessors.

    Output settings live nested as ``output: OutputConfig`` so dump/validate
    round-trips the wire shape, while flat properties (``output_format`` etc.)
    stay the plugin-author convenience surface. Messages and docs use veneer
    types (Message, Document) for helpers like ``.text``.

    Example:
        from genkit.plugin_api import ModelConfig

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
    messages: list[Message]
    docs: list[Document] | None = None
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
    def _wrap_messages(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_message(m) for m in v]

    @field_validator('docs', mode='before')
    @classmethod
    def _wrap_docs(cls, v: object) -> object:
        """A dumped request sends docs as dicts.

        Messages already take a mapping; this wrap has to as well or a bad
        config plus docs= never reaches the GenkitError for the config.
        """
        if v is None:
            return None
        if not isinstance(v, list):
            return v
        return [as_document(d) for d in v]

    @field_validator('output', mode='before')
    @classmethod
    def _wrap_output(cls, v: object) -> object:
        if v is None:
            return v
        return as_output_config(v)

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


def as_model_request(value: object) -> ModelRequest:
    if isinstance(value, ModelRequest):
        return ModelRequest.model_validate(dump_keeping_unknown(value))
    return ModelRequest.model_validate(value)


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

    @field_validator('message', mode='before')
    @classmethod
    def _wrap_message(cls, v: object) -> object:
        if v is None:
            return v
        return as_message(v)

    @field_validator('request', mode='before')
    @classmethod
    def _wrap_request(cls, v: object) -> object:
        if v is None:
            return v
        return as_model_request(v)

    @field_validator('candidates', mode='before')
    @classmethod
    def _wrap_candidates(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_candidate(c) for c in v]

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
            return [as_message(m) for m in self.request.messages] if self.request else []
        return [
            *(as_message(m) for m in (self.request.messages if self.request else [])),
            self.message,
        ]

    @property
    def tool_requests(self) -> list[Part]:
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
        return [part.media for part in self.message.content if part.media is not None]

    @property
    def interrupts(self) -> list[Part]:
        """Tool requests marked as interrupted."""
        if self.message is None:
            return []
        return self.message.interrupts


class ModelResponseChunk(GenkitModel, Generic[OutputT]):
    """Streaming chunk with text, accumulated text, and output parsing."""

    role: Any | None = Field(default=None)
    index: float | None = None
    content: list[Part]
    custom: Any | None = Field(default=None)
    aggregated: bool | None = None
    previous_chunks: list[Any] = Field(default_factory=list, exclude=True)
    chunk_parser: Callable[..., object] | None = Field(default=None, exclude=True)
    schema_type: type[BaseModel] | None = Field(default=None, exclude=True)

    @field_validator('content', mode='before')
    @classmethod
    def _wrap_parts(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_part(p) for p in v]

    def __init__(
        self,
        chunk: ModelResponseChunk[Any] | None = None,
        previous_chunks: list[Any] | None = None,
        index: int | float | None = None,
        chunk_parser: Callable[..., object] | None = None,
        schema_type: type[BaseModel] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize from a chunk or keyword arguments."""
        if chunk is not None:
            payload: dict[str, Any] = {
                'role': chunk.role,
                'index': index,
                'content': chunk.content,
                'custom': chunk.custom,
                'aggregated': chunk.aggregated,
            }
            BaseModel.__init__(self, **cast(Any, payload))
        else:
            if index is not None:
                kwargs.setdefault('index', index)
            if previous_chunks is not None:
                kwargs.setdefault('previous_chunks', previous_chunks)
            if chunk_parser is not None:
                kwargs.setdefault('chunk_parser', chunk_parser)
            if schema_type is not None:
                kwargs.setdefault('schema_type', schema_type)
            BaseModel.__init__(self, **cast(Any, kwargs))
        self.previous_chunks = previous_chunks if previous_chunks is not None else list(self.previous_chunks or [])
        if chunk_parser is not None:
            self.chunk_parser = chunk_parser
        if schema_type is not None:
            self.schema_type = schema_type

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
        return ''.join(p.text for p in self.content if p.text is not None)

    @property
    def accumulated_text(self) -> str:
        """Text from all previous chunks plus this chunk."""
        prior = ''
        if self.previous_chunks:
            prior = ''.join(p.text for chunk in self.previous_chunks for p in chunk.content if p.text)
        return prior + self.text

    @cached_property
    def output(self) -> OutputT | None:
        """Parsed output from accumulated text.

        With no ``output_schema`` class, this is the extracted JSON value
        (a dict, list, scalar, or ``None`` if an object has not started).

        When ``output_schema`` is a Pydantic model, this is an instance of
        that class with missing fields set to ``None``. Values may still be
        prefixes, and constraints are not enforced. Guard each field you
        use. ``(await sr.response).output`` is the only fully validated value.
        """
        parsed = (
            self.chunk_parser(self)
            if self.chunk_parser
            else extract_json(self.accumulated_text, throw_on_bad_json=False)
        )
        if self.schema_type is not None and isinstance(parsed, dict) and not issubclass(self.schema_type, RootModel):
            return cast(
                'OutputT | None',
                construct_partial(schema_type=self.schema_type, data=parsed),
            )
        return cast('OutputT | None', parsed)


def as_model_response_chunk(value: object) -> ModelResponseChunk:
    if isinstance(value, ModelResponseChunk):
        return ModelResponseChunk(
            chunk=value,
            index=value.index,
            previous_chunks=value.previous_chunks,
            chunk_parser=value.chunk_parser,
            schema_type=value.schema_type,
        )
    return ModelResponseChunk.model_validate(value)


class AgentStreamChunk(GenkitModel):
    """One streamed piece of an agent turn."""

    model_chunk: ModelResponseChunk | None = None
    custom_patch: JsonPatch | None = None
    artifact: Artifact | None = None
    turn_end: TurnEnd | None = None

    @field_validator('model_chunk', mode='before')
    @classmethod
    def _wrap_chunk(cls, v: object) -> object:
        if v is None:
            return v
        return as_model_response_chunk(v)

    @field_validator('artifact', mode='before')
    @classmethod
    def _wrap_artifact(cls, v: object) -> object:
        if v is None:
            return v
        return as_artifact(v)


_VENEEER_NS = {**vars(typing_mod), **globals()}
Artifact.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
EmbedRequest.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
SessionState.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
SessionSnapshot.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
AgentInit.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
AgentInput.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
AgentOutput.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
AgentResult.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
Candidate.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
ModelResponse.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
ModelResponseChunk.model_rebuild(force=True, _types_namespace=_VENEEER_NS)
AgentStreamChunk.model_rebuild(force=True, _types_namespace=_VENEEER_NS)


class MultipartToolResponse(GenkitModel, Generic[OutputT]):
    """What ``Action.run()`` returns: ``output`` plus optional media.

    People annotate ``MultipartToolResponse[ShotOut]`` so the model binds
    ``ShotOut``. ``run()`` is still this envelope.
    """

    output: OutputT | None = None
    content: list[Part] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator('content', mode='before')
    @classmethod
    def _wrap_parts(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        return [as_part(p) for p in v]


def text_from_message(msg: Message) -> str:
    """Concatenate text from all parts of a message."""
    return text_from_content(msg.content)


def text_from_content(content: Sequence[Part]) -> str:
    """Concatenate text parts.

    Thoughts ride on a reasoning part, so they stay out of ``.text`` —
    that's the visible reply, not the model's scratch work.
    """
    texts: list[str] = []
    for p in content:
        if p.text is not None:
            texts.append(str(p.text))
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
            if part.text:
                characters += len(part.text)

            media = part.media
            if media:
                content_type = media.content_type or ''
                url = media.url or ''

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
