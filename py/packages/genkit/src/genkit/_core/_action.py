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

"""Action module for defining and managing remotely callable functions."""

import asyncio
import inspect
import json
import re
import sys
import time
import types
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from typing import (
    Any,
    ClassVar,
    Generic,
    NamedTuple,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic.alias_generators import to_camel
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError, PydanticUserError
from typing_extensions import TypeVar

from genkit._core._channel import Channel, CloseableQueue
from genkit._core._compat import StrEnum
from genkit._core._error import GenkitError, Interrupt, RuntimeErrorReason
from genkit._core._model import config_type_path, declared_config_type
from genkit._core._schema import to_json_schema
from genkit._core._telemetry._attrs import Attr, metadata_key
from genkit._core._telemetry._instrumentation import (
    SpanContext,
    run_in_new_span,
    to_json_attr,
)
from genkit._core._typing import Operation

# =============================================================================
# Span attribute types and tracing helpers
# =============================================================================

SpanAttributeValue = str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]


def _record_latency(output: object, latency_ms: float) -> object:
    """Stamp ``latency_ms`` on the output if it has one (in place, or via ``model_copy`` for frozen models)."""
    if hasattr(output, 'latency_ms'):
        try:
            cast(Any, output).latency_ms = latency_ms
        except (TypeError, ValidationError, AttributeError):
            # Frozen Pydantic models reject in-place assignment; fall back to model_copy.
            if hasattr(output, 'model_copy'):
                output = cast(Any, output).model_copy(update={'latency_ms': latency_ms})
    return output


def _sanitize_value(val: object, seen: set[int] | None = None) -> object:
    """Recursively filter out dictionary keys or list items that cannot be serialized to JSON."""
    if seen is None:
        seen = set()

    ref_id = id(val)
    if ref_id in seen:
        return '[Circular]'

    if isinstance(val, dict):
        seen.add(ref_id)
        sanitized = {}
        for k, v in val.items():
            if not isinstance(k, str):
                k = str(k)
            try:
                sanitized[k] = _sanitize_value(v, seen)
            except (TypeError, ValueError):
                sanitized[k] = repr(v)
        seen.remove(ref_id)
        return sanitized
    elif isinstance(val, (list, set, tuple)):
        seen.add(ref_id)
        sanitized_list = []
        for item in val:
            try:
                sanitized_list.append(_sanitize_value(item, seen))
            except (TypeError, ValueError):
                sanitized_list.append(repr(item))
        seen.remove(ref_id)
        return sanitized_list
    else:
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        try:
            json.dumps(val)
            return val
        except (TypeError, ValueError):
            return repr(val)


def context_for_telemetry(context: dict[str, Any]) -> dict[str, Any]:
    """Copy of action context for the Dev UI Context panel.

    ``auth`` and ``secrets`` are what the caller put on the request for
    identity and keys. The live action still sees the real values; the
    panel should not.
    """
    # Sanitize on the caller's dict so a self-pointer becomes '[Circular]'
    # instead of one extra unwrap on the panel.
    cleaned = _sanitize_value(context)
    traced = dict(cleaned) if isinstance(cleaned, dict) else {}
    if 'auth' in traced:
        traced['auth'] = '<redacted>'
    if 'secrets' in traced:
        traced['secrets'] = '<redacted>'
    return traced


# =============================================================================
# Action types
# =============================================================================

# Type alias for action name.
ActionName = str


class ActionKind(StrEnum):
    """Types of actions that can be registered."""

    BACKGROUND_MODEL = 'background-model'
    AGENT = 'agent'
    AGENT_ABORT = 'agent-abort'
    AGENT_SNAPSHOT = 'agent-snapshot'
    CANCEL_OPERATION = 'cancel-operation'
    CHECK_OPERATION = 'check-operation'
    CUSTOM = 'custom'
    DYNAMIC_ACTION_PROVIDER = 'dynamic-action-provider'
    EMBEDDER = 'embedder'
    EVALUATOR = 'evaluator'
    EXECUTABLE_PROMPT = 'executable-prompt'
    FLOW = 'flow'
    INDEXER = 'indexer'
    MODEL = 'model'
    PROMPT = 'prompt'
    RERANKER = 'reranker'
    RETRIEVER = 'retriever'
    # Catalog key for tools. Action.run / Dev UI see the multipart envelope.
    TOOL = 'tool.v2'
    UTIL = 'util'


ResponseT = TypeVar('ResponseT')


class ActionResponse(BaseModel, Generic[ResponseT]):
    """Response from an action with trace ID."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra='forbid', populate_by_name=True, alias_generator=to_camel, arbitrary_types_allowed=True
    )

    response: ResponseT
    trace_id: str
    span_id: str = ''
    latency_ms: float | None = None


ChunkT_co = TypeVar('ChunkT_co', covariant=True)
OutputT_co = TypeVar('OutputT_co', covariant=True)


class StreamResponse(Generic[ChunkT_co, OutputT_co]):
    """Wrapper for streaming action results."""

    def __init__(
        self,
        stream: AsyncIterator[ChunkT_co],
        response: Awaitable[OutputT_co],
    ) -> None:
        self._stream = stream
        self._response = response

    @property
    def stream(self) -> AsyncIterator[ChunkT_co]:
        return self._stream

    @property
    def response(self) -> Awaitable[OutputT_co]:
        return self._response


class ActionMetadataKey(StrEnum):
    """Keys for action metadata."""

    INPUT_KEY = 'inputSchema'
    OUTPUT_KEY = 'outputSchema'
    INIT_KEY = 'initSchema'
    RETURN = 'return'


# =============================================================================
# Action utilities
# =============================================================================


def noop_streaming_callback(_chunk: Any) -> None:  # noqa: ANN401
    pass


def get_func_description(func: Callable[..., Any], description: str | None = None) -> str:
    """Get description from explicit param or function docstring."""
    if description is not None:
        return description
    return func.__doc__ or ''


def parse_plugin_name_from_action_name(name: str) -> str | None:
    """Extract plugin namespace from 'plugin/action' format."""
    tokens = name.split('/')
    if len(tokens) > 1:
        return tokens[0]
    return None


# =============================================================================
# Reading an action's signature
#
# An action function takes at most one input and at most one run context:
#
#   async def lookup(order: Order, ctx: ActionRunContext) -> Receipt: ...
#   async def lookup(ctx: ActionRunContext, order: Order) -> Receipt: ...  # same
#   async def ping(ctx: ActionRunContext) -> str: ...                      # no input
#
# The context is whichever parameter is annotated ActionRunContext or a
# subclass (ToolRunContext, ...). The other one is the input. Genkit calls the
# function with both by name (ActionParams.call), so order never matters.
#
# A TypeError when an action is defined comes from one of two functions:
#   - find_input_and_context: the signature's shape (a second input, two
#     contexts, a positional-only parameter, a tool or flow input with no type)
#   - json_schema_for: an input or output type with no JSON schema, or one
#     whose name Genkit can't find at runtime
# =============================================================================

_CallT = TypeVar('_CallT')


class ActionParams(NamedTuple):
    """An action function's input and run-context parameters, either may be absent."""

    input: inspect.Parameter | None
    context: inspect.Parameter | None

    @property
    def input_optional(self) -> bool:
        """True if the input has a default, so the action can run without one."""
        return self.input is not None and self.input.default is not inspect.Parameter.empty

    def call(self, fn: Callable[..., _CallT], input: object, ctx: 'ActionRunContext') -> _CallT:  # noqa: A002
        """Call ``fn`` with ``input`` and ``ctx`` passed by parameter name.

        A missing input is left out when the parameter has a default, so the
        default applies.
        """
        kwargs: dict[str, object] = {}
        if self.input is not None and not (input is None and self.input_optional):
            kwargs[self.input.name] = input
        if self.context is not None:
            kwargs[self.context.name] = ctx
        return fn(**kwargs)


def _kind_label(kind: ActionKind) -> str:
    """The action kind as error messages say it."""
    # ActionKind.TOOL is 'tool.v2' (the catalog key); people call it a tool.
    return 'tool' if kind == ActionKind.TOOL else str(kind)


def describe_action(kind: ActionKind, name: str) -> str:
    """How error messages name an action, e.g. ``"tool 'weather'"``."""
    return f"{_kind_label(kind)} '{name}'"


def find_input_and_context(
    fn: Callable[..., object],
    hints: Mapping[str, Any],
    *,
    kind: ActionKind,
    name: str,
) -> ActionParams:
    """Find ``fn``'s input and run-context parameters, or raise a TypeError saying how to fix it.

    ``hints`` is ``resolve_type_hints(fn)``. ``*args`` and ``**kwargs`` are ignored.
    """
    owner = describe_action(kind, name)
    context_class = 'ToolRunContext' if kind == ActionKind.TOOL else 'ActionRunContext'

    params = [
        p
        for p in signature_of(fn).parameters.values()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    for p in params:
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{owner} parameter '{p.name}' is positional-only, but Genkit passes the input "
                "and context by name. Remove the '/' from the signature."
            )

    contexts: list[inspect.Parameter] = []
    inputs: list[inspect.Parameter] = []
    for p in params:
        if _is_context_annotation(hints.get(p.name, p.annotation)):
            contexts.append(p)
        else:
            inputs.append(p)

    if len(contexts) > 1:
        first, second = contexts[0].name, contexts[1].name
        raise TypeError(f"{owner} has two {context_class} parameters, '{first}' and '{second}'. Keep one.")
    if len(inputs) > 1:
        extra = inputs[1].name
        raise TypeError(
            f"{owner} takes one input, but '{extra}' is a second parameter. "
            f"Put the fields on one input model, or annotate '{extra}' as {context_class}."
        )

    input_param = inputs[0] if inputs else None
    context_param = contexts[0] if contexts else None

    # A tool's input type is the schema the model sees, and a flow's is its
    # public API, so those need one. Other actions (models, embedders, ...)
    # get a fixed input from Genkit and may leave it unannotated.
    if kind in (ActionKind.TOOL, ActionKind.FLOW) and input_param is not None:
        if hints.get(input_param.name, input_param.annotation) is inspect.Parameter.empty:
            raise TypeError(
                f"{owner} input '{input_param.name}' has no type annotation. "
                f"Annotate it (e.g. '{input_param.name}: str'), or use Any to accept anything."
            )

    return ActionParams(input_param, context_param)


def json_schema_for(
    annotation: object,
    *,
    kind: ActionKind,
    name: str,
    label: str,
) -> tuple[TypeAdapter[Any], dict[str, object]]:
    """A validator and JSON schema for an action's input or output type.

    ``label`` says which one, e.g. ``"input 'query'"`` or ``'output'``.
    Pydantic's own errors here name neither the action nor the fix, so each
    one is re-raised as a TypeError that does.
    """
    owner = describe_action(kind, name)
    type_name = getattr(annotation, '__name__', repr(annotation))
    try:
        adapter: TypeAdapter[Any] = TypeAdapter(annotation)
        return adapter, adapter.json_schema()
    except (PydanticSchemaGenerationError, PydanticInvalidForJsonSchema) as e:
        # e.g. `-> Thermometer`, where Thermometer is a plain class
        raise TypeError(
            f'{owner} {label} has type {type_name}, which has no JSON schema. '
            'Use a Pydantic model, dataclass, TypedDict, or a basic type like str, int, list, or dict.'
        ) from e
    except PydanticUserError as e:
        if isinstance(annotation, str):
            # resolve_type_hints couldn't find this name, so it's still a
            # string. Usually the file has `from __future__ import annotations`
            # and the type is a class defined inside a function, or imported
            # only under `if TYPE_CHECKING:`.
            raise TypeError(
                f"{owner} {label} has type '{annotation}', which can't be found "
                f'when the {_kind_label(kind)} is defined. Define or import it at module level (outside '
                "'if TYPE_CHECKING:'), or remove 'from __future__ import annotations' from this file."
            ) from e
        # e.g. typing.TypedDict on Python < 3.12; keep Pydantic's fix in the message
        raise TypeError(f'{owner} {label} has type {type_name}: {e.message}') from e


# -----------------------------------------------------------------------------
# Annotations
#
# A name in an annotation can be missing at runtime: imported only under
# `if TYPE_CHECKING:` (Ruff's TC rules do this), or a class defined inside a
# function in a file with `from __future__ import annotations`. These helpers
# keep such a name as a string instead of failing, so the rest still resolves.
# -----------------------------------------------------------------------------


def signature_of(fn: Callable[..., object]) -> inspect.Signature:
    """``inspect.signature(fn)``, without failing on a name missing at runtime.

    From Python 3.14 annotations are evaluated when read, so a plain
    ``inspect.signature`` raises on a TYPE_CHECKING-only ``ctx: ToolRunContext``.
    """
    if sys.version_info >= (3, 14):
        import annotationlib

        return inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)
    return inspect.signature(fn)


def resolve_type_hints(fn: Callable[..., object]) -> dict[str, Any]:
    """``fn``'s annotations as types. A name that can't be found stays a string.

    ``get_type_hints`` fails outright if any one name is missing, so then each
    annotation is resolved on its own. A missing context class doesn't also
    hide the input model.
    """
    try:
        return get_type_hints(fn)
    except Exception:
        module_globals = getattr(inspect.unwrap(fn), '__globals__', {})
        return {name: _resolve_one(a, module_globals) for name, a in _annotations_as_written(fn).items()}


def _annotations_as_written(fn: Callable[..., object]) -> dict[str, Any]:
    """``fn``'s annotations without evaluating them; a missing name is a string."""
    if sys.version_info >= (3, 14):
        import annotationlib

        annotations = annotationlib.get_annotations(fn, format=annotationlib.Format.FORWARDREF)
        return {
            name: a.__forward_arg__ if isinstance(a, annotationlib.ForwardRef) else a for name, a in annotations.items()
        }
    return dict(inspect.getfullargspec(fn).annotations)


# Code object of an empty function, for _resolve_one.
_EMPTY_FUNCTION_CODE = (lambda: None).__code__


def _resolve_one(annotation: object, module_globals: dict[str, Any]) -> object:
    """Resolve one annotation the way ``get_type_hints`` would, or return it unchanged."""
    # get_type_hints takes a function, so give it an empty one in fn's module
    # whose only annotation is this one.
    stand_in = types.FunctionType(_EMPTY_FUNCTION_CODE, module_globals)
    stand_in.__annotations__ = {'x': annotation}
    try:
        return get_type_hints(stand_in)['x']
    except Exception:
        return annotation


def _is_context_annotation(annotation: object) -> bool:
    """True if ``annotation`` marks the run-context parameter.

    That's ActionRunContext or any subclass, written any of these ways:
    ``ToolRunContext``, ``ActionRunContext[str]``, ``ToolRunContext | None``,
    or as a string (see _string_names_context).
    """
    # On 3.14 a name missing at runtime is a ForwardRef; use its text.
    forward_arg = getattr(annotation, '__forward_arg__', None)
    if isinstance(forward_arg, str):
        annotation = forward_arg
    if isinstance(annotation, str):
        return _string_names_context(annotation)

    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(_is_context_annotation(arg) for arg in get_args(annotation))
    # ActionRunContext[str] -> ActionRunContext
    cls = origin if origin is not None else annotation
    return isinstance(cls, type) and issubclass(cls, ActionRunContext)


# ActionRunContext and every subclass, by class name. ActionRunContext.__init_subclass__
# adds each one, so a string annotation can be matched without importing it.
_CONTEXT_CLASS_NAMES: set[str] = {'ActionRunContext'}


def _string_names_context(annotation: str) -> bool:
    """True if a string annotation names a run-context class.

    'ToolRunContext', 'ToolRunContext | None', 'Optional[ToolRunContext]',
    'ActionRunContext[str]' and 'genkit.ToolRunContext' all do. The class
    doesn't have to be importable here, only listed in _CONTEXT_CLASS_NAMES.
    """
    for part in annotation.split('|'):
        name = part.strip().strip('\'"')
        for prefix in ('Optional[', 'typing.Optional['):
            if name.startswith(prefix) and name.endswith(']'):
                name = name[len(prefix) : -1].strip()  # Optional[X] -> X
        name = name.split('[', 1)[0]  # ActionRunContext[str] -> ActionRunContext
        name = name.rsplit('.', 1)[-1]  # genkit.ToolRunContext -> ToolRunContext
        if name in _CONTEXT_CLASS_NAMES:
            return True
    return False


# =============================================================================
# Action key utilities
# =============================================================================


# Attribute name used to attach a ``DynamicActionProvider`` (cache + helpers)
# onto the placeholder ``Action`` registered for a DAP. The registry only
# stores the ``Action``; the provider rides along on it as a Python attribute.
# Code holding the ``Action`` recovers the provider via
# ``getattr(action, GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR, None)``.
GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR = '_genkit_dynamic_action_provider'


class DapQualifiedName(NamedTuple):
    """Segments of a DAP-qualified name ``provider:innerKind/innerName``."""

    provider: str
    inner_kind: str
    inner_name: str


def parse_dap_qualified_name(name: str) -> DapQualifiedName | None:
    """Parse DAP-qualified segment ``provider:innerKind/innerName``.

    Used when the action key kind is ``dynamic-action-provider`` and the name
    references a nested action exposed by a provider (e.g. MCP tools).

    Pattern: ``[provider]:[inner_kind]/[inner_name]`` — no slashes in the
    provider segment (``plugin/foo`` is not a valid provider host).

    Returns:
        A :class:`DapQualifiedName` if the string matches; otherwise ``None`` so
        callers can treat the name as a plain dynamic-action-provider id.
    """
    # Pattern: [provider]:[inner_kind]/[inner_name]; no '/' or ':' in provider.
    match = re.match(r'^([^/:]+):([^/:]+)/(.+)$', name)
    if not match:
        return None
    provider, inner_kind, inner_name = match.groups()
    if not provider or not inner_kind or not inner_name:
        return None
    # Catalog kind, not a selector. People write provider:tool/name.
    if inner_kind == ActionKind.TOOL:
        return None
    return DapQualifiedName(provider, inner_kind, inner_name)


def parse_action_key(key: str) -> tuple[ActionKind, str]:
    """Parse '/<kind>/<name>' key into (ActionKind, name)."""
    tokens = key.split('/')
    if len(tokens) < 3 or not tokens[1] or not tokens[2]:
        msg = f'Invalid action key format: `{key}`.Expected format: `/<kind>/<name>`'
        raise ValueError(msg)

    kind_str = tokens[1]
    name = '/'.join(tokens[2:])
    try:
        kind = ActionKind(kind_str)
    except ValueError as e:
        msg = f'Invalid action kind: `{kind_str}`'
        raise ValueError(msg) from e
    # pyrefly: ignore[bad-return] - ActionKind is StrEnum subclass, pyrefly doesn't narrow properly
    return kind, name


def create_action_key(kind: ActionKind | str, name: str) -> str:
    """Create '/<kind>/<name>' key."""
    return f'/{kind}/{name}'


def stamp_background_operation(*, output: object, kind: ActionKind, name: str) -> None:
    """Point an Operation from a background model's start/check/cancel at its start action.

    A caller polls by passing the Operation back, and Genkit finds the check
    action from ``op.action``. For a model named 'veo', start ('veo'), check
    ('veo/check') and cancel ('veo/cancel') all stamp '/background-model/veo'.
    A value the function already set is kept.
    """
    if not isinstance(output, Operation) or output.action:
        return
    if kind == ActionKind.BACKGROUND_MODEL:
        model_name = name
    elif kind in (ActionKind.CHECK_OPERATION, ActionKind.CANCEL_OPERATION):
        model_name = name.rsplit('/', 1)[0]  # 'veo/check' -> 'veo'
    else:
        return
    output.action = create_action_key(ActionKind.BACKGROUND_MODEL, model_name)


# =============================================================================
# Action core
# =============================================================================

InputT = TypeVar('InputT', default=Any)
OutputT = TypeVar('OutputT', default=Any)
ChunkT = TypeVar('ChunkT', default=Any)
InitT = TypeVar('InitT', default=Any)

# Generic streaming callback - use Callable[[ChunkT], None] for typed chunks
# This untyped version is for internal use where chunk type is unknown
StreamingCallback = Callable[[object], None]

# A bidi fn is (init, incoming per-turn inputs, chunk sink) -> output. init is
# the session identity for the whole connection; input_stream yields the per-turn
# inputs (one item for a one-shot call, many for a live chat) and send_chunk emits
# streamed chunks. Keeping init in its own slot is what lets one connection span
# many typed message turns. This is the same shape a plain action fn sees on its
# ctx (input stream + send_chunk), so bidi fns don't need any queue plumbing.
BidiFn = Callable[
    [InitT, AsyncIterator[InputT], Callable[[ChunkT], None]],
    Awaitable[OutputT],
]

_action_context: ContextVar[dict[str, Any] | None] = ContextVar('context')
_ = _action_context.set(None)


class ActionRunContext(Generic[ChunkT]):
    """Execution context for an action.

    Provides read-only access to action context (auth, metadata), streaming
    support, and an abort signal for cooperative cancellation.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        # Lets a string annotation like 'ToolRunContext' mark the context.
        _CONTEXT_CLASS_NAMES.add(cls.__name__)

    def __init__(
        self,
        context: dict[str, Any] | None = None,
        streaming_callback: Callable[[ChunkT], None] | None = None,
        abort_signal: asyncio.Event | None = None,
        init: object | None = None,
        input_stream: AsyncIterator[object] | None = None,
    ) -> None:
        self._context: dict[str, Any] = context if context is not None else {}
        self._streaming_callback = streaming_callback
        self.abort_signal: asyncio.Event = abort_signal if abort_signal is not None else asyncio.Event()
        self._init = init
        self._input_stream = input_stream

    @property
    def context(self) -> dict[str, Any]:
        """The action context."""
        return self._context

    @property
    def init(self) -> object | None:
        """The session initialization value passed on connection open, if any.

        For request-response actions this is None; for bidi streams (like live
        agent chat) this carries whatever credentials/metadata the caller sent
        in the handshake before any turns started.
        """
        return self._init

    @property
    def input_stream(self) -> AsyncIterator[object] | None:
        """The incoming input stream for bidi actions, if any.

        An action that receives inputs continuously across a session (e.g. live
        agent chat) instead gets its turns over time here. Plain actions never
        look at it — only bidi actions drain it turn by turn.
        """
        return self._input_stream

    @property
    def is_streaming(self) -> bool:
        """True if a streaming callback is registered."""
        return self._streaming_callback is not None

    @property
    def streaming_callback(self) -> Callable[[ChunkT], None] | None:
        """The streaming callback, if any.

        Use this when you need to pass the callback to another action.
        For sending chunks directly, use send_chunk() instead.
        """
        return self._streaming_callback

    def send_chunk(self, chunk: ChunkT) -> None:
        """Send a streaming chunk to the client.

        Args:
            chunk: The chunk data to stream.
        """
        if self._streaming_callback is not None:
            self._streaming_callback(chunk)

    @staticmethod
    def _current_context() -> dict[str, Any] | None:
        return _action_context.get(None)


class Action(Generic[InputT, OutputT, ChunkT, InitT]):
    """A named, traced, remotely callable function."""

    def __init__(
        self,
        kind: ActionKind,
        name: str,
        fn: Callable[..., Awaitable[OutputT]],
        metadata_fn: Callable[..., object] | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
        span_metadata: dict[str, SpanAttributeValue] | None = None,
        init_schema: type[BaseModel] | dict[str, object] | None = None,
        config_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None:
        self._kind: ActionKind = kind
        self._name: str = name
        self._metadata: dict[str, object] = metadata if metadata else {}
        self._description: str | None = description
        # Python class for generate's isinstance check. Not in metadata —
        # that bag is JSON for the Dev UI.
        self._config_schema: type[BaseModel] | None = (
            config_schema if isinstance(config_schema, type) and issubclass(config_schema, BaseModel) else None
        )
        self._span_metadata: dict[str, SpanAttributeValue] = span_metadata or {}

        # All action handlers must be async
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"Action handlers must be async functions. Got sync function for '{name}'.")

        # Genkit calls fn. With a metadata_fn, fn is a wrapper (the tool wrapper
        # is one) and metadata_fn is the user's function, whose signature
        # decides the input and context.
        user_fn = metadata_fn if metadata_fn else fn
        hints = resolve_type_hints(user_fn)
        self._params: ActionParams = find_input_and_context(user_fn, hints, kind=kind, name=name)
        self._fn: Callable[..., Awaitable[OutputT]] = fn
        self._fn_is_wrapper: bool = metadata_fn is not None
        self._initialize_io_schemas(hints)
        self._initialize_init_schema(init_schema)

    @property
    def params(self) -> ActionParams:
        """The function's input and context parameters, as Genkit passes them."""
        return self._params

    @property
    def kind(self) -> ActionKind:
        return self._kind

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def metadata(self) -> dict[str, object]:
        return self._metadata

    @property
    def input_type(self) -> TypeAdapter[InputT] | None:
        return self._input_type

    @property
    def input_class(self) -> type | None:
        """The action's input annotation as a concrete class, when it is one."""
        return getattr(self, '_input_class', None)

    @property
    def input_schema(self) -> dict[str, object]:
        return self._input_schema

    @input_schema.setter
    def input_schema(self, value: dict[str, object]) -> None:
        self._input_schema = value
        self._metadata[ActionMetadataKey.INPUT_KEY] = value

    @property
    def output_schema(self) -> dict[str, object]:
        return self._output_schema

    @output_schema.setter
    def output_schema(self, value: dict[str, object]) -> None:
        self._output_schema = value
        self._metadata[ActionMetadataKey.OUTPUT_KEY] = value

    def _override_input_schema(
        self,
        input_schema: type[BaseModel] | dict[str, object],
    ) -> None:
        """Replace inferred input JSON Schema and validation type (e.g. tool schema overrides)."""
        in_js = to_json_schema(input_schema)
        self.input_schema = in_js
        if isinstance(input_schema, dict):
            self._input_type = None
        else:
            self._input_type = cast(TypeAdapter[InputT], TypeAdapter(input_schema))

    async def __call__(self, input: InputT | None = None) -> OutputT:
        """Call the action directly, returning just the response value."""
        return (await self.run(input)).response

    async def run(
        self,
        input: InputT | None = None,
        on_chunk: Callable[[ChunkT], None] | None = None,
        context: dict[str, Any] | None = None,
        on_trace_start: Callable[[str, str], Awaitable[None]] | None = None,
        telemetry_labels: dict[str, object] | None = None,
        abort_signal: asyncio.Event | None = None,
        init: InitT | None = None,
        input_stream: AsyncIterator[InputT] | None = None,
    ) -> ActionResponse[OutputT]:
        """Execute the action with optional input validation.

        Args:
            input: The input to the action. Will be validated against the input schema.
            on_chunk: Optional streaming callback for chunked responses.
            context: Optional context dict for the action.
            on_trace_start: Optional callback invoked when trace starts.
            telemetry_labels: Custom labels to set as direct span attributes.
            abort_signal: Optional shared abort event for cooperative cancellation.
            init: Optional per-run initialization data (e.g. an agent's session
                identity). Validated against the init schema and exposed to the
                action fn via ``ActionRunContext.init``; plain actions ignore it.
            input_stream: Optional live stream of per-turn inputs for a bidi
                action. When omitted, a one-shot run is exactly the single ``input``;
                only bidi actions read it. Exposed via ``ActionRunContext.input_stream``.

        Returns:
            ActionResponse containing the result and trace metadata.

        Raises:
            GenkitError: If input validation fails (INVALID_ARGUMENT status).
        """
        # With a live input_stream, `input` isn't the payload — the stream
        # carries the per-turn inputs — so there's nothing to validate up front.
        if input_stream is None:
            input = self._validate_input(input)
        init = self._validate_init(init)

        token = None
        if context is not None:
            token = _action_context.set(context)

        streaming_cb = cast(StreamingCallback, on_chunk) if on_chunk else None

        try:
            return await self._run_with_telemetry(
                input,
                ActionRunContext(
                    context=_action_context.get(None),
                    streaming_callback=streaming_cb,
                    abort_signal=abort_signal,
                    init=init,
                    input_stream=input_stream,
                ),
                on_trace_start,
                telemetry_labels,
            )
        finally:
            if token is not None:
                _action_context.reset(token)

    def stream(
        self,
        input: InputT | None = None,
        context: dict[str, Any] | None = None,
        telemetry_labels: dict[str, object] | None = None,
        init: InitT | None = None,
        input_stream: AsyncIterator[InputT] | None = None,
    ) -> StreamResponse[ChunkT, OutputT]:
        """Execute and return a StreamResponse with .stream and .response properties."""
        channel: Channel[ChunkT, ActionResponse[OutputT]] = Channel()

        def send_chunk(c: ChunkT) -> None:
            channel.send(c)

        resp = self.run(
            input=input,
            context=context,
            telemetry_labels=telemetry_labels,
            on_chunk=send_chunk,
            init=init,
            input_stream=input_stream,
        )
        channel.set_close_future(asyncio.create_task(resp))

        # Mirror the run's terminal state onto .response so a caller awaiting it
        # sees the same success/error/cancel the run ended with, instead of
        # hanging (or dropping the error on the floor) when the run raises.
        result_future: asyncio.Future[OutputT] = asyncio.Future()

        def _resolve_response(closed: asyncio.Future[ActionResponse[OutputT]]) -> None:
            if result_future.done():
                return
            if closed.cancelled():
                result_future.cancel()
            elif (exc := closed.exception()) is not None:
                result_future.set_exception(exc)
            else:
                result_future.set_result(closed.result().response)

        channel.closed.add_done_callback(_resolve_response)

        return StreamResponse(stream=channel, response=result_future)

    def _initialize_io_schemas(self, annotations: dict[str, Any]) -> None:
        if self._params.input is not None:
            input_type = annotations.get(self._params.input.name, Any)
            type_adapter, self._input_schema = json_schema_for(
                input_type, kind=self._kind, name=self._name, label=f"input '{self._params.input.name}'"
            )
            self._input_type: TypeAdapter[InputT] | None = cast(TypeAdapter[InputT], type_adapter)
            self._input_class: type | None = input_type if isinstance(input_type, type) else None
        else:
            self._input_schema = TypeAdapter(object).json_schema()
            self._input_type = None
            self._input_class = None
        self._metadata[ActionMetadataKey.INPUT_KEY] = self._input_schema

        if ActionMetadataKey.RETURN in annotations:
            _, self._output_schema = json_schema_for(
                annotations[ActionMetadataKey.RETURN], kind=self._kind, name=self._name, label='output'
            )
        else:
            self._output_schema = TypeAdapter(object).json_schema()
        self._metadata[ActionMetadataKey.OUTPUT_KEY] = self._output_schema

    def _initialize_init_schema(
        self,
        init_schema: type[BaseModel] | dict[str, object] | None,
    ) -> None:
        """Register the schema for per-run ``init`` data, if the action declares one.

        Mirrors the input/output schema setup: a Pydantic model gives us a
        validator plus a published JSON schema; a raw dict is published as-is
        but can't be validated.
        """
        if init_schema is None:
            self._init_type: TypeAdapter[InitT] | None = None
            return
        self._init_schema: dict[str, object] = to_json_schema(init_schema)
        self._metadata[ActionMetadataKey.INIT_KEY] = self._init_schema
        if isinstance(init_schema, dict):
            self._init_type = None
        else:
            self._init_type = cast(TypeAdapter[InitT], TypeAdapter(init_schema))

    def _validate_init(self, init: InitT | None) -> InitT | None:
        """Validate per-run ``init`` against the init schema when one is registered.

        A missing ``init`` is validated as an empty object so a schema whose
        fields are all optional (like an agent's session identity) still produces
        a sensible default. A schema with required fields instead surfaces a clear
        "init required" error rather than a raw validation dump about ``{}``.
        """
        if self._init_type is None:
            return init
        try:
            return self._init_type.validate_python(init if init is not None else {})
        except ValidationError as e:
            if init is None:
                raise GenkitError(
                    message=(
                        f"Action '{self.name}' requires init but none was provided. Please supply a valid init payload."
                    ),
                    status='INVALID_ARGUMENT',
                    reason=RuntimeErrorReason.INVALID_INPUT,
                ) from e
            raise GenkitError(
                message=f"Invalid init for action '{self.name}': {e}",
                status='INVALID_ARGUMENT',
                cause=e,
                reason=RuntimeErrorReason.INVALID_INPUT,
            ) from e

    def _validate_input(self, input: InputT | None) -> InputT | None:
        """Validate caller input against the action schema when one is registered."""
        if self._input_type is None:
            return input
        # Skip validation when the caller passed nothing AND the wrapped
        # function declares a Python default for its input — that's the
        # signal that "no input" is a legitimate way to invoke this action.
        if input is None and self._params.input_optional:
            return input
        payload: object = input
        # A differently-typed ModelRequest with a mapping config is dumped and
        # re-parsed into the plugin class. A Pydantic config instance of the
        # wrong class is a caller mistake — dump would silently coerce it.
        if isinstance(input, BaseModel):
            try:
                return self._input_type.validate_python(input)
            except ValidationError:
                config = getattr(input, 'config', None)
                if isinstance(config, BaseModel):
                    expected = declared_config_type(self._input_class) if self._input_class is not None else None
                    want = config_type_path(expected) if isinstance(expected, type) else 'the plugin config class'
                    raise GenkitError(
                        message=(
                            f"Invalid input for action '{self.name}': "
                            f'config must be {want} or a mapping, '
                            f'got {config_type_path(type(config))}'
                        ),
                        status='INVALID_ARGUMENT',
                        reason=RuntimeErrorReason.INVALID_INPUT,
                    ) from None
                payload = input.model_dump(mode='python')

        try:
            return self._input_type.validate_python(payload)
        except ValidationError as e:
            msg = (
                f"Action '{self.name}' requires input but none was provided. Please supply a valid input payload."
                if input is None
                else f"Invalid input for action '{self.name}': {e}"
            )
            raise GenkitError(
                message=msg,
                status='INVALID_ARGUMENT',
                cause=e,
                reason=RuntimeErrorReason.INVALID_INPUT,
            ) from e

    async def _run_with_telemetry(
        self,
        input: object | None,
        ctx: ActionRunContext,
        on_trace_start: Callable[[str, str], Awaitable[None]] | None,
        telemetry_labels: dict[str, object] | None,
        *,
        execute: Callable[[], Awaitable[OutputT]] | None = None,
    ) -> ActionResponse[OutputT]:
        """Open the action span via ``run_in_new_span``, dispatch ``self._fn``, wrap errors in ``GenkitError``."""
        start_time = time.perf_counter()

        # ``telemetry_labels`` are caller-controlled passthrough attrs (e.g.
        # genkitx:ignore-trace, which the Developer UI filters on).
        # ``self._span_metadata`` uses short keys that land as genkit:metadata:<k>.
        extra_metadata: dict[str, str] = {k: str(v) for k, v in self._span_metadata.items()}
        # The Dev UI Context panel shows this dict. auth / secrets are what
        # the caller handed the action for the model or tools — write
        # placeholders so a shared trace dump does not leak them.
        if ctx.context:
            traced_context = context_for_telemetry(ctx.context)
            try:
                extra_metadata['context'] = json.dumps(traced_context)
            except Exception:
                try:
                    cleaned_context = _sanitize_value(traced_context)
                    extra_metadata['context'] = json.dumps(cleaned_context)
                except Exception:
                    extra_metadata['context'] = str(traced_context)

        trace_id = ''
        span_id = ''

        async def body(span: SpanContext) -> OutputT:
            nonlocal trace_id, span_id
            trace_id = span.trace_id
            span_id = span.span_id
            if on_trace_start:
                await on_trace_start(trace_id, span_id)

            try:
                if execute is not None:
                    output = await execute()
                else:
                    output = await self._invoke(input, ctx)
            except Interrupt as e:
                if e.metadata:
                    span.set_metadata({'interrupt': e.metadata})
                raise
            latency_ms = (time.perf_counter() - start_time) * 1000
            return cast(OutputT, _record_latency(output, latency_ms))

        attributes = {k: str(v) for k, v in (telemetry_labels or {}).items()}
        attributes.update({metadata_key(k): v for k, v in extra_metadata.items()})
        if ctx.init is not None:
            attributes[Attr.INIT] = to_json_attr(ctx.init)

        try:
            output = await run_in_new_span(
                self._name,
                body,
                action_type=str(self._kind),
                input=input,
                attributes=attributes,
                is_action=True,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ActionResponse(
                response=output,
                trace_id=trace_id,
                span_id=span_id,
                latency_ms=latency_ms,
            )
        except GenkitError:
            raise
        except Exception as e:
            # Wrap outside the span so we don't clobber ``genkit:error`` (which
            # the renderer already set to ``str(original_e)``).
            raise GenkitError(
                cause=e,
                message=f'Error while running action {self._name}',
                trace_id=trace_id,
            ) from e

    async def _invoke(self, input: object | None, ctx: ActionRunContext) -> OutputT:
        """Call ``self._fn`` with the input and context."""
        if self._fn_is_wrapper:
            # The wrapper takes (input, ctx) and forwards to the user's
            # function with self.params.call.
            output = await self._fn(input, ctx)
        else:
            output = await self._params.call(self._fn, input, ctx)
        stamp_background_operation(output=output, kind=self._kind, name=self._name)
        return output


async def single_item_stream(item: InputT) -> AsyncIterator[InputT]:
    """Present a one-shot input as a one-item stream (no item ⇒ no turns).

    Lets a multi-turn fn be driven by a plain ``run(input)``: the fn just sees a
    stream with exactly one turn (or zero when there's no input to send).
    """
    if item is not None:
        yield item


# =============================================================================
# BidiConnection
# =============================================================================

StreamInT = TypeVar('StreamInT')
StreamOutT_co = TypeVar('StreamOutT_co', covariant=True)
BidiOutT_co = TypeVar('BidiOutT_co', covariant=True)


class BidiConnection(Generic[StreamInT, StreamOutT_co, BidiOutT_co]):
    """Client-side handle for an active bidirectional streaming session.

    Returned by BidiAction.stream_bidi(). It's a thin ergonomic wrapper: send/
    close push per-turn inputs into the run's input stream, while receive/output
    read the run's chunk stream and final result. The run itself goes through the
    same stream()/run() path as any other action.
    """

    def __init__(
        self,
        in_queue: CloseableQueue[StreamInT],
        stream_response: StreamResponse[StreamOutT_co, BidiOutT_co],
    ) -> None:
        self._in_queue = in_queue
        self._stream_response = stream_response
        self.closed = False

    async def send(self, item: StreamInT) -> None:
        """Send a per-turn input to the server."""
        if self.closed:
            raise GenkitError(
                message=(
                    'Cannot send input on BidiConnection because the connection has '
                    'already been closed. No further inputs can be sent after close() '
                    'is called.'
                ),
                status='FAILED_PRECONDITION',
                reason=RuntimeErrorReason.CONNECTION_CLOSED,
            )
        await self._in_queue.put(item)

    async def close(self) -> None:
        """Signal no more inputs will be sent."""
        if not self.closed:
            self.closed = True
            self._in_queue.close()

    async def receive(self) -> AsyncIterator[StreamOutT_co]:
        """Async iterator yielding server-side stream chunks."""
        async for chunk in self._stream_response.stream:
            yield chunk

    async def output(self) -> BidiOutT_co:
        """Await the final output from the server fn."""
        return await self._stream_response.response


# =============================================================================
# BidiAction
# =============================================================================


class BidiAction(Action[InputT, OutputT, ChunkT, InitT]):
    """An Action extended with bidirectional streaming via stream_bidi().

    Both one-shot calls and live sessions run through the same Action.run() /
    stream() path: run() drives the fn with a single input, while stream_bidi()
    is sugar that hands run() a live input stream and returns a connection
    handle for sending per-turn inputs and receiving chunks.
    """

    def __init__(
        self,
        kind: ActionKind,
        name: str,
        bidi_fn: BidiFn[InitT, InputT, ChunkT, OutputT],
        metadata_fn: Callable[..., object] | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
        span_metadata: dict[str, SpanAttributeValue] | None = None,
        init_schema: type[BaseModel] | dict[str, object] | None = None,
        input_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None:
        self.bidi_fn = bidi_fn
        super().__init__(
            kind=kind,
            name=name,
            fn=self.action_fn,
            metadata_fn=metadata_fn,
            description=description,
            # The 'bidi': True metadata flag is used by the Genkit Dev UI and Reflection API
            # to identify this as a bidirectional action and render the interactive chat interface.
            metadata={**(metadata or {}), 'bidi': True},
            span_metadata=span_metadata,
            init_schema=init_schema,
        )
        # The wrapper's input arg is a generic TypeVar, so the derived input
        # schema is untyped. Declaring the per-turn input schema explicitly lets
        # run() coerce a raw payload (e.g. a JSON body) into the real input type
        # before it reaches the fn — the same way the input arrives typed over a
        # live connection.
        if input_schema is not None:
            self._override_input_schema(input_schema)

    async def action_fn(self, input: InputT, ctx: ActionRunContext) -> OutputT:  # noqa: A002
        """Adapt the bidi fn to the plain Action fn shape.

        The bidi fn reads its per-turn inputs from an async stream and emits
        chunks through a callback — the same pair a plain action fn gets on its
        ctx. A live chat supplies ``ctx.input_stream``; a one-shot run has just
        the single ``input``, which we hand over as a one-item stream. init
        (session identity) rides the run's init channel, separate from inputs.
        """
        if ctx.input_stream is not None:
            # ctx carries the stream as AsyncIterator[object]; here we know it's
            # this action's InputT. (ty collapses InputT to object and sees this
            # as redundant; pyright needs it.)
            input_stream = cast('AsyncIterator[InputT]', ctx.input_stream)  # ty: ignore[redundant-cast]
        else:
            input_stream = single_item_stream(input)
        return await self.bidi_fn(cast(InitT, ctx.init), input_stream, ctx.send_chunk)

    async def stream_bidi(
        self,
        init: InitT | None = None,
        context: dict[str, Any] | None = None,
        telemetry_labels: dict[str, object] | None = None,
    ) -> BidiConnection[InputT, ChunkT, OutputT]:
        """Start a bidirectional streaming session over the single stream() primitive.

        Opens an input channel, kicks off ``stream(input_stream=channel)``, and
        returns a BidiConnection whose send/close push per-turn inputs into that
        channel while receive/output read the run's chunks and final result.
        ``init`` is the session identity for the whole connection; per-turn
        inputs arrive later via ``BidiConnection.send``. It's sugar — all
        execution goes through the same run()/stream() path as any other action.
        """
        # Unbounded: turn-level backpressure is managed at the agent runtime intake.
        in_queue: CloseableQueue[InputT] = CloseableQueue()
        stream_response = self.stream(
            init=init,
            context=context,
            telemetry_labels=telemetry_labels,
            input_stream=in_queue,
        )
        return BidiConnection(in_queue, stream_response)


def get_current_context() -> dict[str, Any] | None:
    """Get the current action execution context, or None if not in an action.

    This module-level helper provides public cross-boundary access to
    the private _action_context ContextVar.
    """
    return _action_context.get(None)


def set_action_name(action: Action[Any, Any, Any], name: str) -> None:
    """Set the name of an action.

    Used internally for plugin namespace normalization to mutate the action's
    private name backing field without exposing a setter on the Action class.
    """
    action._name = name
