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
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, Generic, NamedTuple, cast, get_type_hints

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span
from opentelemetry.util import types as otel_types
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic.alias_generators import to_camel
from typing_extensions import Never, TypeVar

from genkit._core._channel import Channel
from genkit._core._compat import StrEnum
from genkit._core._error import GenkitError
from genkit._core._trace._path import build_path
from genkit._core._trace._suppress import suppress_telemetry
from genkit._core._tracing import tracer

# =============================================================================
# Span attribute types and tracing helpers
# =============================================================================

# Type alias for span attribute values
SpanAttributeValue = otel_types.AttributeValue

# Context variable to track parent path across nested spans
_parent_path_context: ContextVar[str] = ContextVar('genkit_parent_path', default='')


@contextmanager
def _save_parent_path() -> Generator[None, None, None]:
    """Context manager to save and restore parent path."""
    saved = _parent_path_context.get()
    try:
        yield
    finally:
        _parent_path_context.set(saved)


def _record_input_metadata(
    span: Span,
    kind: str,
    name: str,
    span_metadata: dict[str, SpanAttributeValue] | None,
    input: object | None,
) -> None:
    """Records input metadata onto an OpenTelemetry span for a Genkit action."""
    span.set_attribute('genkit:type', 'action')
    span.set_attribute('genkit:metadata:subtype', kind)
    span.set_attribute('genkit:name', name)
    if input is not None:
        input_json = input.model_dump_json() if isinstance(input, BaseModel) else json.dumps(input)
        span.set_attribute('genkit:input', input_json)

    # Build and set path attributes (qualified path with full annotations)
    parent_path = _parent_path_context.get()
    qualified_path = build_path(name, parent_path, 'action', kind)

    span.set_attribute('genkit:path', qualified_path)
    span.set_attribute('genkit:qualifiedPath', qualified_path)

    # Update context for nested spans
    _parent_path_context.set(qualified_path)

    if span_metadata is not None:
        for meta_key in span_metadata:
            span.set_attribute(meta_key, span_metadata[meta_key])


def _record_output_metadata(span: Span, output: object) -> None:
    """Records output metadata onto an OpenTelemetry span for a Genkit action."""
    span.set_attribute('genkit:state', 'success')
    try:
        output_json = output.model_dump_json() if isinstance(output, BaseModel) else json.dumps(output)
        span.set_attribute('genkit:output', output_json)
    except Exception:
        # Fallback for non-serializable output
        span.set_attribute('genkit:output', str(output))


# =============================================================================
# Action types
# =============================================================================

# Type alias for action name.
ActionName = str


class ActionKind(StrEnum):
    """Types of actions that can be registered."""

    BACKGROUND_MODEL = 'background-model'
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
    RESOURCE = 'resource'
    RETRIEVER = 'retriever'
    TOOL = 'tool'
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


def extract_action_args_and_types(
    input_spec: inspect.FullArgSpec,
    annotations: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[Any]]:
    """Extract argument names and types from a function spec."""
    arg_types = []
    action_args = input_spec.args.copy()
    resolved_annotations = annotations or input_spec.annotations

    # Special case when using a method as an action, we ignore first "self"
    # arg. (Note: The original condition `len(action_args) <= 3` is preserved
    # from the source snippet).
    if len(action_args) > 0 and len(action_args) <= 3 and action_args[0] == 'self':
        del action_args[0]

    for arg in action_args:
        arg_types.append(resolved_annotations.get(arg, Any))

    return action_args, arg_types


# =============================================================================
# Action key utilities
# =============================================================================


# Attribute name used to attach a ``DynamicActionProvider`` (cache + helpers)
# onto the placeholder ``Action`` registered for a DAP. The registry only
# stores the ``Action``; the provider rides along on it as a Python attribute.
# Code holding the ``Action`` recovers the provider via
# ``getattr(action, GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR, None)``.
GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR = '_genkit_dynamic_action_provider'


def parse_dap_qualified_name(name: str) -> tuple[str, str, str] | None:
    """Parse DAP-qualified segment ``provider:innerKind/innerName``.

    Used when the action key kind is ``dynamic-action-provider`` and the name
    references a nested action exposed by a provider (e.g. MCP tools).

    Pattern: ``[provider]:[inner_kind]/[inner_name]`` — no slashes in the
    provider segment (``plugin/foo`` is not a valid provider host).

    Returns:
        ``(provider_name, inner_kind, inner_name)`` if the string matches the
        pattern; otherwise ``None`` so callers can treat the name as a plain
        dynamic-action-provider id.
    """
    # Pattern: [provider]:[inner_kind]/[inner_name]; no '/' or ':' in provider.
    match = re.match(r'^([^/:]+):([^/:]+)/(.+)$', name)
    if not match:
        return None
    provider, inner_kind, inner_name = match.groups()
    if not provider or not inner_kind or not inner_name:
        return None
    return (provider, inner_kind, inner_name)


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


def parse_dap_provider_host(name: str) -> str | None:
    """Return the segment before the first ``:`` when the name has multiple ``:``-split parts.

    If there is no ``:``, or the first segment is empty, returns ``None``.
    """
    parts = name.split(':')
    if len(parts) < 2:
        return None
    host = parts[0]
    if not host:
        return None
    return host


# =============================================================================
# Action core
# =============================================================================

InputT = TypeVar('InputT', default=Any)
OutputT = TypeVar('OutputT', default=Any)
ChunkT = TypeVar('ChunkT', default=Never)

# Generic streaming callback - use Callable[[ChunkT], None] for typed chunks
# This untyped version is for internal use where chunk type is unknown
StreamingCallback = Callable[[object], None]

_action_context: ContextVar[dict[str, object] | None] = ContextVar('context')
_ = _action_context.set(None)


class ActionRunContext:
    """Execution context for an action.

    Provides read-only access to action context (auth, metadata) and streaming support.
    """

    def __init__(
        self,
        context: dict[str, object] | None = None,
        streaming_callback: StreamingCallback | None = None,
    ) -> None:
        self._context: dict[str, object] = context if context is not None else {}
        self._streaming_callback = streaming_callback

    @property
    def context(self) -> dict[str, object]:
        return self._context

    @property
    def is_streaming(self) -> bool:
        """Returns True if a streaming callback is registered."""
        return self._streaming_callback is not None

    @property
    def streaming_callback(self) -> StreamingCallback | None:
        """Returns the streaming callback, if any.

        Use this when you need to pass the callback to another action.
        For sending chunks directly, use send_chunk() instead.
        """
        return self._streaming_callback

    def send_chunk(self, chunk: object) -> None:
        """Send a streaming chunk to the client.

        Args:
            chunk: The chunk data to stream.
        """
        if self._streaming_callback is not None:
            self._streaming_callback(chunk)

    @staticmethod
    def _current_context() -> dict[str, object] | None:
        return _action_context.get(None)


class Action(Generic[InputT, OutputT, ChunkT]):
    """A named, traced, remotely callable function."""

    _input_type: TypeAdapter[InputT] | None

    def __init__(
        self,
        kind: ActionKind,
        name: str,
        fn: Callable[..., Awaitable[OutputT]],
        metadata_fn: Callable[..., object] | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
        span_metadata: dict[str, SpanAttributeValue] | None = None,
    ) -> None:
        self._kind: ActionKind = kind
        self._name: str = name
        self._metadata: dict[str, object] = metadata if metadata else {}
        self._description: str | None = description
        # Optional matcher function for resource actions
        self.matches: Callable[[object], bool] | None = None

        # All action handlers must be async
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"Action handlers must be async functions. Got sync function for '{name}'.")

        input_spec = inspect.getfullargspec(metadata_fn if metadata_fn else fn)
        try:
            resolved_annotations = get_type_hints(metadata_fn if metadata_fn else fn)
        except (NameError, TypeError, AttributeError):
            resolved_annotations = input_spec.annotations
        action_args, arg_types = extract_action_args_and_types(input_spec, resolved_annotations)
        n_action_args = len(action_args)
        self._fn = _make_tracing_wrapper(name, kind, span_metadata or {}, n_action_args, fn)
        self._initialize_io_schemas(action_args, arg_types, resolved_annotations, input_spec)

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

    def override_input_schema(self, schema: type[BaseModel] | dict[str, object]) -> None:
        """Replace inferred input JSON Schema and validation type (e.g. tool schema overrides)."""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            type_adapter: TypeAdapter[Any] = TypeAdapter(schema)
            self._input_schema = type_adapter.json_schema()
            self._input_type = cast(TypeAdapter[InputT], type_adapter)
        elif isinstance(schema, dict):
            self._input_schema = schema
            self._input_type = None
        else:
            raise TypeError(f'input_schema must be a Pydantic model type or dict, got {type(schema)}')
        self._metadata[ActionMetadataKey.INPUT_KEY] = self._input_schema

    async def __call__(self, input: InputT | None = None) -> OutputT:
        """Call the action directly, returning just the response value."""
        return (await self.run(input)).response

    async def run(
        self,
        input: InputT | None = None,
        on_chunk: Callable[[ChunkT], None] | None = None,
        context: dict[str, object] | None = None,
        on_trace_start: Callable[[str, str], None] | None = None,
        telemetry_labels: dict[str, object] | None = None,
    ) -> ActionResponse[OutputT]:
        """Execute the action with optional input validation.

        Args:
            input: The input to the action. Will be validated against the input schema.
            on_chunk: Optional streaming callback for chunked responses.
            context: Optional context dict for the action.
            on_trace_start: Optional callback invoked when trace starts.
            telemetry_labels: Custom labels to set as direct span attributes.

        Returns:
            ActionResponse containing the result and trace metadata.

        Raises:
            GenkitError: If input validation fails (INVALID_ARGUMENT status).
        """
        # Validate input if we have a schema
        if self._input_type is not None:
            try:
                input = self._input_type.validate_python(input)
            except ValidationError as e:
                if input is None:
                    raise GenkitError(
                        message=(
                            f"Action '{self.name}' requires input but none was provided. "
                            'Please supply a valid input payload.'
                        ),
                        status='INVALID_ARGUMENT',
                    ) from e
                raise GenkitError(
                    message=f"Invalid input for action '{self.name}': {e}",
                    status='INVALID_ARGUMENT',
                    cause=e,
                ) from e

        if context:
            _ = _action_context.set(context)

        streaming_cb = cast(StreamingCallback, on_chunk) if on_chunk else None

        return await self._fn(
            input,
            ActionRunContext(
                context=_action_context.get(None),
                streaming_callback=streaming_cb,
            ),
            streaming_cb,
            on_trace_start,
            telemetry_labels,
        )

    def stream(
        self,
        input: InputT | None = None,
        context: dict[str, object] | None = None,
        telemetry_labels: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> StreamResponse[ChunkT, OutputT]:
        """Execute and return a StreamResponse with .stream and .response properties."""
        channel: Channel[ChunkT, ActionResponse[OutputT]] = Channel(timeout=timeout)

        def send_chunk(c: ChunkT) -> None:
            channel.send(c)

        resp = self.run(
            input=input,
            context=context,
            telemetry_labels=telemetry_labels,
            on_chunk=send_chunk,
        )
        channel.set_close_future(asyncio.create_task(resp))

        result_future: asyncio.Future[OutputT] = asyncio.Future()
        channel.closed.add_done_callback(lambda _: result_future.set_result(channel.closed.result().response))

        return StreamResponse(stream=channel, response=result_future)

    def _initialize_io_schemas(
        self,
        action_args: list[str],
        arg_types: list[type],
        annotations: dict[str, Any],
        _input_spec: inspect.FullArgSpec,
    ) -> None:
        # Allow up to 2 args: (input, ctx) - use ctx.send_chunk() for streaming
        if len(action_args) > 2:
            raise TypeError(f'can only have up to 2 args: {action_args}')

        if len(action_args) > 0:
            type_adapter = TypeAdapter(arg_types[0])
            self._input_schema: dict[str, object] = type_adapter.json_schema()
            self._input_type: TypeAdapter[InputT] | None = cast(TypeAdapter[InputT], type_adapter)
            self._metadata[ActionMetadataKey.INPUT_KEY] = self._input_schema
        else:
            self._input_schema = TypeAdapter(object).json_schema()
            self._input_type = None
            self._metadata[ActionMetadataKey.INPUT_KEY] = self._input_schema

        if ActionMetadataKey.RETURN in annotations:
            type_adapter = TypeAdapter(annotations[ActionMetadataKey.RETURN])
            self._output_schema: dict[str, object] = type_adapter.json_schema()
            self._metadata[ActionMetadataKey.OUTPUT_KEY] = self._output_schema
        else:
            self._output_schema = TypeAdapter(object).json_schema()
            self._metadata[ActionMetadataKey.OUTPUT_KEY] = self._output_schema


def _make_tracing_wrapper(
    name: str,
    kind: ActionKind,
    span_metadata: dict[str, SpanAttributeValue],
    n_action_args: int,
    fn: Callable[..., Awaitable[Any]],
) -> Callable[
    [
        object | None,
        ActionRunContext,
        StreamingCallback | None,
        Callable[[str, str], None] | None,
        dict[str, object] | None,
    ],
    Awaitable[ActionResponse[Any]],
]:
    """Create a tracing wrapper for an async action function."""

    def _record_latency(output: object, start_time: float) -> object:
        latency_ms = (time.perf_counter() - start_time) * 1000
        if hasattr(output, 'latency_ms'):
            try:
                cast(Any, output).latency_ms = latency_ms
            except (TypeError, ValidationError, AttributeError):
                # If immutable (e.g. Pydantic model with frozen=True), try model_copy
                if hasattr(output, 'model_copy'):
                    output = cast(Any, output).model_copy(update={'latency_ms': latency_ms})
        return output

    async def tracing_wrapper(
        input: object | None,
        ctx: ActionRunContext,
        on_chunk: StreamingCallback | None,
        on_trace_start: Callable[[str, str], None] | None,
        telemetry_labels: dict[str, object] | None,
    ) -> ActionResponse[Any]:
        start_time = time.perf_counter()

        suppress = str((telemetry_labels or {}).get('genkitx:ignore-trace', '')).lower() == 'true'
        suppress_token = suppress_telemetry.set(True) if suppress else None
        try:
            with _save_parent_path():
                with tracer.start_as_current_span(name) as span:
                    # Format trace_id and span_id as hex strings (OpenTelemetry standard format)
                    trace_id = format(span.get_span_context().trace_id, '032x')
                    span_id = format(span.get_span_context().span_id, '016x')
                    if on_trace_start:
                        on_trace_start(trace_id, span_id)

                    # Set telemetry labels as direct span attributes (matches JS/Go behavior)
                    if telemetry_labels:
                        for key, value in telemetry_labels.items():
                            span.set_attribute(key, str(value))

                    _record_input_metadata(
                        span=span,
                        kind=kind,
                        name=name,
                        span_metadata=span_metadata,
                        input=input,
                    )

                    try:
                        match n_action_args:
                            case 0:
                                output = await fn()
                            case 1:
                                output = await fn(input)
                            case 2:
                                output = await fn(input, ctx)
                            case _:
                                raise ValueError('action fn must have 0-2 args')
                    except Exception as e:
                        span.set_attribute('genkit:state', 'error')
                        # Bundled Dev UI reads timeEvents.exception.attributes only; stash text for export synthesis.
                        span.set_status(trace_api.StatusCode.ERROR, description=str(e))
                        span.record_exception(e)
                        if isinstance(e, GenkitError):
                            span.set_attribute('genkit:error', e.original_message)
                            raise
                        span.set_attribute('genkit:error', str(e))
                        raise GenkitError(
                            cause=e,
                            message=f'Error while running action {name}',
                            trace_id=trace_id,
                        ) from e

                    output = _record_latency(output, start_time)
                    _record_output_metadata(span, output=output)
                    return ActionResponse(response=output, trace_id=trace_id, span_id=span_id)
        finally:
            if suppress_token is not None:
                suppress_telemetry.reset(suppress_token)

    return tracing_wrapper
