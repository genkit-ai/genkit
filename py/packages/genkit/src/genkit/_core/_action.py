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
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextvars import ContextVar
from typing import Any, ClassVar, Generic, NamedTuple, cast, get_type_hints

from opentelemetry.util import types as otel_types
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic.alias_generators import to_camel
from typing_extensions import Never, TypeVar

from genkit._core._channel import Channel, CloseableQueue
from genkit._core._compat import StrEnum
from genkit._core._error import GenkitError
from genkit._core._schema import to_json_schema
from genkit._core._trace._suppress import suppress_telemetry
from genkit._core._tracing import SpanMetadata, run_in_new_span

# =============================================================================
# Span attribute types and tracing helpers
# =============================================================================

# Type alias for span attribute values
SpanAttributeValue = otel_types.AttributeValue


def _record_latency(output: object, start_time: float) -> object:
    """Stamp ``latency_ms`` on the output if it has one (in place, or via ``model_copy`` for frozen models)."""
    latency_ms = (time.perf_counter() - start_time) * 1000
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


# =============================================================================
# Action types
# =============================================================================

# Type alias for action name.
ActionName = str


class ActionKind(StrEnum):
    """Types of actions that can be registered."""

    BACKGROUND_MODEL = 'background-model'
    AGENT = 'agent'
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


def _first_action_arg_has_default(input_spec: inspect.FullArgSpec, n_action_args: int) -> bool:
    """Return True if the action's first user-facing arg has a Python default.

    Lets `@ai.flow() async def f(name: str = 'world')` be called as `await f()`
    without forcing the caller to pass `None` explicitly. The default makes the
    input semantically optional from the function's perspective; we honour that
    when dispatching.
    """
    if n_action_args == 0:
        return False
    # FullArgSpec.defaults applies to the *trailing* positional args, so the
    # first positional has a default iff defaults covers every positional arg.
    defaults = input_spec.defaults or ()
    return len(defaults) >= n_action_args


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

    Provides read-only access to action context (auth, metadata), streaming
    support, and an abort signal for cooperative cancellation.
    """

    def __init__(
        self,
        context: dict[str, object] | None = None,
        streaming_callback: StreamingCallback | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> None:
        self._context: dict[str, object] = context if context is not None else {}
        self._streaming_callback = streaming_callback
        self.abort_signal: asyncio.Event = abort_signal if abort_signal is not None else asyncio.Event()

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
        self._span_metadata: dict[str, SpanAttributeValue] = span_metadata or {}
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
        # Raw user fn; tracing/dispatch handled by _run_with_telemetry / _invoke.
        self._fn: Callable[..., Awaitable[OutputT]] = fn
        self._n_action_args: int = len(action_args)
        self._action_arg_names: list[str] = action_args
        # When True, calling the action without an input is legal because the
        # wrapped function will fall back to its own Python-level default.
        self._first_arg_optional: bool = _first_action_arg_has_default(input_spec, len(action_args))
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
        context: dict[str, object] | None = None,
        on_trace_start: Callable[[str, str], Awaitable[None]] | None = None,
        telemetry_labels: dict[str, object] | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> ActionResponse[OutputT]:
        """Execute the action with optional input validation.

        Args:
            input: The input to the action. Will be validated against the input schema.
            on_chunk: Optional streaming callback for chunked responses.
            context: Optional context dict for the action.
            on_trace_start: Optional callback invoked when trace starts.
            telemetry_labels: Custom labels to set as direct span attributes.
            abort_signal: Optional shared abort event for cooperative cancellation.

        Returns:
            ActionResponse containing the result and trace metadata.

        Raises:
            GenkitError: If input validation fails (INVALID_ARGUMENT status).
        """
        input = self._validate_input(input)

        token = None
        if context:
            token = _action_context.set(context)

        streaming_cb = cast(StreamingCallback, on_chunk) if on_chunk else None

        try:
            return await self._run_with_telemetry(
                input,
                ActionRunContext(
                    context=_action_context.get(None),
                    streaming_callback=streaming_cb,
                    abort_signal=abort_signal,
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

    def _validate_input(self, input: InputT | None) -> InputT | None:
        """Validate caller input against the action schema when one is registered."""
        if self._input_type is None:
            return input
        # Skip validation when the caller passed nothing AND the wrapped
        # function declares a Python default for its first arg — that's the
        # signal that "no input" is a legitimate way to invoke this action.
        if input is None and self._first_arg_optional:
            return input
        try:
            return self._input_type.validate_python(input)
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
        suppress = str((telemetry_labels or {}).get('genkitx:ignore-trace', '')).lower() == 'true'
        suppress_token = suppress_telemetry.set(True) if suppress else None

        # ``type``/``subtype`` set canonical genkit:type / genkit:metadata:subtype attrs.
        # ``self._span_metadata`` uses short keys; run_in_new_span auto-prefixes them with
        # ``genkit:metadata:``. ``telemetry_labels`` are caller-controlled passthrough attrs.
        extra_metadata: dict[str, str] = {k: str(v) for k, v in self._span_metadata.items()}
        # Surface action context (auth, headers, etc.) on the span so the Dev UI
        # trace inspector can render the "Context" panel for a flow run.
        if ctx.context:
            try:
                extra_metadata['context'] = json.dumps(ctx.context)
            except Exception:
                try:
                    cleaned_context = _sanitize_value(ctx.context)
                    extra_metadata['context'] = json.dumps(cleaned_context)
                except Exception:
                    extra_metadata['context'] = str(ctx.context)
        span_meta = SpanMetadata(
            name=self._name,
            type='action',
            subtype=str(self._kind),
            input=input,
            metadata=extra_metadata or None,
            telemetry_labels={k: str(v) for k, v in (telemetry_labels or {}).items()} or None,
        )

        trace_id = ''
        try:
            with run_in_new_span(span_meta) as span:
                # OpenTelemetry standard hex format.
                trace_id = format(span.get_span_context().trace_id, '032x')
                span_id = format(span.get_span_context().span_id, '016x')
                if on_trace_start:
                    await on_trace_start(trace_id, span_id)

                if execute is not None:
                    output = await execute()
                else:
                    output = await self._invoke(input, ctx)
                output = cast(OutputT, _record_latency(output, start_time))
                # Picked up by run_in_new_span's success branch and written as ``genkit:output``.
                span_meta.output = output
                return ActionResponse(response=output, trace_id=trace_id, span_id=span_id)
        except GenkitError:
            raise
        except Exception as e:
            # Wrap outside the with-block so we don't clobber ``genkit:error`` (which
            # ``run_in_new_span`` already set to ``str(original_e)``).
            raise GenkitError(
                cause=e,
                message=f'Error while running action {self._name}',
                trace_id=trace_id,
            ) from e
        finally:
            if suppress_token is not None:
                suppress_telemetry.reset(suppress_token)

    async def _invoke(self, input: object | None, ctx: ActionRunContext) -> OutputT:
        """Dispatch ``self._fn`` based on its declared arity (0/1/2 args)."""
        # When the caller passed no input and the function's first arg has a
        # Python default, dispatch *without* the input so the default applies.
        # The 2-arg form passes ctx by keyword (using the user's actual
        # parameter name) so the defaulted first arg isn't accidentally
        # supplanted by a positional.
        omit_input = input is None and self._first_arg_optional
        match self._n_action_args:
            case 0:
                return await self._fn()
            case 1:
                if omit_input:
                    return await self._fn()
                return await self._fn(input)
            case 2:
                if omit_input:
                    ctx_param_name = self._action_arg_names[1]
                    return await self._fn(**{ctx_param_name: ctx})
                return await self._fn(input, ctx)
            case _:
                raise ValueError('action fn must have 0-2 args')


# =============================================================================
# BidiConnection
# =============================================================================


class QueueSentinel:
    """Singleton marker for end-of-stream on bidi asyncio queues."""

    __slots__ = ()

    def __repr__(self) -> str:
        return '<queue-sentinel>'


QUEUE_SENTINEL = QueueSentinel()
_SENTINEL = QUEUE_SENTINEL  # module-local alias; tests may import as _SENTINEL

StreamInT = TypeVar('StreamInT', default=Any)
StreamOutT_co = TypeVar('StreamOutT_co', covariant=True, default=Any)
BidiOutT_co = TypeVar('BidiOutT_co', covariant=True, default=Any)


class BidiConnection(Generic[StreamInT, StreamOutT_co, BidiOutT_co]):
    """Client-side handle for an active bidirectional streaming session.

    Returned by BidiAction.stream_bidi(). Wraps two asyncio.Queues and a
    result Future that resolves when the server fn completes.
    """

    def __init__(
        self,
        in_queue: CloseableQueue[Any],
        out_queue: CloseableQueue[Any],
        result: asyncio.Future[BidiOutT_co],
    ) -> None:
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._result = result
        self._closed = False
        self.trace_id: str | None = None

    async def send(self, input: StreamInT) -> None:  # noqa: A002
        """Send a per-turn input to the server.

        Raises GenkitError if the connection is already closed.
        """
        if self._closed:
            raise GenkitError(
                message='BidiConnection: send on closed connection',
                status='FAILED_PRECONDITION',
            )
        await self._in_queue.put(input)

    async def close(self) -> None:
        """Signal no more inputs will be sent.

        Idempotent — safe to call more than once.
        """
        if not self._closed:
            self._closed = True
            if hasattr(self._in_queue, 'close'):
                self._in_queue.close()

    async def receive(self) -> AsyncIterator[StreamOutT_co]:
        """Async iterator yielding server-side stream chunks.

        Terminates when the server fn finishes.
        """
        async for chunk in self._out_queue:
            yield chunk  # type: ignore[misc]

    async def output(self) -> BidiOutT_co:
        """Await the final output from the server fn."""
        return await self._result


# =============================================================================
# BidiAction
# =============================================================================


class BidiAction(Action[InputT, OutputT, ChunkT]):
    """An Action extended with bidirectional streaming via stream_bidi().

    The underlying fn still runs through Action.run() for one-shot calls.
    stream_bidi() wires up two asyncio.Queues and launches the bidi fn as
    a background task.
    """

    def __init__(
        self,
        kind: ActionKind,
        name: str,
        bidi_fn: Callable[..., Awaitable[OutputT]],
        metadata_fn: Callable[..., object] | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
        span_metadata: dict[str, SpanAttributeValue] | None = None,
    ) -> None:
        # Wrap bidi_fn as a standard Action fn (closes in_queue immediately,
        # forwards out_queue chunks to on_chunk callback) so Action.run() works.
        async def _as_streaming_fn(input: InputT, ctx: ActionRunContext) -> OutputT:  # noqa: A002
            in_queue = CloseableQueue(maxsize=1)
            out_queue = CloseableQueue(maxsize=1)
            # Close input immediately — no streaming inputs via one-shot path.
            in_queue.close()

            result_holder: list[Any] = []
            err_holder: list[BaseException] = []
            done = asyncio.Event()

            async def _run() -> None:
                try:
                    result_holder.append(await bidi_fn(input, in_queue, out_queue))
                except Exception as e:  # noqa: BLE001
                    err_holder.append(e)
                finally:
                    out_queue.close()
                    done.set()

            run_task = asyncio.create_task(_run())
            try:
                # Forward chunks to Action's streaming callback.
                async for chunk in out_queue:
                    ctx.send_chunk(chunk)
                await done.wait()
            finally:
                if not run_task.done():
                    run_task.cancel()
                    try:
                        await run_task
                    except asyncio.CancelledError:
                        pass
            if err_holder:
                raise err_holder[0]
            return result_holder[0] if result_holder else cast(OutputT, None)  # type: ignore[return-value]

        super().__init__(
            kind=kind,
            name=name,
            fn=_as_streaming_fn,
            metadata_fn=metadata_fn,
            description=description,
            metadata={**(metadata or {}), 'bidi': True},
            span_metadata=span_metadata,
        )
        self._bidi_fn = bidi_fn

    async def stream_bidi(
        self,
        input: InputT,  # noqa: A002
        context: dict[str, object] | None = None,
        on_trace_start: Callable[[str, str], Awaitable[None]] | None = None,
        telemetry_labels: dict[str, object] | None = None,
    ) -> BidiConnection[Any, ChunkT, OutputT]:
        """Start a bidirectional streaming session.

        Launches the bidi fn as a background asyncio task and returns a
        BidiConnection the caller uses to send inputs and receive chunks.
        """
        validated = self._validate_input(input)
        if validated is None:
            raise GenkitError(
                message=f"Action '{self.name}' requires input but none was provided.",
                status='INVALID_ARGUMENT',
            )
        input = validated

        # in_queue is size 1 (backpressure: caller blocks between sends).
        # out_queue is unbounded: agent fn emits multiple chunks per turn
        # and the caller may not be consuming yet — put_nowait must not block.
        in_queue = CloseableQueue(maxsize=1)
        out_queue = CloseableQueue()
        result_future: asyncio.Future[OutputT] = asyncio.get_event_loop().create_future()
        conn = BidiConnection(in_queue, out_queue, result_future)

        token = None
        if context:
            token = _action_context.set(context)

        async def _on_trace_start(trace_id: str, span_id: str) -> None:
            conn.trace_id = trace_id
            if on_trace_start is not None:
                await on_trace_start(trace_id, span_id)

        async def _run() -> None:
            try:

                async def _execute_bidi() -> OutputT:
                    return await self._bidi_fn(input, in_queue, out_queue)

                response = await self._run_with_telemetry(
                    input,
                    ActionRunContext(context=_action_context.get(None)),
                    _on_trace_start,
                    telemetry_labels,
                    execute=_execute_bidi,
                )
                result_future.set_result(response.response)
            except asyncio.CancelledError as e:
                # CancelledError must propagate out of the background task so that
                # the asyncio loop can correctly mark it as CANCELLED.
                result_future.set_exception(e)
                raise
            except Exception as e:  # noqa: BLE001
                # Standard exceptions are forwarded to the client's result_future.
                # We swallow them here to prevent asyncio from duplicate-logging
                # "exception was never retrieved" warnings on the background task.
                result_future.set_exception(e)
            finally:
                # Close out_queue to signal the end of the streaming iterator.
                if hasattr(out_queue, 'close'):
                    out_queue.close()

        try:
            asyncio.create_task(_run())
            return conn
        finally:
            if token is not None:
                _action_context.reset(token)


def define_bidi_action(
    registry: Any,  # noqa: ANN401
    kind: ActionKind,
    name: str,
    bidi_fn: Callable[..., Awaitable[Any]],
    metadata_fn: Callable[..., object] | None = None,
    description: str | None = None,
    metadata: dict[str, object] | None = None,
) -> BidiAction:
    """Create and register a BidiAction."""
    action = BidiAction(
        kind=kind,
        name=name,
        bidi_fn=bidi_fn,
        metadata_fn=metadata_fn,
        description=description,
        metadata=metadata,
    )
    registry.register_action_from_instance(action)
    return action


def get_current_context() -> dict[str, object] | None:
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
