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

"""Action module for defining and managing remotely callable functions.

This module provides the core `Action` class, the fundamental building block for
defining operations within the Genkit framework.

## What is an Action?

An Action represents a named, observable, and strongly-typed unit of work that
wraps a standard Python function (either synchronous or asynchronous). It serves
as a consistent interface for executing logic like calling models, running tools,
or orchestrating flows.

## How it Works:

1.  Initialization

    *   An `Action` is created with a unique `name`, a `kind` (e.g., MODEL,
        TOOL, FLOW), and the Python function (`fn`) containing the core logic.

    *   It automatically inspects the function's type hints (specifically the
        first argument for input and the return annotation for output) using
        Pydantic's `TypeAdapter` to generate JSON schemas (`input_schema`,
        `output_schema`). These are stored for validation and metadata. It
        raises a `TypeError` if the function signature has more than two
        arguments (input, context).

    *   It internally creates tracing wrappers (`_fn`, `_afn`) around the
        original function using the `_make_tracing_wrappers` helper. These
        wrappers handle OpenTelemetry span creation, recording input/output
        metadata, and standardizing error handling by raising `GenkitError`
        with a trace ID. The `_afn` wrapper ensures even synchronous functions
        can be awaited.

2.  Execution Methods

    *   `run()`: Executes the action synchronously. It calls the internal
        synchronous tracing wrapper (`_fn`).

    *   `arun()`: Executes the action asynchronously. It calls the internal
        asynchronous tracing wrapper (`_afn`). This wrapper handles
        awaiting the original function if it was async or running it via
        `ensure_async` if it was sync.

    *   `arun_raw()`: Similar to `arun`, but performs Pydantic validation on
        the `raw_input` before calling `arun`.

    *   `stream()`: Initiates an asynchronous execution via `arun` and returns
        an `AsyncIterator` (via `Channel`) for receiving chunks and an
        `asyncio.Future` that resolves with the final `ActionResponse`.

3.  Streaming and Context

    *   During execution (`run`/`arun`/`stream`), an `ActionRunContext` instance
        is created.

    *   This context holds an `on_chunk` callback (provided by the caller, e.g.,
        by `stream()`) and any user-provided `context` dictionary.

    *   If the wrapped function (`fn`) accepts a context argument (`ctx`), this
        `ActionRunContext` instance is passed, allowing the function to send
        intermediate chunks using `ctx.send_chunk()`.

    *   A `ContextVar` (`_action_context`) is also used to propagate the user
        context dictionary implicitly.

The `Action` class provides a robust way to define executable units, abstracting
away the complexities of sync/async handling (for async callers), schema
generation, tracing, and streaming mechanics.
"""

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar
from functools import cached_property
from typing import Any, Generic, Protocol, cast, get_type_hints

from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import Never, TypeVar

from genkit.aio import Channel, ensure_async
from genkit.core.error import GenkitError
from genkit.core.tracing import tracer

from ._tracing import (
    SpanAttributeValue,
    record_input_metadata,
    record_output_metadata,
    save_parent_path,
)
from ._util import extract_action_args_and_types, noop_streaming_callback
from .types import ActionKind, ActionMetadataKey, ActionResponse

InputT = TypeVar('InputT', default=Any)
OutputT = TypeVar('OutputT', default=Any)
ChunkT = TypeVar('ChunkT', default=Never)

StreamingCallback = Callable[[object], None]

_action_context: ContextVar[dict[str, object] | None] = ContextVar('context')
_ = _action_context.set(None)


class _LatencyTrackable(Protocol):
    """Protocol for objects that support latency tracking."""

    latency_ms: float


class _ModelCopyable(Protocol):
    """Protocol for objects that support model_copy."""

    def model_copy(self, *, update: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
        """Copy the model with optional updates."""
        ...


class ActionRunContext:
    """Provides context for an action's execution, including streaming support.

    This class holds context information relevant to a single execution of an
    Action. It manages the streaming callback (`on_chunk`) and any additional
    context dictionary provided during the action call.

    Attributes:
        context: A dictionary containing arbitrary context data.
        is_streaming: Whether the action is being executed in streaming mode.
    """

    def __init__(
        self,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, object] | None = None,
        on_trace_start: Callable[[str, str], None] | None = None,
    ) -> None:
        """Initializes an ActionRunContext instance.

        Sets up the context with an optional streaming callback and an optional
        context dictionary. If `on_chunk` is None, a no-op callback is used.

        Args:
            on_chunk: A callable to be invoked when a chunk of data is ready
                      during streaming execution. Defaults to a no-op function.
            context: An optional dictionary containing context data to be made
                     available within the action execution. Defaults to an empty
                     dictionary.
            on_trace_start: A callable to be invoked with the trace ID and span
                            ID when the trace is started.
        """
        self._on_chunk: StreamingCallback = on_chunk if on_chunk is not None else noop_streaming_callback
        self._context: dict[str, object] = context if context is not None else {}
        self._on_trace_start: Callable[[str, str], None] = on_trace_start if on_trace_start else lambda _t, _s: None

    @property
    def context(self) -> dict[str, object]:
        return self._context

    @cached_property
    def is_streaming(self) -> bool:
        """Indicates whether the action is being executed in streaming mode.

        This property checks if a valid streaming callback (`on_chunk`)
        was provided during initialization.

        Returns:
            True if a streaming callback (other than the no-op default) is set,
            False otherwise.
        """
        return self._on_chunk != noop_streaming_callback

    def send_chunk(self, chunk: object) -> None:
        """Send a chunk to from the action to the client.

        Args:
            chunk: The chunk to send to the client.
        """
        self._on_chunk(chunk)

    @staticmethod
    def _current_context() -> dict[str, object] | None:
        """Obtains current context if running within an action.

        Returns:
            The current context if running within an action, None otherwise.
        """
        return _action_context.get(None)


class Action(Generic[InputT, OutputT, ChunkT]):
    """Represents a strongly-typed, remotely callable function within Genkit.

    Actions are the fundamental building blocks for defining operations in Genkit.
    They are named, observable (via tracing), and support both streaming and
    non-streaming execution modes. An Action wraps a Python function, handling
    input validation, execution, tracing, and output serialization.

    Attributes:
        name: A unique identifier for the action.
        kind: The type category of the action (e.g., MODEL, TOOL, FLOW).
        description: An optional human-readable description.
        input_schema: The JSON schema definition for the expected input type.
        output_schema: The JSON schema definition for the expected output type.
        metadata: A dictionary for storing arbitrary metadata associated with the action.
        is_async: Whether the action is asynchronous.
    """

    def __init__(
        self,
        kind: ActionKind,
        name: str,
        fn: Callable[..., OutputT | Awaitable[OutputT]],
        metadata_fn: Callable[..., object] | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
        span_metadata: dict[str, SpanAttributeValue] | None = None,
    ) -> None:
        """Initialize an Action.

        Args:
            kind: The kind of action (e.g., TOOL, MODEL, etc.).
            name: Unique name identifier for this action.
            fn: The function to call when the action is executed.
            metadata_fn: The function to be used to infer metadata (e.g.
                schemas).
            description: Optional human-readable description of the action.
            metadata: Optional dictionary of metadata about the action.
            span_metadata: Optional dictionary of tracing span metadata.
        """
        self._kind: ActionKind = kind
        self._name: str = name
        self._metadata: dict[str, object] = metadata if metadata else {}
        self._description: str | None = description
        self._is_async: bool = inspect.iscoroutinefunction(fn)
        # Optional matcher function for resource actions
        self.matches: Callable[[object], bool] | None = None

        input_spec = inspect.getfullargspec(metadata_fn if metadata_fn else fn)
        try:
            resolved_annotations = get_type_hints(metadata_fn if metadata_fn else fn)
        except (NameError, TypeError, AttributeError):
            resolved_annotations = input_spec.annotations
        action_args, arg_types = extract_action_args_and_types(input_spec, resolved_annotations)
        n_action_args = len(action_args)
        fn_pair = _make_tracing_wrappers(name, kind, span_metadata or {}, n_action_args, fn)
        self._fn: Callable[..., ActionResponse[OutputT]] = fn_pair[0]
        self._afn: Callable[..., Awaitable[ActionResponse[OutputT]]] = fn_pair[1]
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
        """Update input schema (used by lazy-loaded prompts to set schema after registration)."""
        self._input_schema = value
        self._metadata[ActionMetadataKey.INPUT_KEY] = value

    @property
    def output_schema(self) -> dict[str, object]:
        return self._output_schema

    @output_schema.setter
    def output_schema(self, value: dict[str, object]) -> None:
        """Update output schema (used by lazy-loaded prompts to set schema after registration)."""
        self._output_schema = value
        self._metadata[ActionMetadataKey.OUTPUT_KEY] = value

    @property
    def is_async(self) -> bool:
        return self._is_async

    def run(
        self,
        input: InputT | None = None,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, object] | None = None,
        _telemetry_labels: dict[str, object] | None = None,
    ) -> ActionResponse[OutputT]:
        """Executes the action synchronously with the given input.

        This method runs the action's underlying function synchronously.
        It handles input validation, tracing, and output serialization.
        If the action's function is async, it will be run in the current event loop.

        Args:
            input: The input data for the action. It should conform to the action's
                   input schema.
            on_chunk: An optional callback function to receive streaming output chunks.
                      Note: For synchronous execution of streaming actions, chunks
                      will be delivered synchronously via this callback.
            context: An optional dictionary containing context data for the execution.
            telemetry_labels: Optional labels for telemetry.

        Returns:
            An ActionResponse object containing the final result and trace ID.

        Raises:
            GenkitError: If an error occurs during action execution.
        """
        # TODO(#4348): handle telemetry_labels

        if context:
            _ = _action_context.set(context)

        return self._fn(
            input,
            ActionRunContext(on_chunk=on_chunk, context=_action_context.get(None)),
        )

    async def arun(
        self,
        input: InputT | None = None,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, object] | None = None,
        on_trace_start: Callable[[str, str], None] | None = None,
        _telemetry_labels: dict[str, object] | None = None,
    ) -> ActionResponse[OutputT]:
        """Executes the action asynchronously with the given input.

        This method runs the action's underlying function asynchronously.
        It handles input validation, tracing, and output serialization.
        If the action's function is synchronous, it will be wrapped to run
        asynchronously.

        Args:
            input: The input data for the action. It should conform to the action's
                   input schema.
            on_chunk: An optional callback function to receive streaming output chunks.
            context: An optional dictionary containing context data for the execution.
            on_trace_start: An optional callback to be invoked with the trace ID
                            and span ID when the trace is started.
            telemetry_labels: Optional labels for telemetry.

        Returns:
            An awaitable ActionResponse object containing the final result and trace ID.

        Raises:
            GenkitError: If an error occurs during action execution.
        """
        # TODO(#4348): handle telemetry_labels

        if context:
            _ = _action_context.set(context)

        return await self._afn(
            input,
            ActionRunContext(on_chunk=on_chunk, context=_action_context.get(None), on_trace_start=on_trace_start),
        )

    async def arun_raw(
        self,
        raw_input: InputT | None = None,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, object] | None = None,
        on_trace_start: Callable[[str, str], None] | None = None,
        telemetry_labels: dict[str, object] | None = None,
    ) -> ActionResponse[OutputT]:
        """Executes the action asynchronously with raw, unvalidated input.

        This method bypasses the Pydantic input validation and calls the underlying
        action function directly with the provided `raw_input`. It still handles
        tracing and context management.

        Use this method when you need to work with input that may not conform
        to the defined schema or when you have already validated the input.

        Args:
            raw_input: The raw input data to pass directly to the action function.
            on_chunk: An optional callback function to receive streaming output chunks.
            context: An optional dictionary containing context data for the execution.
            on_trace_start: An optional callback to be invoked with the trace ID
                            and span ID when the trace is started.
            telemetry_labels: Optional labels for telemetry.

        Returns:
            An awaitable ActionResponse object containing the final result and trace ID.

        Raises:
            GenkitError: If an error occurs during action execution, or if
                the action requires input but none was provided.
        """
        input_action: InputT | None = None
        if self._input_type is not None:
            if raw_input is None:
                raise GenkitError(
                    message=(
                        f"Action '{self.name}' requires input but none was provided. "
                        'Please supply a valid input payload.'
                    ),
                    status='INVALID_ARGUMENT',
                )
            input_action = self._input_type.validate_python(raw_input)

        return await self.arun(
            input=input_action,
            on_chunk=on_chunk,
            context=context,
            on_trace_start=on_trace_start,
            _telemetry_labels=telemetry_labels,
        )

    def stream(
        self,
        input: InputT | None = None,
        context: dict[str, object] | None = None,
        telemetry_labels: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> tuple[AsyncIterator[ChunkT], asyncio.Future[ActionResponse[OutputT]]]:
        """Executes the action asynchronously and provides a streaming response.

        This method initiates an asynchronous action execution and returns immediately
        with a tuple containing an async iterator for the chunks and a future for the
        final response.

        Args:
            input: The input data for the action. It should conform to the action's
                   input schema.
            context: An optional dictionary containing context data for the execution.
            telemetry_labels: Optional labels for telemetry.
            timeout: Optional timeout for the stream.

        Returns:
            A tuple: (chunk_iterator, final_response_future)
            - chunk_iterator: An AsyncIterator yielding output chunks as they become available.
            - final_response_future: An asyncio.Future that will resolve to the
                                     complete ActionResponse when the action finishes.
        """
        stream: Channel[ChunkT, ActionResponse[OutputT]] = Channel(timeout=timeout)

        def send_chunk(c: object) -> None:
            stream.send(cast(ChunkT, c))

        resp = self.arun(
            input=input,
            context=context,
            _telemetry_labels=telemetry_labels,
            on_chunk=send_chunk,
        )
        stream.set_close_future(asyncio.create_task(resp))

        result_future: asyncio.Future[ActionResponse[OutputT]] = asyncio.Future()
        stream.closed.add_done_callback(lambda _: result_future.set_result(stream.closed.result()))

        return (stream, result_future)

    def _initialize_io_schemas(
        self,
        action_args: list[str],
        arg_types: list[type],
        annotations: dict[str, Any],
        _input_spec: inspect.FullArgSpec,
    ) -> None:
        """Initializes input/output schemas based on function signature and hints.

        Uses Pydantic's TypeAdapter to generate JSON schemas for the first
        argument (if present) and the return type annotation (if present).
        Stores schemas on the instance (input_schema, output_schema) and in
        the metadata dictionary.

        Args:
            action_args: List of detected argument names.
            arg_types: List of detected argument types.
            annotations: Type annotations dict from function signature.
            _input_spec: The FullArgSpec object from inspecting the function.

        Raises:
            TypeError: If the function has more than two arguments.
        """
        if len(action_args) > 2:
            raise TypeError(f'can only have up to 2 arg: {action_args}')

        if len(action_args) > 0:
            type_adapter = TypeAdapter(arg_types[0])
            self._input_schema: dict[str, object] = type_adapter.json_schema()
            self._input_type: TypeAdapter[Any] | None = type_adapter
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


class ActionMetadata(BaseModel):
    """Metadata for actions."""

    kind: ActionKind
    name: str
    description: str | None = None
    input_schema: object | None = None
    input_json_schema: dict[str, object] | None = None
    output_schema: object | None = None
    output_json_schema: dict[str, object] | None = None
    stream_schema: object | None = None
    metadata: dict[str, object] | None = None


_SyncTracingWrapper = Callable[[object | None, ActionRunContext], ActionResponse[Any]]
_AsyncTracingWrapper = Callable[[object | None, ActionRunContext], Awaitable[ActionResponse[Any]]]


def _make_tracing_wrappers(
    name: str,
    kind: ActionKind,
    span_metadata: dict[str, SpanAttributeValue],
    n_action_args: int,
    fn: Callable[..., object],
) -> tuple[_SyncTracingWrapper, _AsyncTracingWrapper]:
    """Make the sync and async tracing wrappers for an action function.

    Args:
        name: The name of the action.
        kind: The kind of action.
        span_metadata: The span metadata for the action.
        n_action_args: The arguments of the action.
        fn: The function to wrap.
    """

    def _record_latency(output: object, start_time: float) -> object:
        """Record latency for the action if the output supports it.

        Args:
            output: The action output.
            start_time: The start time of the action execution.

        Returns:
            The updated action output.
        """
        latency_ms = (time.perf_counter() - start_time) * 1000
        if hasattr(output, 'latency_ms'):
            try:
                cast(_LatencyTrackable, output).latency_ms = latency_ms
            except (TypeError, ValidationError, AttributeError):
                # If immutable (e.g. Pydantic model with frozen=True), try model_copy
                if hasattr(output, 'model_copy'):
                    output = cast(_ModelCopyable, output).model_copy(update={'latency_ms': latency_ms})
        return output

    async def async_tracing_wrapper(input: object | None, ctx: ActionRunContext) -> ActionResponse[Any]:
        """Wrap the function in an async tracing wrapper.

        Args:
            input: The input to the action.
            ctx: The context to pass to the action.

        Returns:
            The action response.
        """
        afn = ensure_async(fn)
        start_time = time.perf_counter()

        with save_parent_path():
            with tracer.start_as_current_span(name) as span:
                # Format trace_id and span_id as hex strings (OpenTelemetry standard format)
                trace_id = format(span.get_span_context().trace_id, '032x')
                span_id = format(span.get_span_context().span_id, '016x')
                ctx._on_trace_start(trace_id, span_id)  # pyright: ignore[reportPrivateUsage]
                record_input_metadata(
                    span=span,
                    kind=kind,
                    name=name,
                    span_metadata=span_metadata,
                    input=input,
                )

                try:
                    match n_action_args:
                        case 0:
                            output = await afn()
                        case 1:
                            output = await afn(input)
                        case 2:
                            output = await afn(input, ctx)
                        case _:
                            raise ValueError('action fn must have 0-2 args...')
                except Exception as e:
                    # Re-raise existing GenkitError instances to avoid double-wrapping
                    if isinstance(e, GenkitError):
                        raise
                    raise GenkitError(
                        cause=e,
                        message=f'Error while running action {name}',
                        trace_id=trace_id,
                    ) from e

                output = _record_latency(output, start_time)
                record_output_metadata(span, output=output)
                return ActionResponse(response=output, trace_id=trace_id, span_id=span_id)

    def sync_tracing_wrapper(input: object | None, ctx: ActionRunContext) -> ActionResponse[Any]:
        """Wrap the function in a sync tracing wrapper.

        Args:
            input: The input to the action.
            ctx: The context to pass to the action.

        Returns:
            The action response.
        """
        start_time = time.perf_counter()

        with save_parent_path():
            with tracer.start_as_current_span(name) as span:
                # Format trace_id and span_id as hex strings (OpenTelemetry standard format)
                trace_id = format(span.get_span_context().trace_id, '032x')
                span_id = format(span.get_span_context().span_id, '016x')
                ctx._on_trace_start(trace_id, span_id)  # pyright: ignore[reportPrivateUsage]
                record_input_metadata(
                    span=span,
                    kind=kind,
                    name=name,
                    span_metadata=span_metadata,
                    input=input,
                )

                try:
                    match n_action_args:
                        case 0:
                            output = fn()
                        case 1:
                            output = fn(input)
                        case 2:
                            output = fn(input, ctx)
                        case _:
                            raise ValueError('action fn must have 0-2 args...')
                except Exception as e:
                    # Re-raise existing GenkitError instances to avoid double-wrapping
                    if isinstance(e, GenkitError):
                        raise
                    raise GenkitError(
                        cause=e,
                        message=f'Error while running action {name}',
                        trace_id=trace_id,
                    ) from e

                output = _record_latency(output, start_time)
                record_output_metadata(span, output=output)
                return ActionResponse(response=output, trace_id=trace_id, span_id=span_id)

    return sync_tracing_wrapper, async_tracing_wrapper
