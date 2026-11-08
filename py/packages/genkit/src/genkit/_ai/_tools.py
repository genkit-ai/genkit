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

"""Tool registration, interruption, and request execution helpers for the Genkit framework."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NoReturn, TypeVar, cast, overload

from pydantic import BaseModel, ValidationError

from genkit._core._action import Action
from genkit._core._error import ErrorReason, GenkitError, Interrupt as Interrupt
from genkit._core._logger import get_logger
from genkit._core._middleware import MiddlewareContext
from genkit._core._model import MultipartToolResponse as MultipartToolResponse, Part, as_part, as_resumed
from genkit._core._registry import Registry
from genkit._core._tool import (
    ORIGINAL_OUTPUT_SCHEMA_KEY as ORIGINAL_OUTPUT_SCHEMA_KEY,
    Tool as _Tool,
    ToolContext as ToolContext,
    _tool_original_input as _tool_original_input,
    _tool_resumed_metadata as _tool_resumed_metadata,
    as_multipart_tool_response as as_multipart_tool_response,
    dump_part,
    dump_tool_metadata as dump_tool_metadata,
    dump_tool_output as dump_tool_output,
    envelope_output_type as envelope_output_type,
    live_payload_error,
    wire_part_is_live,
)
from genkit._core._typing import ToolResponse

ToolInputT = TypeVar('ToolInputT')
ToolOutputT = TypeVar('ToolOutputT')
ToolChunkT = TypeVar('ToolChunkT')


def normalize_pending_content(pending_content: object, *, tool_name: str) -> list[dict[str, Any]] | None:
    """Validate a resume stash as the same part list ``MultipartToolResponse.content`` accepts."""
    if pending_content is None:
        return None
    if not isinstance(pending_content, list):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                f'Tool {tool_name!r} pendingContent must be a list of parts, got {type(pending_content).__name__}.'
            ),
            reason=ErrorReason.INVALID_PART,
        )
    out: list[dict[str, Any]] = []
    for i, item in enumerate(pending_content):
        if isinstance(item, Part):
            part = item
        else:
            try:
                part = as_part(item)
            except ValidationError as e:
                detail = e.errors()[0]['msg'] if e.errors() else f'must be a part, got {type(item).__name__}'
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=f'Tool {tool_name!r} pendingContent[{i}]: {detail}',
                    cause=e,
                    reason=ErrorReason.INVALID_PART,
                ) from e
            except (ValueError, TypeError) as e:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=f'Tool {tool_name!r} pendingContent[{i}] must be a part, got {type(item).__name__}.',
                    cause=e,
                    reason=ErrorReason.INVALID_PART,
                ) from e
        dumped = dump_part(part, tool_name=tool_name, what=f'pendingContent[{i}]')
        if not wire_part_is_live(dumped):
            raise live_payload_error(tool_name=tool_name, where=f'pendingContent[{i}]')
        out.append(dumped)
    return out


def parts_to_wire(parts: Sequence[Part] | None, *, tool_name: str | None = None) -> list[dict[str, Any]] | None:
    """Dump parts the way a model plugin expects them on the tool message.

    A bare dump keeps unused fields as null and uses snake_case names. The
    model request — and anything that later re-parses that history — wants
    only the live fields, in camelCase.
    """
    if not parts:
        return None
    out: list[dict[str, Any]] = []
    name = tool_name if tool_name is not None else 'tool'
    for part in parts:
        dumped = dump_part(part, tool_name=name, what='content')
        if not wire_part_is_live(dumped):
            raise live_payload_error(tool_name=name, where='content')
        out.append(dumped)
    return out


logger = get_logger(__name__)


def _resume_context_from_tool_request_part(
    tool_request_part: Part,
) -> tuple[dict[str, Any] | None, Any | None]:
    """Read resume/restart fields from a tool request part's metadata."""
    meta = tool_request_part.metadata or {}
    return as_resumed(tool_request_part), meta.get('replacedInput')


class _ScopedToolChunkSink:
    """Callable chunk sink bound to a single ``run_tool_request`` lifetime."""

    __slots__ = ('_tool_request_part', '_on_tool_chunk')

    def __init__(
        self,
        tool_request_part: Part,
        on_tool_chunk: Callable[[Part, object], None] | None,
    ) -> None:
        self._tool_request_part = tool_request_part
        self._on_tool_chunk = on_tool_chunk

    def __call__(self, chunk: object) -> None:
        callback = self._on_tool_chunk
        if callback is not None:
            callback(self._tool_request_part, chunk)

    def close(self) -> None:
        self._on_tool_chunk = None


async def run_tool_request(
    *,
    tool: Action,
    tool_request_part: Part,
    ctx: MiddlewareContext | None = None,
    on_tool_chunk: Callable[[Part, object], None] | None = None,
) -> Any:  # noqa: ANN401 - tool output follows registered handler
    """Execute a tool request with generate-scoped context and resume metadata.

    Pipes ``on_tool_chunk`` (as ``on_chunk``), ``MiddlewareContext.context``,
    and ``telemetry_labels`` into ``tool._run``, and sets resume ContextVars
    from ``tool_request_part`` metadata so ``ToolContext`` reflects
    ``resumed`` / ``replacedInput``.
    """
    tool_req = tool_request_part.tool_request
    if tool_req is None:
        raise ValueError('run_tool_request needs a tool request part')
    resumed_meta, original_input = _resume_context_from_tool_request_part(tool_request_part)
    token_meta = _tool_resumed_metadata.set(resumed_meta)
    token_input = _tool_original_input.set(original_input)
    sink = _ScopedToolChunkSink(tool_request_part, on_tool_chunk) if on_tool_chunk is not None else None
    run_context = dict(ctx.context) if ctx and ctx.context else None
    telemetry_labels = cast(dict[str, object], dict(ctx.telemetry_labels)) if ctx and ctx.telemetry_labels else None
    try:
        return (
            await tool._run(
                tool_req.input,
                on_chunk=sink,
                context=run_context,
                telemetry_labels=telemetry_labels,
            )
        ).response
    finally:
        if sink is not None:
            sink.close()
        _tool_resumed_metadata.reset(token_meta)
        _tool_original_input.reset(token_input)


def restart_interrupt_error(interrupt: Interrupt) -> GenkitError:
    """Build the FAILED_PRECONDITION error for an Interrupt raised during tool restart.

    A tool cannot raise another interrupt while it is being restarted. Include the underlying
    interrupt reason (e.g. ToolApproval's ``Tool not in approved list: ...``) so the
    error points at missing approval metadata instead of sounding like a missing SDK feature.
    """
    metadata = interrupt.metadata
    if isinstance(metadata, dict):
        reason = metadata.get('message')
    elif isinstance(metadata, str):
        # Defensive: Interrupt is typed as dict metadata, but a plain string
        # argument would land here and must not AttributeError on .get().
        reason = metadata
    else:
        reason = None
    if isinstance(reason, str) and reason.strip():
        message = f'Tool interrupted again during restart: {reason}'
    else:
        message = 'Tool interrupted again during restart.'
    return GenkitError(
        status='FAILED_PRECONDITION',
        message=message,
        cause=interrupt,
        reason=ErrorReason.INVALID_RESUME,
    )


async def run_tool_after_restart(
    *,
    tool: Action,
    restart_trp: Part,
    ctx: MiddlewareContext | None = None,
    on_tool_chunk: Callable[[Part, object], None] | None = None,
) -> Part:
    """Run a tool for ``resume_restart``: applies ``resumed`` / ``replacedInput`` from metadata.

    Sets the same context variables as the tool wrapper so ToolContext reflects
    a resumed run. A tool cannot raise another interrupt while it is being restarted.
    """
    try:
        return await run_restarted_tool(tool=tool, restart_trp=restart_trp, ctx=ctx, on_tool_chunk=on_tool_chunk)
    except Interrupt as e:
        logger.debug(
            'restarted tool triggered an interrupt',
            tool=restart_trp.tool_request.name if restart_trp.tool_request else tool.name,
        )
        raise restart_interrupt_error(e) from e


async def run_restarted_tool(
    *,
    tool: Action,
    restart_trp: Part,
    ctx: MiddlewareContext | None = None,
    on_tool_chunk: Callable[[Part, object], None] | None = None,
) -> Part:
    """Like ``run_tool_after_restart``, but a second Interrupt propagates as is.

    run_restart calls this inside the wrap_tool chain. Converting the Interrupt
    there would hand middleware an Exception its catch-all could swallow,
    skipping the approval the restart is asking for.
    """
    tool_req = restart_trp.tool_request
    if tool_req is None:
        raise ValueError('run_tool_after_restart needs a tool request part')
    raw = await run_tool_request(tool=tool, tool_request_part=restart_trp, ctx=ctx, on_tool_chunk=on_tool_chunk)
    envelope = as_multipart_tool_response(raw, tool_name=tool_req.name)
    return Part(
        tool_response=ToolResponse(
            name=tool_req.name,
            ref=tool_req.ref,
            output=envelope.output,
            content=parts_to_wire(envelope.content, tool_name=tool_req.name),
        ),
        metadata=envelope.metadata,
    )


def _define_tool(
    registry: Registry,
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> _Tool[Any, Any, Any]:
    """Build a tool and register it (``/tool/<name>`` and its ``/tool.v2/<name>`` envelope).

    The return annotation is what the model binds. ``input_schema=`` is for
    cases like define_interrupt where the handler signature cannot express
    the input shape.
    """
    built: _Tool[Any, Any, Any] = _Tool(func, name=name, description=description, input_schema=input_schema)
    registry.register_action_from_instance(built)
    return built


# Overload order matters: type checkers pick the first shape that fits.
# - ``(ctx, x: Any)`` fits ``[InputT, ToolContext]`` with the ctx as the input,
#   and ``(x: Any)`` / ``(x: object)`` fit the ctx-only shape. The ``object``
#   shapes come first and catch them; a real ``ToolContext`` param can't take
#   an ``object``, so it skips them.
# - The context-only shape comes before input-only because a ``ctx`` function
#   also fits ``[InputT]``. It still returns ``Tool[Any, ...]`` so that shape
#   keeps accepting an input.
# - ty types ``(ctx, x: Any)`` as Unknown: the typing spec's overload rule for
#   Any arguments makes it ambiguous, and no order fixes that.
@overload
def define_tool(
    registry: Registry,
    func: Callable[[ToolContext[ToolChunkT], object], Awaitable[ToolOutputT]],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> _Tool[Any, ToolOutputT, ToolChunkT]: ...


@overload
def define_tool(
    registry: Registry,
    func: Callable[[ToolInputT, ToolContext[ToolChunkT]], Awaitable[ToolOutputT]],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> _Tool[ToolInputT, ToolOutputT, ToolChunkT]: ...


@overload
def define_tool(
    registry: Registry,
    func: Callable[[ToolContext[ToolChunkT], ToolInputT], Awaitable[ToolOutputT]],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> _Tool[ToolInputT, ToolOutputT, ToolChunkT]: ...


@overload
def define_tool(
    registry: Registry,
    func: Callable[[object], Awaitable[ToolOutputT]],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> _Tool[Any, ToolOutputT]: ...


@overload
def define_tool(
    registry: Registry,
    func: Callable[[ToolContext[ToolChunkT]], Awaitable[ToolOutputT]],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> _Tool[Any, ToolOutputT, ToolChunkT]: ...


@overload
def define_tool(
    registry: Registry,
    func: Callable[[ToolInputT], Awaitable[ToolOutputT]],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> _Tool[ToolInputT, ToolOutputT]: ...


@overload
def define_tool(
    registry: Registry,
    func: Callable[[], Awaitable[ToolOutputT]],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> _Tool[None, ToolOutputT]: ...


def define_tool(
    registry: Registry,
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> _Tool[Any, Any, Any]:
    """Register a function as a tool.

    The model sees the handler's return annotation as ``outputSchema``.
    ``await tool(x)`` returns what the function returned. The registry also
    stores ``/tool.v2/<name>``, which returns the ``MultipartToolResponse``
    envelope; that's the one the Dev UI runs.

    Args:
        registry: The registry to register the tool in.
        func: The async function to register as a tool. Must be a coroutine function.
            It takes at most one input, annotated with a type that has a JSON
            schema (``Any`` accepts anything), plus an optional parameter
            annotated ``ToolContext`` in any position.
        name: Optional name for the tool. Defaults to the function name.
        description: Optional description. Defaults to the function's docstring.
        input_schema: Optional input schema override (Pydantic model or JSON-schema dict).

    Raises:
        TypeError: If func is not an async function, has more than one input,
            has an input with no annotation or no JSON schema, or has more than
            one ``ToolContext`` parameter.
    """
    return _define_tool(registry, func, name, description, input_schema=input_schema)


def define_interrupt(
    registry: Registry,
    name: str,
    *,
    description: str | None = None,
    request_metadata: dict[str, Any] | Callable[[Any], dict[str, Any]] | None = None,  # noqa: ANN401
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> _Tool[Any, NoReturn]:
    """Register a tool that always interrupts execution.

    An interrupt tool is a special tool that always raises ``Interrupt`` with
    optional metadata. This is useful for explicit human-in-the-loop checkpoints.
    For tools that sometimes run logic and sometimes interrupt, use ``define_tool``
    and raise ``Interrupt`` from the handler (or use ``ToolContext``).

    Args:
        registry: The registry to register the interrupt tool in
        name: Tool name (registry key)
        description: Tool description shown to the model
        request_metadata: Static metadata dict or ``(input) -> dict`` for the interrupt
        input_schema: Optional wire input schema (Pydantic model or JSON schema dict). The
            interrupt handler is typed as ``Any``; pass this so the model sees a concrete shape.

    Returns:
        The registered tool. Awaiting it directly raises ``Interrupt``; it
        never returns a value.

    Example:
        def get_meta(input: dict) -> dict:
            return {"action": input.get("action"), "requires_approval": True}

        confirm = define_interrupt(
            registry,
            "confirm",
            description="Requires user approval",
            request_metadata=get_meta,
        )
    """

    async def interrupt_wrapper(input: Any) -> Any:  # noqa: ANN401 - wire JSON args; never returns (raises Interrupt)
        # Interrupt tools accept arbitrary JSON args like any tool.
        meta = None
        if callable(request_metadata):
            meta = request_metadata(input)
        elif request_metadata is not None:
            meta = request_metadata
        raise Interrupt(meta)

    # The handler stays annotated ``Any`` so the model is told it may return anything.
    return cast(
        _Tool[Any, NoReturn],
        _define_tool(
            registry,
            interrupt_wrapper,
            name=name,
            description=description,
            input_schema=input_schema,
        ),
    )
