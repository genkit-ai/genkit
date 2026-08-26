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

"""Tool-specific types and utilities for the Genkit framework."""

import inspect
import json
from collections.abc import Callable
from contextvars import ContextVar
from types import NoneType, UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

from opentelemetry import trace as trace_api
from pydantic import BaseModel, TypeAdapter

from genkit._core._action import Action, ActionKind, ActionRunContext
from genkit._core._error import GenkitError, GenkitInterrupt
from genkit._core._middleware import GenerateMiddlewareContext
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._typing import (
    CustomPart,
    DataPart,
    MediaPart,
    Metadata,
    MultipartToolResponse,
    Part,
    ReasoningPart,
    ResourcePart,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)

PART_VARIANTS = (
    TextPart,
    MediaPart,
    ToolRequestPart,
    ToolResponsePart,
    DataPart,
    CustomPart,
    ReasoningPart,
    ResourcePart,
)


def response(
    output: Any = None,  # noqa: ANN401 - tool output is JSON of any shape
    *,
    parts: list[Part] | Part | None = None,
    metadata: Metadata | None = None,
) -> MultipartToolResponse:
    """Build a tool result the model can see as structured output plus media.

    Return this from a tool when the reply is more than a JSON value — a caption
    and a screenshot, for example. A plain ``return value`` still works; that is
    treated as ``output`` only. ``parts`` may be one part or a list.
    """
    if metadata is not None and not isinstance(metadata, dict):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'response() metadata must be a dict, got {type(metadata).__name__}.',
        )
    return MultipartToolResponse(output=output, content=normalize_response_parts(parts), metadata=metadata)


def coerce_part(value: object) -> Part | None:
    if isinstance(value, Part):
        return value
    if isinstance(value, PART_VARIANTS):
        return Part(root=value)
    return None


def normalize_response_parts(parts: object) -> list[Part] | None:
    if parts is None:
        return None
    if isinstance(parts, list):
        out: list[Part] = []
        for item in parts:
            part = coerce_part(item)
            if part is None:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=f'response() parts must be a list of Parts, got {type(item).__name__} in the list.',
                )
            out.append(require_live_part(part))
        return out
    part = coerce_part(parts)
    if part is not None:
        return [require_live_part(part)]
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f'response() parts must be a Part or a list of Parts, got {type(parts).__name__}.',
    )


def normalize_pending_content(pending_content: object, *, tool_name: str) -> list[dict[str, Any]] | None:
    """Validate a resume stash as the same part list ``response()`` accepts."""
    if pending_content is None:
        return None
    if not isinstance(pending_content, list):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                f'Tool {tool_name!r} pendingContent must be a list of parts, got {type(pending_content).__name__}.'
            ),
        )
    out: list[dict[str, Any]] = []
    for i, item in enumerate(pending_content):
        part = coerce_part(item)
        if part is None and isinstance(item, dict):
            try:
                part = Part.model_validate(item)
            except Exception as e:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=f'Tool {tool_name!r} pendingContent[{i}] must be a part, got {type(item).__name__}.',
                    cause=e,
                ) from e
        if part is None:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'Tool {tool_name!r} pendingContent[{i}] must be a part, got {type(item).__name__}.',
            )
        dumped = dump_part(part, tool_name=tool_name, what=f'pendingContent[{i}]')
        if not wire_part_is_live(dumped):
            raise live_payload_error(tool_name=tool_name, where=f'pendingContent[{i}]')
        out.append(dumped)
    return out


DECLARED_OUTPUT_SCHEMA_KEY = 'declaredOutputSchema'


def wire_part_is_live(dumped: dict[str, Any]) -> bool:
    """True when a dumped part has a payload a model plugin can actually use."""
    if isinstance(dumped.get('text'), str):
        return True
    media = dumped.get('media')
    if isinstance(media, dict) and _usable_locator(media.get('url')):
        return True
    if dumped.get('data') is not None or dumped.get('custom') is not None:
        return True
    if isinstance(dumped.get('reasoning'), str):
        return True
    resource = dumped.get('resource')
    if isinstance(resource, dict) and _usable_locator(resource.get('uri')):
        return True
    tool_request = dumped.get('toolRequest')
    if isinstance(tool_request, dict) and _usable_locator(tool_request.get('name')):
        return True
    tool_response = dumped.get('toolResponse')
    if isinstance(tool_response, dict) and _usable_locator(tool_response.get('name')):
        return True
    return False


def _usable_locator(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parts_to_wire(parts: list[Part] | None, *, tool_name: str | None = None) -> list[dict[str, Any]] | None:
    """Dump parts the way a model plugin expects them on the tool message.

    A bare dump keeps every unused union sibling as null and uses snake_case
    field names. The model request — and anything that later re-parses that
    history — wants only the live fields, in camelCase.
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


def dump_part(part: Part, *, tool_name: str | None = None, what: str = 'content') -> dict[str, Any]:
    try:
        return part.model_dump(mode='json', by_alias=True, exclude_none=True)
    except GenkitError:
        raise
    except Exception as e:
        if tool_name is not None:
            message = f'Tool {tool_name!r} {what} is not JSON-serializable.'
        else:
            message = f'response() {what} is not JSON-serializable.'
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=message,
            cause=e,
        ) from e


def live_payload_error(*, tool_name: str, where: str) -> GenkitError:
    return GenkitError(
        status='INVALID_ARGUMENT',
        message=f'Tool {tool_name!r} {where} includes a part with no live payload.',
    )


def require_live_part(part: Part, *, tool_name: str | None = None, where: str = 'content') -> Part:
    dumped = dump_part(part, tool_name=tool_name, what=where)
    if wire_part_is_live(dumped):
        return part
    if tool_name is not None:
        raise live_payload_error(tool_name=tool_name, where=where)
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message='response() parts include a part with no live payload.',
    )


def annotation_is_envelope(ret: object) -> bool:
    """True when ``ret`` is the envelope type, including ``Optional`` / ``| None``."""
    # PEP 695 ``type Out = MultipartToolResponse`` is a TypeAliasType; unwrap it
    # so the model does not bind the envelope graph as outputSchema.
    if type(ret).__name__ == 'TypeAliasType':
        return annotation_is_envelope(getattr(ret, '__value__', None))
    if ret is MultipartToolResponse:
        return True
    origin = get_origin(ret)
    if origin is Union or origin is UnionType:
        members = [a for a in get_args(ret) if a is not NoneType and a is not type(None)]
        return bool(members) and all(annotation_is_envelope(a) for a in members)
    if isinstance(ret, str):
        cleaned = ret.replace(' ', '')
        if cleaned.startswith('Optional[') and cleaned.endswith(']'):
            return annotation_is_envelope(cleaned[len('Optional[') : -1])
        members = [m for m in cleaned.replace(',', '|').split('|') if m and m not in {'None', 'NoneType'}]
        return bool(members) and all(m.rsplit('.', 1)[-1] == 'MultipartToolResponse' for m in members)
    return False


def annotation_includes_envelope(ret: object) -> bool:
    """True when any leaf is the envelope — the model should not bind that graph."""
    if annotation_is_envelope(ret):
        return True
    if type(ret).__name__ == 'TypeAliasType':
        return annotation_includes_envelope(getattr(ret, '__value__', None))
    if isinstance(ret, str):
        return 'MultipartToolResponse' in ret.replace(' ', '')
    origin = get_origin(ret)
    if origin is None:
        return False
    return any(annotation_includes_envelope(a) for a in get_args(ret))


def return_annotation_is_envelope(func: Callable[..., Any]) -> bool:
    """True when the handler annotation includes the envelope ``run`` already returns."""
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = dict(getattr(func, '__annotations__', {}))
    return annotation_includes_envelope(hints.get('return'))


def override_output_schema(output_schema: object, *, tool_name: str) -> dict[str, Any]:
    if isinstance(output_schema, dict):
        return cast(dict[str, Any], output_schema)
    if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
        try:
            return to_json_schema(output_schema)
        except Exception as e:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f'Tool {tool_name!r} output_schema is not a JSON Schema or Pydantic model, got {output_schema!r}.'
                ),
                cause=e,
            ) from e
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=(
            f'Tool {tool_name!r} output_schema must be a Pydantic model or a JSON Schema dict, got {output_schema!r}.'
        ),
    )


def dump_tool_output(value: Any, *, tool_name: str | None = None, what: str = 'output') -> Any:  # noqa: ANN401
    """Dump structured output the way the advertised JSON Schema describes it."""
    try:
        if isinstance(value, BaseModel):
            return value.model_dump(mode='json', by_alias=True)
        return TypeAdapter(object).dump_python(value, mode='json', by_alias=True)
    except GenkitError:
        raise
    except Exception as e:
        # The handler already ran. A dump crash here would look like an
        # internal failure, and a retry would do the side effect twice.
        name = tool_name if tool_name is not None else 'tool'
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'Tool {name!r} {what} is not JSON-serializable.',
            cause=e,
        ) from e


def dump_tool_metadata(value: dict[str, Any] | None, *, tool_name: str | None = None) -> dict[str, Any] | None:
    """Dump envelope metadata the same way as structured output."""
    if value is None:
        return None
    dumped = dump_tool_output(value, tool_name=tool_name, what='metadata')
    if isinstance(dumped, dict):
        return dumped
    name = tool_name if tool_name is not None else 'tool'
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f'Tool {name!r} metadata is not a JSON object.',
    )


def as_multipart_tool_response(value: Any, *, tool_name: str | None = None) -> MultipartToolResponse:  # noqa: ANN401
    """Normalize a tool handler return into the envelope generate already speaks."""
    if isinstance(value, MultipartToolResponse):
        content = value.content
        if content:
            content = [require_live_part(part, tool_name=tool_name) for part in content]
        return MultipartToolResponse(
            output=dump_tool_output(value.output, tool_name=tool_name),
            content=content,
            metadata=dump_tool_metadata(value.metadata, tool_name=tool_name),
        )
    if isinstance(value, BaseModel):
        return MultipartToolResponse(output=dump_tool_output(value, tool_name=tool_name))
    return MultipartToolResponse(output=dump_tool_output(value, tool_name=tool_name))


class Tool:
    """A registered tool: a callable handle backed by an :class:`~genkit._core._action.Action`.

    Obtain instances via :func:`define_tool`, :func:`define_interrupt`, :func:`tool`, or the
    ``@ai.tool`` decorator rather than constructing directly.
    """

    def __init__(
        self,
        action: Action,
        *,
        declared_output_schema: dict[str, object] | None = None,
    ) -> None:
        self._action = action
        # What the model should expect as ``output``. ``action.output_schema`` is
        # the envelope ``run`` actually returns (output plus optional media).
        self._declared_output_schema = declared_output_schema

    @property
    def name(self) -> str:
        """Tool name (registry key)."""
        return self._action.name

    @property
    def description(self) -> str:
        """Human-readable description sent to the model."""
        return self._action.description or ''

    @property
    def input_schema(self) -> dict[str, object] | None:
        """JSON Schema for the tool's input, as sent on the wire."""
        return self._action.input_schema

    @property
    def output_schema(self) -> dict[str, object] | None:
        """JSON Schema for the structured ``output`` the model should expect.

        ``None`` means the handler is annotated as the envelope itself — the
        model should not bind a schema. An unannotated handler still infers
        ``{}``.
        """
        return self._declared_output_schema

    def definition(self) -> ToolDefinition:
        """Return the wire-format ToolDefinition for this tool."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
        )

    def action(self) -> Action:
        """Return the underlying :class:`~genkit._core._action.Action` registered for this tool."""
        return self._action

    async def __call__(self, *args: Any, **kwargs: Any) -> MultipartToolResponse:  # noqa: ANN401
        """Run the tool and return the envelope (structured output plus optional media)."""
        result = (await self._action.run(*args, **kwargs)).response
        return as_multipart_tool_response(result, tool_name=self.name)


# Context variables for propagating resumed metadata to tools
_tool_resumed_metadata: ContextVar[dict[str, Any] | None] = ContextVar('tool_resumed_metadata', default=None)
# Stashed copy of tool_request.input when restart replaces input (JSON; shape is per tool).
_tool_original_input: ContextVar[Any | None] = ContextVar('tool_original_input', default=None)  # noqa: ANN401


class ToolRunContext(ActionRunContext):
    """Tool execution context with interrupt support."""

    def __init__(
        self,
        ctx: ActionRunContext,
        resumed_metadata: dict[str, Any] | None = None,
        original_input: Any = None,  # noqa: ANN401 - prior tool_request.input when replacing on restart
    ) -> None:
        """Initialize from parent ActionRunContext.

        Args:
            ctx: Parent action context
            resumed_metadata: Metadata from previous interrupt (if resumed)
            original_input: Original tool input before replacement (if resumed)
        """
        super().__init__(
            context=ctx.context,
            streaming_callback=ctx.streaming_callback,
            abort_signal=ctx.abort_signal,
        )
        self.resumed_metadata = resumed_metadata
        self.original_input = original_input

    def is_resumed(self) -> bool:
        """Return True if this execution is resuming after an interrupt."""
        return self.resumed_metadata is not None


class Interrupt(GenkitInterrupt):  # noqa: N818 - public Genkit name; not renamed *Error for style
    """Exception for interrupting tool execution with user-facing API.

    Raise ``Interrupt(metadata)`` from a tool or from tool middleware (e.g. ``wrap_tool``).
    Exceptions from ``tool.run`` are wrapped in GenkitError
    with ``cause=Interrupt``; generation attaches interrupt metadata to the pending tool
    request.

    To resume, use ``respond_to_interrupt`` or ``restart_tool``.
    """

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        """Initialize an Interrupt exception.

        Args:
            metadata: Attached to the tool request on the wire. Use a plain dict; for a
                Pydantic model, pass ``m.model_dump(mode="json")``.
        """
        super().__init__()
        self.metadata: dict[str, Any] = {} if metadata is None else metadata
        if self.metadata:
            span = trace_api.get_current_span()
            if span.is_recording():
                try:
                    span.set_attribute('genkit:metadata:interrupt', json.dumps(self.metadata))
                except Exception:
                    span.set_attribute('genkit:metadata:interrupt', str(self.metadata))


def _tool_response_part(
    interrupt: ToolRequestPart,
    output: Any,  # noqa: ANN401 - arbitrary tool/interrupt reply payload (JSON)
    metadata: dict[str, Any] | None = None,
) -> ToolResponsePart:
    """Build a ``ToolResponsePart`` for an interrupted tool request (interrupt reply channel)."""
    interrupt_metadata = metadata if metadata is not None else True
    tool_req = interrupt.tool_request
    return ToolResponsePart(
        tool_response=ToolResponse(
            ref=tool_req.ref,
            name=tool_req.name,
            output=output,
        ),
        metadata={'interruptResponse': interrupt_metadata},
    )


def respond_to_interrupt(
    response: Any,  # noqa: ANN401 - user reply or tool output for resume_respond
    *,
    interrupt: ToolRequestPart,
    metadata: dict[str, Any] | None = None,
) -> ToolResponsePart:
    """Build a ``ToolResponsePart`` for a pending tool interrupt.

    Pass the return value to ``generate(..., resume_respond=interrupt_response)``.

    Args:
        response: Tool output / user reply for this interrupt.
        interrupt: The interrupted ``ToolRequestPart`` (e.g. from ``response.interrupts``).
        metadata: Optional metadata for the interrupt response channel.
    """
    return _tool_response_part(interrupt, response, metadata)


def restart_tool(
    *,
    interrupt: ToolRequestPart,
    replace_input: Any | None = None,  # noqa: ANN401 - new tool input; shape is per tool
    resumed_metadata: dict[str, Any] | None = None,
) -> ToolRequestPart:
    """Build a restart ``ToolRequestPart`` for a pending tool interrupt.

    Pass the return value to ``generate(..., resume_restart=...)``.

    Args:
        interrupt: The interrupted ``ToolRequestPart`` (e.g. from ``response.interrupts``).
        replace_input: Optional new ``tool_request.input`` for this run (previous input is
            stored in ``metadata.replacedInput`` when this is set).
        resumed_metadata: Passed to the tool as ``ToolRunContext.resumed_metadata``.

    Returns:
        A ``ToolRequestPart`` for ``resume_restart`` / message history.

    Example:
        ``restart_tool(interrupt=trp, resumed_metadata={"tool_approved": True})``
    """
    tool_req = interrupt.tool_request
    new_meta: dict[str, Any] = dict(interrupt.metadata or {})

    new_meta['resumed'] = resumed_metadata if resumed_metadata is not None else True

    new_input = tool_req.input
    if replace_input is not None:
        new_meta['replacedInput'] = tool_req.input
        new_input = replace_input

    return ToolRequestPart(
        tool_request=ToolRequest(
            name=tool_req.name,
            ref=tool_req.ref,
            input=new_input,
        ),
        metadata=new_meta,
    )


def _resume_context_from_tool_request_part(
    tool_request_part: ToolRequestPart,
) -> tuple[dict[str, Any] | None, Any | None]:
    """Read resume/restart fields from a tool request part's metadata."""
    meta = tool_request_part.metadata or {}
    raw_resumed = meta.get('resumed')
    if raw_resumed is True:
        resumed_meta: dict[str, Any] | None = {}
    elif isinstance(raw_resumed, dict):
        resumed_meta = raw_resumed
    else:
        resumed_meta = None

    original_input = meta.get('replacedInput')
    return resumed_meta, original_input


async def run_tool_request(
    *,
    tool: Action,
    tool_request_part: ToolRequestPart,
    ctx: GenerateMiddlewareContext | None = None,
) -> Any:  # noqa: ANN401 - tool output follows registered handler
    """Execute a tool request with generate-scoped context and resume metadata.

    Pipes ``GenerateMiddlewareContext.custom_context`` and ``telemetry_labels``
    into ``tool.run``, and sets resume ContextVars from ``tool_request_part``
    metadata so ``ToolRunContext`` reflects ``resumed`` / ``replacedInput``.
    """
    resumed_meta, original_input = _resume_context_from_tool_request_part(tool_request_part)
    token_meta = _tool_resumed_metadata.set(resumed_meta)
    token_input = _tool_original_input.set(original_input)
    run_context = dict(ctx.custom_context) if ctx and ctx.custom_context else None
    telemetry_labels = cast(dict[str, object], dict(ctx.telemetry_labels)) if ctx and ctx.telemetry_labels else None
    try:
        return (
            await tool.run(
                tool_request_part.tool_request.input,
                context=run_context,
                telemetry_labels=telemetry_labels,
                abort_signal=ctx.abort_signal if ctx else None,
            )
        ).response
    finally:
        _tool_resumed_metadata.reset(token_meta)
        _tool_original_input.reset(token_input)


def restart_interrupt_error(interrupt: Interrupt) -> GenkitError:
    """Build the FAILED_PRECONDITION error for an Interrupt raised during tool restart.

    Nested interrupts during restart are not supported yet. Include the underlying
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
        message = 'Tool interrupted again during a restart execution; not supported yet.'
    return GenkitError(status='FAILED_PRECONDITION', message=message, cause=interrupt)


async def run_tool_after_restart(
    *,
    tool: Action,
    restart_trp: ToolRequestPart,
    ctx: GenerateMiddlewareContext | None = None,
) -> ToolResponsePart:
    """Run a tool for ``resume_restart``: applies ``resumed`` / ``replacedInput`` from metadata.

    Sets the same context variables as the tool wrapper so ToolRunContext reflects
    a resumed run. Nested interrupts during restart are not supported and raise GenkitError.
    """
    try:
        raw = await run_tool_request(tool=tool, tool_request_part=restart_trp, ctx=ctx)
    except (GenkitError, Interrupt) as e:
        intr = (
            e.cause
            if isinstance(e, GenkitError) and isinstance(e.cause, Interrupt)
            else (e if isinstance(e, Interrupt) else None)
        )
        if intr is not None:
            raise restart_interrupt_error(intr) from e
        raise

    envelope = as_multipart_tool_response(raw, tool_name=restart_trp.tool_request.name)
    return ToolResponsePart(
        tool_response=ToolResponse(
            name=restart_trp.tool_request.name,
            ref=restart_trp.tool_request.ref,
            output=envelope.output,
            content=parts_to_wire(envelope.content, tool_name=restart_trp.tool_request.name),
        ),
        metadata=envelope.metadata,
    )


def _get_func_description(func: Callable[..., Any], description: str | None = None) -> str:
    """Return description if provided, otherwise use the function's docstring."""
    if description is not None:
        return description
    if func.__doc__ is not None:
        return func.__doc__
    return ''


def _define_tool(
    registry: Registry,
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
    output_schema: type[BaseModel] | dict[str, object] | None = None,
) -> Tool:
    """Register a function as a tool.

    Normally, the input_schema and output_schema are inferred from func. However,
    in some cases, like define_interrupt, the app developer doesn't have a way to
    express the input schema in the func signature.

    In that case, the app developer can pass in an input_schema to override the inferred schema.
    This will ensure that the model requesting the tool will see the correct input shape.
    """
    if not inspect.iscoroutinefunction(func):
        raise TypeError(f'Tool function must be async. Got sync function: {getattr(func, "__name__", repr(func))}')

    tool_name = name if name is not None else getattr(func, '__name__', None)
    if tool_name is None:
        raise ValueError(f'Cannot infer a tool name from {func!r}; pass name= explicitly.')
    tool_description = _get_func_description(func, description)

    input_spec = inspect.getfullargspec(func)

    async def tool_fn_wrapper(*args: Any) -> Any:  # noqa: ANN401 - arity dispatch; args/return follow registered tool
        # Record resumed metadata on the current span for observability.
        resumed_meta = _tool_resumed_metadata.get()
        if resumed_meta:
            span = trace_api.get_current_span()
            if span.is_recording():
                try:
                    span.set_attribute('genkit:metadata:resumed', json.dumps(resumed_meta))
                except Exception:
                    span.set_attribute('genkit:metadata:resumed', str(resumed_meta))

        # Dynamic dispatch by arity; payload types follow the registered tool (not expressible here).
        match len(input_spec.args):
            case 0:
                raw = await func()
            case 1:
                raw = await func(args[0])
            case 2:
                original_input = _tool_original_input.get()
                raw = await func(
                    args[0],
                    ToolRunContext(
                        cast(ActionRunContext, args[1]),
                        resumed_metadata=resumed_meta,
                        original_input=original_input,
                    ),
                )
            case _:
                raise ValueError('tool must have 0-2 args...')
        return as_multipart_tool_response(raw, tool_name=tool_name)

    action = registry.register_action(
        name=tool_name,
        kind=ActionKind.TOOL,
        description=tool_description,
        fn=tool_fn_wrapper,
        metadata_fn=func,
    )
    if input_schema is not None:
        action._override_input_schema(input_schema)

    # A bare return is the inner output (and the model schema). response() is
    # already the action result — don't advertise that wrapper as outputSchema
    # unless output_schema= says otherwise.
    if output_schema is not None:
        declared_output_schema = override_output_schema(output_schema, tool_name=tool_name)
    elif return_annotation_is_envelope(func):
        declared_output_schema = None
    else:
        declared_output_schema = action.output_schema
    action.metadata[DECLARED_OUTPUT_SCHEMA_KEY] = declared_output_schema
    action.output_schema = TypeAdapter(MultipartToolResponse).json_schema()

    return Tool(action, declared_output_schema=declared_output_schema)


def define_tool(
    registry: Registry,
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    *,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
    output_schema: type[BaseModel] | dict[str, object] | None = None,
) -> Tool:
    """Register a function as a tool.

    The model sees the handler's return type as ``outputSchema`` (or
    ``output_schema=`` if you pass one). ``Action.output_schema`` / Dev UI
    ``run`` advertise the envelope that can also carry media.

    Args:
        registry: The registry to register the tool in.
        func: The async function to register as a tool. Must be a coroutine function.
        name: Optional name for the tool. Defaults to the function name.
        description: Optional description. Defaults to the function's docstring.
        input_schema: Optional input schema override (Pydantic model or JSON-schema dict).
        output_schema: Optional inner schema for ``tools[].outputSchema``.
            Use this when the handler returns ``response(...)`` and you still
            want the model to see a shape for ``output``.

    Raises:
        TypeError: If func is not an async function.
    """
    return _define_tool(registry, func, name, description, input_schema=input_schema, output_schema=output_schema)


def tool(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    input_schema: type[BaseModel] | dict[str, object] | None = None,
    output_schema: type[BaseModel] | dict[str, object] | None = None,
) -> Tool:
    """Dynamically define a tool that can passed into a `generate` call.

    Compared to `define_tool`, the `tool` constructor doesn't register the tool.
    The Tool instance cannot be referenced by name later.

    Use when there are dynamic or ephemeral tools that need to be available
    for a particular `generate` call.

    Args:
        func: Async tool implementation (same 0–2 argument rules as :func:`define_tool`).
        name: Tool name for the model. Defaults to ``func.__name__``.
        description: Sent to the model. Defaults to the function docstring.
        input_schema: Optional input schema override (Pydantic model or JSON-schema dict).
        output_schema: Optional output schema the model should bind.

    Raises:
        TypeError: If ``func`` is not a coroutine function.
        ValueError: If no ``name`` is given and ``func`` has no ``__name__``.
    """
    return _define_tool(Registry(), func, name, description, input_schema=input_schema, output_schema=output_schema)


def define_interrupt(
    registry: Registry,
    name: str,
    *,
    description: str | None = None,
    request_metadata: dict[str, Any] | Callable[[Any], dict[str, Any]] | None = None,  # noqa: ANN401
    input_schema: type[BaseModel] | dict[str, object] | None = None,
) -> Tool:
    """Register a tool that always interrupts execution.

    An interrupt tool is a special tool that always raises ``Interrupt`` with
    optional metadata. This is useful for explicit human-in-the-loop checkpoints.
    For tools that sometimes run logic and sometimes interrupt, use ``define_tool``
    and raise ``Interrupt`` from the handler (or use ``ToolRunContext``).

    Args:
        registry: The registry to register the interrupt tool in
        name: Tool name (registry key)
        description: Tool description shown to the model
        request_metadata: Static metadata dict or ``(input) -> dict`` for the interrupt
        input_schema: Optional wire input schema (Pydantic model or JSON schema dict). The
            interrupt handler is typed as ``Any``; pass this so the model sees a concrete shape.

    Returns:
        The registered tool callable (same shape as ``define_tool``).

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

    return _define_tool(
        registry,
        interrupt_wrapper,
        name=name,
        description=description,
        input_schema=input_schema,
    )
