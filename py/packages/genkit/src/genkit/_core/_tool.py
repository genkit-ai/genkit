# Copyright 2026 Google LLC
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

"""Core tool action, run context, and response envelope helpers for the Genkit framework."""

from __future__ import annotations

import base64
import inspect
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, overload

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypeVar

from genkit._core._action import (
    Action,
    ActionKind,
    ActionRunContext,
    get_func_description,
    resolve_type_hints,
)
from genkit._core._error import ErrorReason, GenkitError
from genkit._core._model import MultipartToolResponse, Part, as_part
from genkit._core._schema import to_json_schema
from genkit._core._telemetry._instrumentation import set_custom_metadata_attributes
from genkit._core._typing import MultipartToolResponse as MultipartToolResponseData, ToolDefinition

InputT = TypeVar('InputT', default=Any)
OutputT = TypeVar('OutputT', default=Any)
ChunkT = TypeVar('ChunkT', default=Any)

# Solved per call by the ``Tool(...)`` overloads below.
_I = TypeVar('_I')
_O = TypeVar('_O')
_C = TypeVar('_C')

# Metadata key for the schema the model binds as the tool's ``outputSchema``.
# For ``-> MultipartToolResponse[T]`` that's ``T``; for ``-> T`` it's ``T``.
ORIGINAL_OUTPUT_SCHEMA_KEY = 'originalOutputSchema'

# Context variables for propagating resumed metadata to tools.
_tool_resumed_metadata: ContextVar[dict[str, Any] | None] = ContextVar('tool_resumed_metadata', default=None)
# Stashed copy of tool_request.input when restart replaces input (JSON; shape is per tool).
_tool_original_input: ContextVar[Any | None] = ContextVar('tool_original_input', default=None)  # noqa: ANN401


def _usable_locator(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


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


def _wire_fallback(obj: object) -> object:
    if isinstance(obj, bytes):
        try:
            return base64.b64encode(obj).decode('utf-8')
        except Exception:
            return '<bytes>'
    raise TypeError(f'{type(obj).__name__} is not JSON-serializable')


def dump_part(part: Part, *, tool_name: str | None = None, what: str = 'content') -> dict[str, Any]:
    """Dump a part to JSON-compatible wire dict, raising INVALID_ARGUMENT if unserializable."""
    try:
        return as_part(part).model_dump(mode='json', by_alias=True, exclude_none=True, fallback=_wire_fallback)
    except GenkitError:
        raise
    except Exception as e:
        if tool_name is not None:
            message = f'Tool {tool_name!r} {what} is not JSON-serializable.'
        else:
            message = f'MultipartToolResponse {what} is not JSON-serializable.'
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=message,
            cause=e,
            reason=ErrorReason.INVALID_PART,
        ) from e


def live_payload_error(*, tool_name: str, where: str) -> GenkitError:
    """Build the INVALID_ARGUMENT error for a part with no usable payload."""
    return GenkitError(
        status='INVALID_ARGUMENT',
        message=f'Tool {tool_name!r} {where} includes a part with no live payload.',
        reason=ErrorReason.INVALID_PART,
    )


def require_live_part(part: Part, *, tool_name: str | None = None, where: str = 'content') -> Part:
    """Return ``part`` as a :class:`Part` if it carries a live payload, else raise ``INVALID_ARGUMENT``."""
    p = as_part(part)
    dumped = dump_part(p, tool_name=tool_name, what=where)
    if wire_part_is_live(dumped):
        return p
    if tool_name is not None:
        raise live_payload_error(tool_name=tool_name, where=where)
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message='MultipartToolResponse content includes a part with no live payload.',
        reason=ErrorReason.INVALID_PART,
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
            reason=ErrorReason.INVALID_INPUT,
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
        reason=ErrorReason.INVALID_INPUT,
    )


def as_multipart_tool_response(value: Any, *, tool_name: str | None = None) -> MultipartToolResponse[Any]:  # noqa: ANN401
    """Normalize a tool handler return into the envelope generate already speaks."""
    if isinstance(value, (MultipartToolResponse, MultipartToolResponseData)):
        content = value.content
        parts: list[Part] | None = None
        if content:
            parts = [require_live_part(as_part(part), tool_name=tool_name) for part in content]
        return MultipartToolResponse(
            output=dump_tool_output(value.output, tool_name=tool_name),
            content=parts,
            metadata=dump_tool_metadata(value.metadata, tool_name=tool_name),
        )
    return MultipartToolResponse(output=dump_tool_output(value, tool_name=tool_name))


NOT_ENVELOPE = object()


def envelope_output_type(ret: object) -> object:
    """Inner ``output`` type from a return annotation, or ``NOT_ENVELOPE``.

    People write ``-> MultipartToolResponse[ShotOut]`` so the model binds
    ``ShotOut``. This only peels that ``[T]``. Anything else (``-> str``,
    ``-> ShotOut``, no annotation) is ``NOT_ENVELOPE`` and define uses the
    inferred schema instead.
    """
    # ``type Shot = MultipartToolResponse[ShotOut]`` — unwrap and try again.
    if type(ret).__name__ == 'TypeAliasType':
        return envelope_output_type(getattr(ret, '__value__', None))
    # ``MultipartToolResponse[ShotOut]`` is a real Pydantic subclass at
    # runtime. ``get_origin`` is None; ``[ShotOut]`` lives here.
    meta = getattr(ret, '__pydantic_generic_metadata__', None)
    if isinstance(meta, dict):
        origin = meta.get('origin')
        if origin is MultipartToolResponse or origin is MultipartToolResponseData:
            args = meta.get('args') or ()
            return args[0] if args else Any
    origin = get_origin(ret)
    # Typing-only form, if the annotation never became a Pydantic subclass.
    if origin is MultipartToolResponse or origin is MultipartToolResponseData:
        args = get_args(ret)
        return args[0] if args else Any
    # Bare ``-> MultipartToolResponse``: output is anything.
    if ret is MultipartToolResponse or ret is MultipartToolResponseData:
        return Any
    # ``-> MultipartToolResponse[ShotOut] | None`` — one real type, peel it.
    if origin is Union or origin is UnionType:
        members = [a for a in get_args(ret) if a is not type(None)]
        if len(members) == 1:
            return envelope_output_type(members[0])
    return NOT_ENVELOPE


def model_schema_from_return_annotation(
    func: Callable[..., Any],
    *,
    inferred: dict[str, object] | None,
) -> dict[str, object] | None:
    """JSON Schema the model should bind, from the handler's return annotation."""
    inner = envelope_output_type(resolve_type_hints(func).get('return'))
    if inner is NOT_ENVELOPE:
        return inferred
    if inner is Any or inner is object:
        return None
    return to_json_schema(cast(type | dict[str, Any] | str, inner))


class ToolContext(ActionRunContext[ChunkT]):
    """What a tool sees while it runs: request context, streaming, and resume state.

    Annotate a tool parameter with it to receive it:

        @ai.tool()
        async def check_allergens(dish: str, ctx: ToolContext) -> str:
            return 'resumed' if ctx.is_resumed() else ctx.context.get('user', 'guest')

    ``ToolContext`` is not a ``FlowContext``. A helper shared by a tool and a
    flow takes the union:

        # 1. Read the caller's uid from either context
        def user_id(ctx: FlowContext | ToolContext) -> str:
            return ctx.context['auth']['uid']

        # 2. Call it from the tool
        @ai.tool()
        async def list_allergies(guest: str, ctx: ToolContext) -> str:
            return f'{guest} ({user_id(ctx)}): shellfish'

        await list_allergies('Ana', context={'auth': {'uid': 'diner-42'}})
        # => 'Ana (diner-42): shellfish'
    """

    def __init__(
        self,
        ctx: ActionRunContext[ChunkT],
        resumed_metadata: dict[str, Any] | None = None,
        original_input: Any = None,  # noqa: ANN401 - prior tool_request.input when replacing on restart
    ) -> None:
        """Initialize from the tool action's run context.

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


async def call_tool_fn(
    tool: Action[Any, Any, Any, Any],
    func: Callable[..., Awaitable[OutputT]],
    raw_input: object,
    ctx: ActionRunContext[Any],
) -> OutputT:
    """Call a tool function with a ``ToolContext`` carrying resume state.

    Both the ``tool`` action and its ``tool.v2`` envelope action run the
    function through here, so resume metadata reaches it either way.
    """
    resumed_meta = _tool_resumed_metadata.get()
    if resumed_meta:
        # Record resumed metadata on the current span for observability.
        set_custom_metadata_attributes({'resumed': resumed_meta})
    # The ctx parameter always receives this ToolContext, whatever context
    # class it is annotated with; only ToolContext (or a union containing it)
    # types it correctly.
    tool_ctx = ToolContext(ctx, resumed_metadata=resumed_meta, original_input=_tool_original_input.get())
    return await tool.params.call(func, raw_input, tool_ctx)


class Tool(Action[InputT, OutputT, ChunkT, None]):
    """A tool the model can call: an :class:`~genkit._core._action.Action` of kind ``tool``.

    ``await tool(x)`` is ``(await tool.run(x)).response``: what the function
    returned. A ``str`` comes back as the string, a Pydantic model as the
    instance, and ``response(...)`` as that ``MultipartToolResponse``.

    Each tool also carries a ``tool.v2`` action that returns the
    ``MultipartToolResponse`` envelope. The registry stores both, like JS and
    Go; the Dev UI lists and runs the ``tool.v2`` one.
    """

    _wire_action: Action[InputT, MultipartToolResponse[Any], ChunkT, None] | None

    # One overload per handler shape, same as ``ai.tool`` / ``define_tool``, so
    # ``Tool(place_order)`` infers ``Tool[Order, Receipt, Step]`` from the function.
    @overload
    def __init__(
        self: Tool[Any, _O, _C],
        func: Callable[[ToolContext[_C], object], Awaitable[_O]],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: Tool[_I, _O, _C],
        func: Callable[[_I, ToolContext[_C]], Awaitable[_O]],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: Tool[_I, _O, _C],
        func: Callable[[ToolContext[_C], _I], Awaitable[_O]],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: Tool[Any, _O, Any],
        func: Callable[[object], Awaitable[_O]],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: Tool[Any, _O, _C],
        func: Callable[[ToolContext[_C]], Awaitable[_O]],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: Tool[_I, _O, Any],
        func: Callable[[_I], Awaitable[_O]],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: Tool[None, _O, Any],
        func: Callable[[], Awaitable[_O]],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None: ...

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None:
        """Build a tool from an async function without registering it.

        ``define_tool`` / ``@ai.tool()`` register it. Passed straight to
        ``generate(tools=[...])``, it is an ephemeral tool for that call.

        Args:
            func: Async tool implementation. It takes at most one input,
                annotated with a type that has a JSON schema, plus an optional
                ``ToolContext`` parameter.
            name: Tool name for the model. Defaults to ``func.__name__``.
            description: Sent to the model. Defaults to the function docstring.
            input_schema: Optional input schema override (Pydantic model or JSON-schema dict).

        Raises:
            TypeError: If ``func`` is not a coroutine function, takes more than one input,
                or its input has no annotation or no JSON schema.
            ValueError: If no ``name`` is given and ``func`` has no ``__name__``.

        Example:
            from genkit import Tool

            async def lookup_price(sku: str) -> str:
                return '12.00'

            ephemeral = Tool(lookup_price, name='lookup_price')
            res = await ai.generate(prompt='Price of SKU-1?', tools=[ephemeral])
        """
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f'Tool function must be async. Got sync function: {getattr(func, "__name__", repr(func))}')
        tool_name = name if name is not None else getattr(func, '__name__', None)
        if tool_name is None:
            raise ValueError(f'Cannot infer a tool name from {func!r}; pass name= explicitly.')
        tool_description = get_func_description(func, description)
        typed_func = cast(Callable[..., Awaitable[OutputT]], func)
        self._wire_action = None

        async def tool_fn(raw_input: object, ctx: ActionRunContext[Any]) -> OutputT:
            return await call_tool_fn(self, typed_func, raw_input, ctx)

        super().__init__(
            kind=ActionKind.TOOL,
            name=tool_name,
            fn=tool_fn,
            metadata_fn=func,
            description=tool_description,
            input_schema=input_schema,
        )
        model_schema = model_schema_from_return_annotation(func, inferred=self.output_schema)
        self.metadata[ORIGINAL_OUTPUT_SCHEMA_KEY] = model_schema

        async def wire_fn(raw_input: object, ctx: ActionRunContext[Any]) -> MultipartToolResponse[Any]:
            raw = await call_tool_fn(self, typed_func, raw_input, ctx)
            return as_multipart_tool_response(raw, tool_name=self.name)

        wire: Action[InputT, MultipartToolResponse[Any], ChunkT, None] = Action(
            kind=ActionKind.TOOL_V2,
            name=tool_name,
            fn=wire_fn,
            metadata_fn=func,
            description=tool_description,
            input_schema=input_schema,
        )
        wire.output_schema = TypeAdapter(MultipartToolResponseData).json_schema()
        wire.metadata[ORIGINAL_OUTPUT_SCHEMA_KEY] = model_schema
        self._wire_action = wire

    @classmethod
    def from_action(cls, action: Action[InputT, OutputT, ChunkT, Any]) -> Tool[InputT, OutputT, ChunkT]:
        """View a tool ``Action`` that wasn't built as a ``Tool`` (plugin, DAP, hand-registered) as one.

        Shares the action's function and metadata; there is no ``tool.v2`` companion.
        """
        if isinstance(action, Tool):
            return action
        return _ActionTool(action)

    @property
    def model_output_schema(self) -> dict[str, object] | None:
        """JSON Schema the model binds as this tool's ``outputSchema``."""
        if ORIGINAL_OUTPUT_SCHEMA_KEY in self.metadata:
            original = self.metadata[ORIGINAL_OUTPUT_SCHEMA_KEY]
            return cast(dict[str, object], original) if isinstance(original, dict) else None
        return self.output_schema

    @model_output_schema.setter
    def model_output_schema(self, value: dict[str, object] | None) -> None:
        self._metadata[ORIGINAL_OUTPUT_SCHEMA_KEY] = value
        if self._wire_action is not None:
            self._wire_action.metadata[ORIGINAL_OUTPUT_SCHEMA_KEY] = value

    @property
    def wire_action(self) -> Action[InputT, MultipartToolResponse[Any], ChunkT, None] | None:
        """The ``tool.v2`` action: same function, returns the ``MultipartToolResponse`` envelope.

        The registry stores it next to the tool under ``/tool.v2/<name>``; the
        Dev UI and other wire consumers run it. ``None`` for a ``Tool`` viewed
        from a plain action.
        """
        return self._wire_action

    def definition(self) -> ToolDefinition:
        """Return the wire-format ToolDefinition for this tool.

        ``output_schema`` is what the model binds: ``T`` for ``-> T`` and for
        ``-> MultipartToolResponse[T]``, ``None`` for a bare
        ``-> MultipartToolResponse``.
        """
        return ToolDefinition(
            name=self.name,
            description=self.description or '',
            input_schema=self.input_schema,
            output_schema=self.model_output_schema,
        )

    async def _invoke(self, input: object | None, ctx: ActionRunContext) -> OutputT:
        """Run the function, then check its result the way generate would send it to a model.

        A value the model couldn't receive (unserializable output or metadata,
        a part with no payload) raises ``INVALID_ARGUMENT`` on every call path,
        not just inside ``generate``. The function's own value is returned.
        """
        output = await super()._invoke(input, ctx)
        as_multipart_tool_response(output, tool_name=self.name)
        return output


class _ActionTool(Tool[InputT, OutputT, ChunkT]):
    """A ``Tool`` view over a plain ``Action`` with no ``tool.v2`` companion."""

    def __init__(self, action: Action[InputT, OutputT, ChunkT, Any]) -> None:
        self._wire_action = None
        Action.__init__(
            self,
            kind=action.kind,
            name=action.name,
            fn=action._fn,
            metadata_fn=action._user_fn if action._fn_is_wrapper else None,
            description=action.description,
            metadata=action.metadata,
            span_metadata=action._span_metadata,
            config_schema=action.config_schema,
            _strict_io=action._strict_io,
            _io_schemas=(action.input_schema, action.output_schema or {}),
        )
        self._input_type = action._input_type
        self._input_class = action._input_class
        self.input_schema = action.input_schema
        self._output_type = action._output_type
        self._set_output_schema(action.output_schema)
        self._init_type = action._init_type
        if hasattr(action, '_init_schema'):
            self._init_schema = action._init_schema
