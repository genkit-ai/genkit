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
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, cast

from pydantic import BaseModel

from genkit._core._action import Action, ActionKind, ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._registry import Registry
from genkit._core._typing import ToolDefinition, ToolRequest, ToolRequestPart, ToolResponse, ToolResponsePart


class Tool:
    """A registered tool: a callable handle backed by an :class:`~genkit._core._action.Action`.

    Obtain instances via :func:`define_tool`, :func:`define_interrupt`, or the
    ``@ai.tool`` decorator rather than constructing directly.
    """

    def __init__(self, action: Action) -> None:
        self._action = action

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
        """JSON Schema for the tool's output."""
        return self._action.output_schema

    def definition(self) -> ToolDefinition:
        """Return the wire-format ToolDefinition for this tool."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
        )

    def action(self) -> Action[Any, Any, Any]:
        """Return the underlying :class:`~genkit._core._action.Action` registered for this tool."""
        return self._action

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run the tool and return the unwrapped response value."""
        return (await self._action.run(*args, **kwargs)).response

    def restart(
        self,
        replace_input: Any | None = None,  # noqa: ANN401
        *,
        interrupt: ToolRequestPart,
        resumed_metadata: dict[str, Any] | None = None,
    ) -> ToolRequestPart:
        """Create a restart request for an interrupted tool call.

        Args:
            replace_input: Optional new ``tool_request.input`` for this run (previous input is
                stored in ``metadata.replacedInput`` when this is set).
            interrupt: The interrupted ``ToolRequestPart`` (e.g. from ``response.interrupts``).
            resumed_metadata: Passed to the tool as ``ToolRunContext.resumed_metadata``.

        Returns:
            A ``ToolRequestPart`` for ``resume_restart`` / message history.

        Example:
            ``pay_invoice.restart({**trp.tool_request.input, "confirmed": True}, interrupt=trp,``
            ``resumed_metadata={"by": "bob"})``
        """
        tool_req = interrupt.tool_request
        if tool_req.name != self.name:
            raise ValueError(f"Interrupt is for tool '{tool_req.name}', not '{self.name}'")

        existing_meta = interrupt.metadata or {}
        new_meta: dict[str, Any] = dict(existing_meta) if existing_meta else {}

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
        super().__init__(context=ctx.context)
        self.resumed_metadata = resumed_metadata
        self.original_input = original_input

    def is_resumed(self) -> bool:
        """Return True if this execution is resuming after an interrupt."""
        return self.resumed_metadata is not None


class Interrupt(Exception):  # noqa: N818 - public Genkit name; not renamed *Error for style
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
    replace_input: Any | None = None,  # noqa: ANN401 - new tool input; shape is per tool
    *,
    tool: Tool,
    interrupt: ToolRequestPart,
    resumed_metadata: dict[str, Any] | None = None,
) -> ToolRequestPart:
    """Build a restart ``ToolRequestPart`` for a pending tool interrupt.

    Thin wrapper around :meth:`Tool.restart` for symmetry with
    :func:`respond_to_interrupt`. Pass the return value to
    ``generate(..., resume_restart=...)``.

    Args:
        replace_input: Optional new ``tool_request.input`` for this run (previous input is
            stored in ``metadata.replacedInput`` when this is set).
        tool: The registered :class:`Tool` that was interrupted.
        interrupt: The interrupted ``ToolRequestPart`` (e.g. from ``response.interrupts``).
        resumed_metadata: Passed to the tool as ``ToolRunContext.resumed_metadata``.

    Returns:
        A ``ToolRequestPart`` for ``resume_restart`` / message history.

    Example:
        ``restart_tool({**trp.tool_request.input, "confirmed": True}, tool=pay_invoice,``
        ``interrupt=trp, resumed_metadata={"by": "bob"})``
    """
    return tool.restart(replace_input, interrupt=interrupt, resumed_metadata=resumed_metadata)


async def run_tool_after_restart(tool: Action[Any, Any, Any], restart_trp: ToolRequestPart) -> ToolResponsePart:
    """Run a tool for ``resume_restart``: applies ``resumed`` / ``replacedInput`` from metadata.

    Sets the same context variables as the tool wrapper so ToolRunContext reflects
    a resumed run. Nested interrupts during restart are not supported and raise GenkitError.
    """
    meta = restart_trp.metadata or {}
    raw_resumed = meta.get('resumed')
    if raw_resumed is True:
        resumed_meta: dict[str, Any] | None = {}
    elif isinstance(raw_resumed, dict):
        resumed_meta = raw_resumed
    else:
        resumed_meta = None
    original_input = meta.get('replacedInput')

    token_meta = _tool_resumed_metadata.set(resumed_meta)
    token_input = _tool_original_input.set(original_input)
    try:
        try:
            tool_response = (await tool.run(restart_trp.tool_request.input)).response
        except GenkitError as e:
            if e.cause and isinstance(e.cause, Interrupt):
                raise GenkitError(
                    status='FAILED_PRECONDITION',
                    message='Tool interrupted again during a restart execution; not supported yet.',
                    cause=e.cause,
                ) from e
            raise
    finally:
        _tool_resumed_metadata.reset(token_meta)
        _tool_original_input.reset(token_input)

    return ToolResponsePart(
        tool_response=ToolResponse(
            name=restart_trp.tool_request.name,
            ref=restart_trp.tool_request.ref,
            output=tool_response.model_dump() if isinstance(tool_response, BaseModel) else tool_response,
        )
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
) -> Tool:
    """Register a function as a tool.

    Normally, the input_schema and output_schem are inferred from func. However,
    in some cases, like define_interrupt, the app developer doesn't have a way to
    express the input schema in the func signature.

    In that case, the app developer can pass in an input_schema to override the inferred schema.
    This will ensure that the model requesting the tool will see the correct input shape.
    """
    if not inspect.iscoroutinefunction(func):
        raise TypeError(f'Tool function must be async. Got sync function: {getattr(func, "__name__", repr(func))}')

    tool_name = name if name is not None else getattr(func, '__name__', 'unnamed_tool')
    tool_description = _get_func_description(func, description)

    input_spec = inspect.getfullargspec(func)

    async def tool_fn_wrapper(*args: Any) -> Any:  # noqa: ANN401 - arity dispatch; args/return follow registered tool
        # Dynamic dispatch by arity; payload types follow the registered tool (not expressible here).
        match len(input_spec.args):
            case 0:
                return await func()
            case 1:
                return await func(args[0])
            case 2:
                # Read from context variables for resumed metadata
                resumed_meta = _tool_resumed_metadata.get()
                original_input = _tool_original_input.get()
                return await func(
                    args[0],
                    ToolRunContext(
                        cast(ActionRunContext, args[1]),
                        resumed_metadata=resumed_meta,
                        original_input=original_input,
                    ),
                )
            case _:
                raise ValueError('tool must have 0-2 args...')

    action = registry.register_action(
        name=tool_name,
        kind=ActionKind.TOOL,
        description=tool_description,
        fn=tool_fn_wrapper,
        metadata_fn=func,
    )
    if input_schema is not None:
        action._override_input_schema(input_schema)

    return Tool(action)


def define_tool(
    registry: Registry,
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Register a function as a tool.

    Tool input/output JSON Schemas are inferred from ``func`` (first parameter and return type).

    Args:
        registry: The registry to register the tool in.
        func: The async function to register as a tool. Must be a coroutine function.
        name: Optional name for the tool. Defaults to the function name.
        description: Optional description. Defaults to the function's docstring.

    Raises:
        TypeError: If func is not an async function.
    """
    return _define_tool(registry, func, name, description)


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
