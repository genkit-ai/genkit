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
from typing import Any, NoReturn, cast

from pydantic import BaseModel

from genkit._core._action import Action, ActionKind, ActionRunContext
from genkit._core._registry import Registry
from genkit._core._typing import Part, ToolDefinition, ToolRequest, ToolRequestPart, ToolResponse, ToolResponsePart


class Tool:
    """A registered tool: a callable handle backed by an :class:`~genkit._core._action.Action`.

    Obtain instances via :func:`define_tool` or the ``@ai.tool`` decorator rather than constructing directly.
    """

    def __init__(self, action: Action) -> None:
        self._action = action

    @property
    def action(self) -> Action:
        """Underlying TOOL :class:`~genkit._core._action.Action` registered for this tool."""
        return self._action

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

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run the tool and return the unwrapped response value."""
        return (await self._action.run(*args, **kwargs)).response


class ToolRunContext(ActionRunContext):
    """Tool execution context with interrupt support."""

    def __init__(
        self,
        ctx: ActionRunContext,
    ) -> None:
        """Initialize from parent ActionRunContext."""
        super().__init__(context=ctx.context)

    def interrupt(self, metadata: dict[str, Any] | None = None) -> NoReturn:
        """Raise ToolInterruptError to pause execution (e.g., for user input)."""
        raise ToolInterruptError(metadata=metadata)


# TODO(#4346): make this extend GenkitError once it has INTERRUPTED status
class ToolInterruptError(Exception):
    """Controlled interruption of tool execution (e.g., to request user input)."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        """Initialize with optional interrupt metadata."""
        super().__init__()
        self.metadata: dict[str, Any] = metadata or {}


def tool_response(
    interrupt: Part | ToolRequestPart,
    response_data: object | None = None,
    metadata: dict[str, Any] | None = None,
) -> Part:
    """Create a ToolResponse Part for an interrupted tool request."""
    # TODO(#4347): validate against tool schema
    tool_request = interrupt.root.tool_request if isinstance(interrupt, Part) else interrupt.tool_request

    interrupt_metadata: dict[str, Any] | bool = True
    if isinstance(metadata, dict):
        interrupt_metadata = metadata
    elif metadata:
        interrupt_metadata = metadata

    tr = cast(ToolRequest, tool_request)
    return Part(
        root=ToolResponsePart(
            tool_response=ToolResponse(
                name=tr.name,
                ref=tr.ref,
                output=response_data,
            ),
            metadata={'interruptResponse': interrupt_metadata},
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

    The input_schema and output_schema are normally inferred from ``func``. Pass
    ``input_schema`` to override the inferred wire schema when needed.
    """
    if not inspect.iscoroutinefunction(func):
        raise TypeError(f'Tool function must be async. Got sync function: {getattr(func, "__name__", repr(func))}')

    tool_name = name if name is not None else getattr(func, '__name__', 'unnamed_tool')
    tool_description = _get_func_description(func, description)

    input_spec = inspect.getfullargspec(func)

    async def tool_fn_wrapper(*args: Any) -> Any:  # noqa: ANN401
        match len(input_spec.args):
            case 0:
                return await func()
            case 1:
                return await func(args[0])
            case 2:
                return await func(args[0], ToolRunContext(cast(ActionRunContext, args[1])))
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
        action.override_input_schema(input_schema)

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
