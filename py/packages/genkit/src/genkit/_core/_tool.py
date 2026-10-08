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

"""Core tool handle abstraction for the Genkit framework."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Generic, cast

from typing_extensions import TypeVar

from genkit._core._action import NO_INPUT, Action
from genkit._core._typing import ToolDefinition

InputT = TypeVar('InputT', contravariant=True, default=Any)
OutputT = TypeVar('OutputT', covariant=True, default=Any)


class DirectCall:
    """Where the tool wrapper drops the function's own return value for ``await tool(...)``."""

    __slots__ = ('action', 'returned')

    def __init__(self, action: Action) -> None:
        self.action = action
        self.returned: list[object] = []


# Set only while a direct ``await tool(...)`` is running that tool's action, so
# generate and Dev UI runs keep getting the envelope.
direct_call: ContextVar[DirectCall | None] = ContextVar('genkit_tool_direct_call', default=None)


class Tool(Generic[InputT, OutputT]):
    """A registered tool: a callable handle backed by an :class:`~genkit._core._action.Action`."""

    def __init__(
        self,
        action: Action,
        *,
        original_output_schema: dict[str, object] | None = None,
    ) -> None:
        self._action = action
        # What the model should expect as ``output``. ``action.output_schema`` is
        # the envelope ``run`` actually returns (output plus optional media).
        self._original_output_schema = original_output_schema

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
        ``{}``. ``await tool(...)`` returns the function's own value, not
        something shaped by this schema.
        """
        return self._original_output_schema

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

    async def __call__(self, input: InputT | None = NO_INPUT, *, context: dict[str, Any] | None = None) -> OutputT:  # noqa: A002
        """Run the tool and return what the function returned.

        A ``str`` comes back as the string, a Pydantic model as the instance,
        and ``response(...)`` as that ``MultipartToolResponse``. The result is
        still checked the way generate would send it to a model, so a value
        the model couldn't receive raises ``INVALID_ARGUMENT`` here too.
        """
        from genkit._ai._tools import as_multipart_tool_response  # noqa: PLC0415

        call = DirectCall(self._action)
        token = direct_call.set(call)
        try:
            result = (await self._action.run(input, context=context)).response
        finally:
            direct_call.reset(token)
        if call.returned:
            return cast(OutputT, call.returned[0])
        # A Tool built straight from an Action has no wrapper to hand back the
        # raw value; check it the same way and return it as-is.
        as_multipart_tool_response(result, tool_name=self.name)
        return cast(OutputT, result)
