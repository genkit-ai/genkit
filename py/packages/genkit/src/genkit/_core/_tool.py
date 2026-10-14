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

"""Core tool action for the Genkit framework."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypeVar

from genkit._core._action import Action, ActionKind, ActionRunContext, get_func_description
from genkit._core._model import MultipartToolResponse
from genkit._core._typing import MultipartToolResponse as MultipartToolResponseData, ToolDefinition

InputT = TypeVar('InputT', default=Any)
OutputT = TypeVar('OutputT', default=Any)
# What ``ctx.send_chunk`` streams. Untyped on this branch, so ``Any``.
ChunkT = TypeVar('ChunkT', default=Any)

# Metadata key for the schema the model binds as the tool's ``outputSchema``.
# For ``-> MultipartToolResponse[T]`` that's ``T``; for ``-> T`` it's ``T``.
ORIGINAL_OUTPUT_SCHEMA_KEY = 'originalOutputSchema'


class Tool(Action[InputT, OutputT, ChunkT]):
    """A tool the model can call: an :class:`~genkit._core._action.Action` of kind ``tool``.

    ``await tool(x)`` is ``(await tool.run(x)).response``: what the function
    returned. A ``str`` comes back as the string, a Pydantic model as the
    instance, and ``response(...)`` as that ``MultipartToolResponse``.

    Each tool also carries a ``tool.v2`` action that returns the
    ``MultipartToolResponse`` envelope. The registry stores both, like JS and
    Go; the Dev UI lists and runs the ``tool.v2`` one.
    """

    _wire_action: Action[InputT, MultipartToolResponse[Any], ChunkT] | None

    def __init__(
        self,
        func: Callable[..., Awaitable[OutputT]],
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
        """
        # _ai/_tools imports _core; the run-context and envelope helpers live there.
        from genkit._ai._tools import (  # noqa: PLC0415
            as_multipart_tool_response,
            call_tool_fn,
            model_schema_from_return_annotation,
        )

        if not inspect.iscoroutinefunction(func):
            raise TypeError(f'Tool function must be async. Got sync function: {getattr(func, "__name__", repr(func))}')
        tool_name = name if name is not None else getattr(func, '__name__', None)
        if tool_name is None:
            raise ValueError(f'Cannot infer a tool name from {func!r}; pass name= explicitly.')
        tool_description = get_func_description(func, description)

        async def tool_fn(input: object, ctx: ActionRunContext) -> object:  # noqa: A002
            return await call_tool_fn(self, func, input, ctx)

        super().__init__(
            kind=ActionKind.TOOL,
            name=tool_name,
            fn=cast(Callable[..., Awaitable[OutputT]], tool_fn),
            metadata_fn=func,
            description=tool_description,
        )
        if input_schema is not None:
            self._override_input_schema(input_schema)
        model_schema = model_schema_from_return_annotation(func, inferred=self.output_schema)
        self.metadata[ORIGINAL_OUTPUT_SCHEMA_KEY] = model_schema

        async def wire_fn(input: object, ctx: ActionRunContext) -> MultipartToolResponse[Any]:  # noqa: A002
            raw = await call_tool_fn(self, func, input, ctx)
            return as_multipart_tool_response(raw, tool_name=self.name)

        wire: Action[InputT, MultipartToolResponse[Any], ChunkT] = Action(
            kind=ActionKind.TOOL_V2,
            name=tool_name,
            fn=wire_fn,
            metadata_fn=func,
            description=tool_description,
        )
        if input_schema is not None:
            wire._override_input_schema(input_schema)
        wire.output_schema = TypeAdapter(MultipartToolResponseData).json_schema()
        wire.metadata[ORIGINAL_OUTPUT_SCHEMA_KEY] = model_schema
        self._wire_action = wire

    @classmethod
    def _from_action(cls, action: Action) -> Tool[Any, Any, Any]:
        """View a tool ``Action`` that wasn't built as a ``Tool`` (plugin, DAP, hand-registered) as one.

        Shares the action's function and metadata; there is no ``tool.v2`` companion.
        """
        tool = cls.__new__(cls)
        tool.__dict__.update(action.__dict__)
        tool._wire_action = None
        return tool

    @property
    def wire_action(self) -> Action[InputT, MultipartToolResponse[Any], ChunkT] | None:
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
        if ORIGINAL_OUTPUT_SCHEMA_KEY in self.metadata:
            original = self.metadata[ORIGINAL_OUTPUT_SCHEMA_KEY]
            output_schema = cast(dict[str, object], original) if isinstance(original, dict) else None
        else:
            output_schema = self.output_schema
        return ToolDefinition(
            name=self.name,
            description=self.description or '',
            input_schema=self.input_schema,
            output_schema=output_schema,
        )

    async def _invoke(self, input: object | None, ctx: ActionRunContext) -> OutputT:  # noqa: A002
        """Run the function, then check its result the way generate would send it to a model.

        A value the model couldn't receive (unserializable output or metadata,
        a part with no payload) raises ``INVALID_ARGUMENT`` on every call path,
        not just inside ``generate``. The function's own value is returned.
        """
        from genkit._ai._tools import as_multipart_tool_response  # noqa: PLC0415

        output = await super()._invoke(input, ctx)
        as_multipart_tool_response(output, tool_name=self.name)
        return output
