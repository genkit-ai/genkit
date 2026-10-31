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

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from genkit._core._action import Action
from genkit._core._model import MultipartToolResponse
from genkit._core._typing import ActionMetadata, ToolDefinition


class Tool:
    """A registered or ephemeral tool: a callable handle backed by an :class:`~genkit._core._action.Action`."""

    _action: Action
    _original_output_schema: dict[str, object] | None

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: type[BaseModel] | dict[str, object] | None = None,
    ) -> None:
        """Define an ephemeral tool for a single ``generate`` call.

        Unlike ``@ai.tool()``, this does not register the tool on the app, so it
        cannot be referenced by name later. Use for dynamic tools that exist only
        for one call.

        Args:
            func: Async tool implementation (same one-input rule as :func:`define_tool`).
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
        from genkit._ai._tools import _define_tool  # noqa: PLC0415
        from genkit._core._registry import Registry  # noqa: PLC0415

        built = _define_tool(Registry(), func, name, description, input_schema=input_schema)
        self._action = built._action
        self._original_output_schema = built._original_output_schema

    @classmethod
    def _from_action(
        cls,
        action: Action,
        *,
        original_output_schema: dict[str, object] | None = None,
    ) -> Tool:
        """Wrap an already-registered tool ``Action`` without re-running ``_define_tool``."""
        instance = cls.__new__(cls)
        instance._action = action
        # What the model should expect as ``output``. ``action.output_schema`` is
        # the envelope ``run`` actually returns (output plus optional media).
        instance._original_output_schema = original_output_schema
        return instance

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

    def _action_metadata(self) -> ActionMetadata:
        """Return reflection metadata for the underlying tool :class:`~genkit._core._action.Action`."""
        return self._action._action_metadata()

    async def __call__(self, *args: Any, **kwargs: Any) -> MultipartToolResponse:  # noqa: ANN401
        """Run the tool and return the envelope (structured output plus optional media)."""
        from genkit._ai._tools import as_multipart_tool_response  # noqa: PLC0415

        result = (await self._action.run(*args, **kwargs)).response
        return as_multipart_tool_response(result, tool_name=self.name)
