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

"""Core tool handle abstraction for the Genkit framework."""

from __future__ import annotations

from typing import Any

from genkit._core._action import Action
from genkit._core._model import MultipartToolResponse
from genkit._core._typing import ToolDefinition


class Tool:
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

    async def __call__(self, *args: Any, **kwargs: Any) -> MultipartToolResponse:  # noqa: ANN401
        """Run the tool and return the envelope (structured output plus optional media)."""
        result = (await self._action.run(*args, **kwargs)).response
        if isinstance(result, MultipartToolResponse):
            return result
        return MultipartToolResponse(output=result)
