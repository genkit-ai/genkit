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

"""Flow and tool decorator classes for type-safe registration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from genkit._ai._tools import ToolContext, define_tool
from genkit._core._action import ActionRunContext
from genkit._core._flow import Flow, define_flow
from genkit._core._registry import Registry
from genkit._core._tool import Tool

# TypeVars for generic input/output typing
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
ChunkT = TypeVar('ChunkT')


class _FlowDecorator:
    """Decorator class for flow registration with proper type inference."""

    def __init__(self, registry: Registry, name: str | None, description: str | None) -> None:
        self._registry = registry
        self._name = name
        self._description = description

    # Overload order matters: pyright picks the first matching overload, and a
    # flow with a defaulted input arg (`async def f(x: str = '...') -> ...`)
    # is structurally callable as both 1-arg and 0-arg. We list the most
    # specific shapes first so the input type stays as the caller wrote it
    # instead of collapsing to `None`.
    @overload
    def __call__(
        self, func: Callable[[InputT, ActionRunContext[ChunkT]], Awaitable[OutputT]]
    ) -> Flow[InputT, OutputT, ChunkT]: ...

    @overload
    def __call__(
        self, func: Callable[[ActionRunContext[ChunkT], InputT], Awaitable[OutputT]]
    ) -> Flow[InputT, OutputT, ChunkT]: ...

    @overload
    def __call__(
        self, func: Callable[[ActionRunContext[ChunkT]], Awaitable[OutputT]]
    ) -> Flow[None, OutputT, ChunkT]: ...

    @overload
    def __call__(self, func: Callable[[InputT], Awaitable[OutputT]]) -> Flow[InputT, OutputT]: ...

    @overload
    def __call__(self, func: Callable[[], Awaitable[OutputT]]) -> Flow[None, OutputT]: ...

    @overload
    def __call__(self, func: Callable[..., Awaitable[OutputT]]) -> Flow[Any, OutputT, Any]: ...

    def __call__(self, func: Callable[..., Awaitable[Any]]) -> Flow[Any, Any, Any]:
        return define_flow(self._registry, func, self._name, self._description)


class _ToolDecorator:
    """Decorator class for tool registration, typed so ``await tool(...)`` returns the function's own type."""

    def __init__(
        self,
        registry: Registry,
        name: str | None,
        description: str | None,
        input_schema: type[BaseModel] | dict[str, object] | None,
    ) -> None:
        self._registry = registry
        self._name = name
        self._description = description
        self._input_schema = input_schema

    # Same order as define_tool's overloads: context-only before input-only,
    # since a ``ctx`` function also fits ``[InputT]``.
    @overload
    def __call__(
        self, func: Callable[[InputT, ToolContext[ChunkT]], Awaitable[OutputT]]
    ) -> Tool[InputT, OutputT, ChunkT]: ...

    @overload
    def __call__(
        self, func: Callable[[ToolContext[ChunkT], InputT], Awaitable[OutputT]]
    ) -> Tool[InputT, OutputT, ChunkT]: ...

    @overload
    def __call__(self, func: Callable[[ToolContext[ChunkT]], Awaitable[OutputT]]) -> Tool[Any, OutputT, ChunkT]: ...

    @overload
    def __call__(self, func: Callable[[InputT], Awaitable[OutputT]]) -> Tool[InputT, OutputT]: ...

    @overload
    def __call__(self, func: Callable[[], Awaitable[OutputT]]) -> Tool[None, OutputT]: ...

    def __call__(self, func: Callable[..., Awaitable[Any]]) -> Tool[Any, Any, Any]:
        return define_tool(
            self._registry,
            func,
            self._name,
            self._description,
            input_schema=self._input_schema,
        )
