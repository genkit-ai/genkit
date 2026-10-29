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

"""Flow decorator classes for type-safe flow registration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, overload

from genkit._core._action import ActionRunContext
from genkit._core._flow import Flow, define_flow
from genkit._core._registry import Registry

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

    # Overload order matters: checkers pick the first matching overload.
    #
    # - A flow with a defaulted input (`async def f(x: str = '...')`) is
    #   callable as both 1-arg and 0-arg, so the 1-arg shapes come first and
    #   the input type stays as written instead of collapsing to `None`.
    # - `(x: Any)` and `(x: object)` also accept a FlowContext, so they'd match
    #   the ctx-only shape and type as Flow[None, ...]. The `object` shapes
    #   come first and catch them as Flow[Any, ...]. A real `(ctx: FlowContext)`
    #   can't take an `object`, so it skips them.
    # - ty follows the typing spec's overload rule for Any: `(x: Any)` matches
    #   overloads with different returns, so ty types it Unknown. That's the
    #   best any arrangement can do there without breaking `(ctx: FlowContext)`.
    @overload
    def __call__(
        self, func: Callable[[ActionRunContext[ChunkT], object], Awaitable[OutputT]]
    ) -> Flow[Any, OutputT, ChunkT]: ...

    @overload
    def __call__(
        self, func: Callable[[InputT, ActionRunContext[ChunkT]], Awaitable[OutputT]]
    ) -> Flow[InputT, OutputT, ChunkT]: ...

    @overload
    def __call__(
        self, func: Callable[[ActionRunContext[ChunkT], InputT], Awaitable[OutputT]]
    ) -> Flow[InputT, OutputT, ChunkT]: ...

    @overload
    def __call__(self, func: Callable[[object], Awaitable[OutputT]]) -> Flow[Any, OutputT]: ...

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
