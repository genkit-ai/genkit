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
from typing import Any, Generic, TypeVar, cast, overload

from genkit._core._action import Action, ActionRunContext
from genkit._core._flow import define_flow
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

    @overload
    def __call__(self, func: Callable[[], Awaitable[OutputT]]) -> Action[None, OutputT]: ...

    @overload
    def __call__(self, func: Callable[[InputT], Awaitable[OutputT]]) -> Action[InputT, OutputT]: ...

    @overload
    def __call__(self, func: Callable[[InputT, ActionRunContext], Awaitable[OutputT]]) -> Action[InputT, OutputT]: ...

    def __call__(self, func: Callable[..., Awaitable[Any]]) -> Action[Any, Any]:
        return define_flow(self._registry, func, self._name, self._description)


class _FlowDecoratorWithChunk(Generic[ChunkT]):
    """Decorator class for streaming flow registration with chunk type inference."""

    def __init__(self, registry: Registry, name: str | None, description: str | None, chunk_type: type[ChunkT]) -> None:
        self._registry = registry
        self._name = name
        self._description = description
        self._chunk_type = chunk_type

    @overload
    def __call__(self, func: Callable[[], Awaitable[OutputT]]) -> Action[None, OutputT, ChunkT]: ...

    @overload
    def __call__(self, func: Callable[[InputT], Awaitable[OutputT]]) -> Action[InputT, OutputT, ChunkT]: ...

    @overload
    def __call__(
        self, func: Callable[[InputT, ActionRunContext], Awaitable[OutputT]]
    ) -> Action[InputT, OutputT, ChunkT]: ...

    def __call__(self, func: Callable[..., Awaitable[Any]]) -> Action[Any, Any, ChunkT]:
        # Cast is safe: chunk_type is purely for static typing, runtime behavior is identical
        return cast(Action[Any, Any, ChunkT], define_flow(self._registry, func, self._name, self._description))
