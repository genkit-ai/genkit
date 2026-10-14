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

"""Flow action class and registration for Genkit."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, overload

from typing_extensions import TypeVar

from genkit._core._action import Action, ActionKind, ActionRunContext, get_func_description
from genkit._core._registry import Registry

InputT = TypeVar('InputT', default=Any)
OutputT = TypeVar('OutputT', default=Any)
ChunkT = TypeVar('ChunkT', default=Any)

_FlowInputT = TypeVar('_FlowInputT')
_FlowOutputT = TypeVar('_FlowOutputT')
_FlowChunkT = TypeVar('_FlowChunkT')


class Flow(Action[InputT, OutputT, ChunkT, None]):
    """A typed, traced, streamable flow action.

    ``@ai.flow`` constructs and registers a ``Flow`` on the ``Genkit`` instance.
    Constructing ``Flow(fn)`` directly builds a standalone (unregistered) flow.
    """

    @overload
    def __init__(
        self,
        func: Callable[[InputT, ActionRunContext[ChunkT]], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[ActionRunContext[ChunkT], InputT], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[ActionRunContext[ChunkT]], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[InputT], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[..., Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize a ``Flow`` from ``func``.

        Args:
            func: Async flow handler taking at most one input and an optional
                ``FlowContext[ChunkT]`` parameter.
            name: Optional flow name. Defaults to ``func.__name__``.
            description: Optional description. Defaults to ``func.__doc__``.
        """
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f'Flow must be async: {getattr(func, "__name__", repr(func))}')

        flow_name = name or getattr(func, '__name__', None) or 'unnamed_flow'
        super().__init__(
            kind=ActionKind.FLOW,
            name=flow_name,
            fn=func,
            description=get_func_description(func, description),
            span_metadata={'flow:name': flow_name},
            _strict_io=True,
        )


@overload
def define_flow(
    registry: Registry,
    func: Callable[[_FlowInputT, ActionRunContext[_FlowChunkT]], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[_FlowInputT, _FlowOutputT, _FlowChunkT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[ActionRunContext[_FlowChunkT], _FlowInputT], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[_FlowInputT, _FlowOutputT, _FlowChunkT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[ActionRunContext[_FlowChunkT]], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[None, _FlowOutputT, _FlowChunkT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[_FlowInputT], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[_FlowInputT, _FlowOutputT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[None, _FlowOutputT]: ...


def define_flow(
    registry: Registry,
    func: Callable[..., Awaitable[Any]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[Any, Any, Any]:
    """Register an async function as a flow action."""
    flow = Flow(func, name=name, description=description)
    registry.register_action_from_instance(flow)
    return flow
