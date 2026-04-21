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

"""Flow registration for Genkit."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, overload

from typing_extensions import TypeVar

from genkit._core._action import Action, ActionKind, ActionRunContext, get_func_description
from genkit._core._registry import Registry

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


@overload
def define_flow(
    registry: Registry,
    func: Callable[[], Awaitable[OutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Action[None, OutputT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[InputT], Awaitable[OutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Action[InputT, OutputT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[InputT, ActionRunContext], Awaitable[OutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Action[InputT, OutputT]: ...


def define_flow(
    registry: Registry,
    func: Callable[..., Awaitable[Any]],
    name: str | None = None,
    description: str | None = None,
) -> Action[Any, Any]:
    """Register an async function as a flow action."""
    # All Python functions have __name__, but ty is strict about Callable protocol
    if not inspect.iscoroutinefunction(func):
        raise TypeError(f'Flow must be async: {getattr(func, "__name__", repr(func))}')

    flow_name = name or getattr(func, '__name__', None) or 'unnamed_flow'
    return registry.register_action(
        name=flow_name,
        kind=ActionKind.FLOW,
        fn=func,
        description=get_func_description(func, description),
        span_metadata={'genkit:metadata:flow:name': flow_name},
    )
