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

"""Type definitions and transforms for Genkit agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict, TypeVar

from genkit._core._typing import (
    AgentFinishReason,
    AgentStreamChunk,
    RuntimeError as GenkitRuntimeError,
    SessionState,
)

StateManagement = Literal['server', 'client']

StateT = TypeVar('StateT')
StreamT = TypeVar('StreamT')

# StateTransform — redact or reshape session state before it leaves the server.
StateTransform = Callable[[SessionState], SessionState | None]

# ChunkTransform — reshape or drop a stream chunk before it reaches the client.
ChunkTransform = Callable[[AgentStreamChunk], AgentStreamChunk | None]


class ClientTransform(TypedDict, total=False):
    """Project server-side agent data onto the client-visible view."""

    state: StateTransform
    chunk: ChunkTransform


def resolve_client_transform(
    *,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
) -> ClientTransform | None:
    """``transform`` is shorthand for ``client_transform={'state': transform}``."""
    if client_transform is not None:
        return client_transform
    if transform is not None:
        return {'state': transform}
    return None


@dataclass
class TurnResult:
    """Per-turn execution result returned by an agent loop step."""

    finish_reason: AgentFinishReason | None = None
    error: GenkitRuntimeError | None = None
