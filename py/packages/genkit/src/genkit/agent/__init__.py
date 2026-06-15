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

"""Agent types for defining and running bidirectional streaming agents."""

from genkit._ai._agent import (
    Agent,
    AgentConnection,
    AgentFn,
    ChunkTransform,
    ClientTransform,
    SessionRunner,
    StateTransform,
)
from genkit._ai._session import (
    InMemorySessionStore,
    Session,
    SessionStore,
    SnapshotAborter,
    SnapshotCallback,
    SnapshotContext,
    run_with_session,
    supports_abort,
)
from genkit._core._action import ActionRunContext
from genkit._core._typing import (
    AgentFinishReason,
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentResult,
    AgentStreamChunk,
    Artifact,
    SessionSnapshot,
    SessionState,
    SnapshotEvent,
    SnapshotStatus,
    TurnEnd,
)

__all__ = [
    'Agent',
    'AgentConnection',
    # Agent function protocol
    'ActionRunContext',
    'AgentFn',
    'SessionRunner',
    # Session persistence
    'Session',
    'SessionStore',
    'SnapshotAborter',
    'supports_abort',
    'run_with_session',
    'InMemorySessionStore',
    # Callbacks and transforms
    'SnapshotCallback',
    'SnapshotContext',
    'StateTransform',
    'ChunkTransform',
    'ClientTransform',
    # Wire types
    'AgentFinishReason',
    'AgentInit',
    'AgentInput',
    'AgentOutput',
    'AgentResult',
    'AgentStreamChunk',
    'Artifact',
    'SessionSnapshot',
    'SessionState',
    'SnapshotEvent',
    'SnapshotStatus',
    'TurnEnd',
]
