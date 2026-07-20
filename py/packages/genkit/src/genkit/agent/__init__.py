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

from genkit._ai._agents._base import Agent
from genkit._ai._agents._client import (
    AgentChat,
    AgentChunk,
    AgentClient,
    AgentError,
    AgentInterrupt,
    AgentResponse,
    AgentTransport,
    AgentTurn,
    DetachedTask,
)
from genkit._ai._agents._runtime import AgentFn, SessionRunner
from genkit._ai._agents._session import (
    Session,
    SessionStore,
    SnapshotSubscriber,
)
from genkit._ai._agents._transports._http import HttpAgentTransport, remote_agent
from genkit._ai._agents._types import (
    ChunkTransform,
    StateTransform,
    TurnResult,
)
from genkit._ai._agents.session_stores.file_store import FileSessionStore
from genkit._ai._agents.session_stores.inmemory_store import InMemorySessionStore
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
    SnapshotStatus,
    TurnEnd,
)

__all__ = [
    # Agent handles
    'Agent',
    'AgentClient',
    # Agent Client APIs
    'AgentChat',
    'AgentTurn',
    'AgentChunk',
    'AgentError',
    'AgentInterrupt',
    'AgentResponse',
    'DetachedTask',
    'AgentTransport',
    'HttpAgentTransport',
    'remote_agent',
    # Agent function protocol
    'AgentFn',
    'SessionRunner',
    'TurnResult',
    # Session persistence
    'Session',
    'SessionStore',
    'SnapshotSubscriber',
    'InMemorySessionStore',
    'FileSessionStore',
    # Callbacks and transforms
    'StateTransform',
    'ChunkTransform',
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
    'SnapshotStatus',
    'TurnEnd',
]
