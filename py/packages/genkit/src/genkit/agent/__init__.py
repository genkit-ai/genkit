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

"""Agent types for defining and running bidirectional streaming agents.

Example:
    from genkit import Genkit
    from genkit.agent import InMemorySessionStore
    from genkit_google_genai import GoogleAI

    ai = Genkit(plugins=[GoogleAI()])

    @ai.tool()
    async def current_weather(city: str) -> str:
        return f'Sunny in {city}'

    agent = ai.define_agent(
        name='weatherAgent',
        model=GoogleAI.gemini_model('gemini-flash-latest'),
        system='Weather assistant.',
        tools=[current_weather],
        store=InMemorySessionStore(),
    )
    chat = agent.chat()
    res = await chat.send('Weather in Paris?')
"""

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
from genkit._ai._agents._runtime import AgentFn, AgentInitError, SessionRunner
from genkit._ai._agents._session import (
    Session,
    SessionStore,
    SnapshotStatusStream,
    SnapshotSubscriber,
)
from genkit._ai._agents._session_stores._file_store import FileSessionStore
from genkit._ai._agents._session_stores._inmemory_store import InMemorySessionStore
from genkit._ai._agents._transports._http import HttpAgentTransport, remote_agent
from genkit._ai._agents._types import (
    ChunkTransform,
    StateTransform,
    TurnContext,
    TurnResult,
)
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
    'AgentInitError',
    'AgentInterrupt',
    'AgentResponse',
    'DetachedTask',
    'AgentTransport',
    'HttpAgentTransport',
    'remote_agent',
    # Agent function protocol
    'AgentFn',
    'SessionRunner',
    'TurnContext',
    'TurnResult',
    # Session persistence
    'Session',
    'SessionStore',
    'SnapshotStatusStream',
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
