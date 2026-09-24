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

"""Experimental APIs. Agents stay here even after Genkit Python is GA.

```python
from genkit.exp import Genkit, InMemorySessionStore
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
agent = ai.define_agent(
    name='weatherAgent',
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    system='Weather assistant.',
    store=InMemorySessionStore(),
)
```
"""

from genkit.exp._api import Genkit
from genkit.exp.agent import (
    Agent,
    AgentChat,
    AgentChunk,
    AgentClient,
    AgentError,
    AgentFinishReason,
    AgentFn,
    AgentInit,
    AgentInitError,
    AgentInput,
    AgentInterrupt,
    AgentOutput,
    AgentResponse,
    AgentResult,
    AgentStreamChunk,
    AgentTransport,
    AgentTurn,
    Artifact,
    ChunkTransform,
    DetachedTask,
    FileSessionStore,
    HttpAgentTransport,
    InMemorySessionStore,
    Session,
    SessionRunner,
    SessionSnapshot,
    SessionState,
    SessionStore,
    SnapshotStatus,
    SnapshotStatusStream,
    SnapshotSubscriber,
    StateTransform,
    TurnContext,
    TurnEnd,
    TurnResult,
    remote_agent,
)

__all__ = [
    'Genkit',
    'Agent',
    'AgentClient',
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
    'AgentFn',
    'SessionRunner',
    'TurnContext',
    'TurnResult',
    'Session',
    'SessionStore',
    'SnapshotStatusStream',
    'SnapshotSubscriber',
    'InMemorySessionStore',
    'FileSessionStore',
    'StateTransform',
    'ChunkTransform',
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
