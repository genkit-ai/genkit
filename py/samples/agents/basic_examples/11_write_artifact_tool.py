#!/usr/bin/env python3
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

"""Backend: model writes session artifacts via the Artifacts middleware using AgentAPI."""

from __future__ import annotations

from uuid import uuid4

from genkit import Genkit
from genkit.agent import AgentInit, InMemoryLatestStateStore
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.middleware import Artifacts, Middleware

ai = Genkit(plugins=[GoogleAI(), Middleware()])
store = InMemoryLatestStateStore()

agent = ai.define_agent(
    name='workspaceAgent',
    model='googleai/gemini-flash-latest',
    use=[Artifacts()],
    store=store,
)


async def main() -> None:
    session = agent.chat(AgentInit(session_id=str(uuid4())))
    print('--- SENDING TURN ---')
    turn = session.send('Write poem.txt with a short poem about Python agents.')
    async for chunk in turn.stream:
        print('chunk:', chunk)

    await turn.output
    print('session artifacts:', session.artifacts)
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
