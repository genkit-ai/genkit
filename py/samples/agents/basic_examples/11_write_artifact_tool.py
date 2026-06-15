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

"""Backend: model writes session artifacts via the Artifacts middleware.

``define_agent`` with ``use=[Artifacts()]`` injects ``read_artifact`` / ``write_artifact``
and refreshes an ``<artifacts>`` listing in the system prompt each generate. No
custom system prompt or session closure is required.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from genkit import Genkit
from genkit.agent import AgentInit, InMemorySessionStore
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.middleware import Artifacts, Middleware


async def main() -> None:
    ai = Genkit(plugins=[GoogleAI(), Middleware()])
    store = InMemorySessionStore()

    agent = ai.define_agent(
        name='workspaceAgent',
        model='googleai/gemini-flash-latest',
        use=[Artifacts()],
        store=store,
    )

    conn = await agent.stream_bidi(AgentInit(session_id=str(uuid4())))
    await conn.send_text('Write poem.txt with a short poem about Python agents.')
    await conn.close()

    async for chunk in conn.receive():
        print('chunk:', chunk.model_dump(by_alias=True, exclude_none=True))

    out = await conn.output()
    print('output artifacts:', [a.model_dump(by_alias=True) for a in (out.artifacts or [])])


if __name__ == '__main__':
    asyncio.run(main())
