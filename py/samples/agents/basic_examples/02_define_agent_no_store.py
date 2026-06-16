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

"""Backend: define_agent without store — pass out.state into the next AgentInit."""

from __future__ import annotations

from genkit import Genkit
from genkit.agent import AgentInit
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])

agent = ai.define_agent(
    name='echoNoStore',
    model='googleai/gemini-flash-latest',
    system='Echo assistant. Answer briefly and remember context.',
)


async def main() -> None:
    conn = await agent.stream_bidi(AgentInit())
    await conn.send_text('My name is Ada. Remember it.')
    await conn.close()
    async for chunk in conn.receive():
        print('turn1 chunk:', chunk.model_dump(by_alias=True, exclude_none=True))
    out1 = await conn.output()
    print('turn1 state:', out1.state)

    conn2 = await agent.stream_bidi(AgentInit(state=out1.state))
    await conn2.send_text('What is my name? One word.')
    await conn2.close()
    async for chunk in conn2.receive():
        print('turn2 chunk:', chunk.model_dump(by_alias=True, exclude_none=True))
    out2 = await conn2.output()
    print('turn2 output:', out2.model_dump(by_alias=True, exclude_none=True))


if __name__ == '__main__':
    ai.run_main(main())
