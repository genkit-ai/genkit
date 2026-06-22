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

"""Backend: define_agent without store — client-managed state using AgentAPI."""

from __future__ import annotations

from genkit import Genkit
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])

agent = ai.define_agent(
    name='echoNoStore',
    model='googleai/gemini-flash-latest',
    system='Echo assistant. Answer briefly and remember context.',
)


async def main() -> None:
    session = agent.connect()
    # Turn 1
    print('--- SENDING TURN 1 ---')
    turn1 = session.send('My name is Ada. Remember it.')
    async for chunk in turn1.stream:
        print('turn1 chunk:', chunk)
    await turn1.output
    print('turn1 state:', session.state)

    # Turn 2
    print('--- SENDING TURN 2 ---')
    turn2 = session.send('What is my name? One word.')
    async for chunk in turn2.stream:
        print('turn2 chunk:', chunk)
    out2 = await turn2.output
    print('turn2 output:', out2)
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
