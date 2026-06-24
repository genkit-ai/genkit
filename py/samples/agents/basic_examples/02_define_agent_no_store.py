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
from genkit.agent import AgentInit, SessionState
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])

agent = ai.define_agent(
    name='echoNoStore',
    model='googleai/gemini-flash-latest',
    system='Echo assistant. Answer briefly and remember context.',
)


async def main() -> None:
    # --- 1. START A NEW SESSION & RUN TURN 1 ---
    print('--- STARTING A NEW SESSION (CLIENT-MANAGED) ---')
    session = agent.chat()

    print('Sending: My name is Ada. Remember it.')
    out1 = await session.send('My name is Ada. Remember it.').output
    print('Response:', out1.message.content if out1.message else '')

    # Capture the entire session state on the client side!
    # In a client-managed model, you must save the messages and custom state yourself.
    saved_state = SessionState(
        messages=session.messages,
        custom=session.state,
        artifacts=session.artifacts,
    )
    print(f'[Client] Saved {len(session.messages)} messages and custom state.')
    await session.close()

    print('\n======================================================')
    print('   Simulating client disconnect / server restart...   ')
    print('======================================================\n')

    # --- 2. RESUME THE SESSION BY PASSING THE STATE ---
    # We pass the saved state back into agent.chat() to restore the conversation!
    print('--- RESUMING SESSION WITH SAVED STATE ---')
    resumed_session = agent.chat(AgentInit(state=saved_state))

    print('Sending: What is my name? One word.')
    out2 = await resumed_session.send('What is my name? One word.').output
    print('Response:', out2.message.content if out2.message else '')
    await resumed_session.close()


if __name__ == '__main__':
    ai.run_main(main())
