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

"""No store: you own the conversation state on the client.

Without a store the agent keeps nothing between sessions — snapshot_id stays None.
You capture messages + custom state yourself, then hand them back through AgentInit
to resume. session_id is minted client-side on the first send(), so you own the
identity too. Requires GEMINI_API_KEY.
"""

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
    session = agent.chat()
    turn = session.send('My name is Ada. Remember it.')

    # Three ways to consume a turn — same as store-backed agents:
    #   await session.send(msg).output     output only, skip the stream
    #   async for chunk in turn: ...       stream chunks to a UI
    #   stream first, then await turn.output   both on the same turn
    out = await turn.output
    assert out.text

    # → session_id minted client-side on first send — you own the identity
    # → snapshot_id stays None — no server-managed checkpoint to resume from
    assert session.session_id
    assert session.snapshot_id is None

    saved = SessionState(
        session_id=session.session_id,
        messages=session.messages,
        custom=session.state,
        artifacts=session.artifacts,
    )
    await session.close()

    resumed = agent.chat(AgentInit(state=saved))
    await resumed.send('What is my name? One word.')


if __name__ == '__main__':
    ai.run_main(main())
