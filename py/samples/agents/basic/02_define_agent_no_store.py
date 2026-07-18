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

Without a store the agent keeps nothing between sessions — snapshot_id and
session_id stay None. You capture messages + custom state yourself, then hand
them back through chat(messages=..., artifacts=..., state=...) to resume.
Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit_google_genai import GoogleAI

from genkit import Genkit

ai = Genkit(plugins=[GoogleAI()])

agent = ai.define_agent(
    name='echoNoStore',
    model='googleai/gemini-flash-latest',
    system='Echo assistant. Answer briefly and remember context.',
)


async def main() -> None:
    chat = agent.chat()
    turn = chat.send('My name is Ada. Remember it.')

    # Two ways to consume a turn:
    #   await chat.send(msg)            output only, skip the stream
    #   async for chunk in turn: ...       stream chunks, then await turn
    out = await turn
    assert out.text

    # → no server-managed ids — resume by passing the state blob you saved
    assert chat.session_id is None
    assert chat.snapshot_id is None

    # You own the state: capture the conversation (messages + custom state +
    # artifacts) yourself, then hand them straight back to resume.
    messages, state, artifacts = chat.messages, chat.state, chat.artifacts

    resumed = agent.chat(messages=messages, state=state, artifacts=artifacts)
    await resumed.send('What is my name? One word.')


if __name__ == '__main__':
    ai.run_main(main())
