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

Without a store there is no snapshot_id to persist. Capture messages, state,
and artifacts yourself, then hand them back through
chat(messages=..., artifacts=..., state=...) to resume.
Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit_google_genai import GoogleAI

from genkit import Genkit

ai = Genkit(plugins=[GoogleAI()])

agent = ai.define_agent(
    name='echoNoStore',
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    system='Echo assistant. Answer briefly and remember context.',
)


async def main() -> None:
    chat = agent.chat()
    turn = chat.send_stream('My name is Ada. Remember it.')

    # Prefer await chat.send(msg) when you don't need chunks. send_stream is for
    # streaming (or abort/timeout handles); awaiting turn.response skips the stream.
    out = await turn.response
    assert out.text

    # No store means no snapshot to reload. Resume by handing back the
    # messages / state / artifacts you kept.
    assert chat.snapshot_id is None

    # You own the state: capture the conversation (messages + custom state +
    # artifacts) yourself, then hand them straight back to resume.
    messages, state, artifacts = chat.messages, chat.state, chat.artifacts

    resumed = agent.chat(messages=messages, state=state, artifacts=artifacts)
    await resumed.send('What is my name? One word.')


if __name__ == '__main__':
    ai.run_main(main())
