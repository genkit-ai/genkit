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

"""Put a deadline on a turn with plain asyncio.

A turn is a normal awaitable/async-iterable, so the native Python tools work on
it: wrap `await turn` (or the stream) in `asyncio.wait_for(...)`, or cancel the
surrounding task, and the turn detaches just like turn.abort() — the client
stops listening, the server finishes in the background, and the prompt you sent
stays in history. The deadline then surfaces as TimeoutError. Requires
GEMINI_API_KEY.
"""

from __future__ import annotations

import asyncio

from genkit_google_genai import GoogleAI

from genkit import Genkit

ai = Genkit(plugins=[GoogleAI()])

agent = ai.define_agent(
    name='essayist',
    model='googleai/gemini-flash-latest',
    system='You are a helpful assistant. When asked to write something long, write many paragraphs.',
)


async def main() -> None:
    chat = agent.chat()
    await chat.send('My name is Ada.')  # turn 1 establishes context the session should keep

    # Give the turn 1.5s to finish; if it overruns, the deadline cancels the await,
    # the turn detaches, and we get a TimeoutError — no genkit-specific cancel API.
    turn = chat.send('Write a very long, multi-paragraph essay about the history of tea.')
    try:
        await asyncio.wait_for(turn, 1.5)
    except asyncio.TimeoutError:
        print('deadline hit — detached from the essay turn')

    # Detach is client-side only: the prompt stays in history (it was still asked),
    # and turn 1's context is intact, so the next turn continues cleanly.
    answer = await chat.send('What is my name? One word.')
    assert 'Ada' in answer.text  # → still remembers turn 1


if __name__ == '__main__':
    ai.run_main(main())
