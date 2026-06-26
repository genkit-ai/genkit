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

"""Two ways to stop an agent: turn.abort() (client) vs session.abort() (server).

turn.abort() is a client-side detach — you stop reading the stream and the turn
settles immediately, while the server turn keeps running to completion. The
optimistic user message is rolled back, so the conversation just continues from the
last good turn; it works with or without a store. session.abort() is the opposite:
it halts the work on the server, firing the abort_signal inside running tools so a
detached turn settles ABORTED. It needs a store. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import asyncio

from genkit import Genkit, GenkitError, ToolRunContext
from genkit.agent import InMemoryLatestStateStore, SnapshotStatus
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemoryLatestStateStore()

# No store: state lives on the client, so turn.abort()'s rollback is fully local.
chatty = ai.define_agent(
    name='chattyAgent',
    model='googleai/gemini-flash-latest',
    system='You are a helpful assistant. When asked to write something long, write many paragraphs.',
)


@ai.tool(name='slowWork', description='Simulate long background work.')
async def slow_work(_: dict, ctx: ToolRunContext) -> dict:
    for _i in range(30):
        if ctx.abort_signal.is_set():  # session.abort() reaches the tool here so it can bail out
            raise GenkitError(status='ABORTED', message='Task aborted')
        await asyncio.sleep(0.5)
    return {'done': True}


# Store-backed: session.abort() cancels a server-side snapshot, so it needs a store.
worker = ai.define_agent(
    name='workerAgent',
    model='googleai/gemini-flash-latest',
    system='When asked for a long task, call slowWork.',
    tools=[slow_work],
    store=store,
)


async def main() -> None:
    # --- turn.abort(): client-side stop button ---
    chat = chatty.chat()
    await chat.send('My name is Ada.')  # turn 1 establishes context the session should keep
    history_len = len(chat.messages)

    # Ask for something long, then bail out partway like a user hitting "stop".
    turn = chat.send('Write a very long, multi-paragraph essay about the history of tea.')
    seen = 0
    async for chunk in turn:
        seen += len(chunk.text or '')
        if seen > 200:
            await turn.abort()  # detach now; the server finishes the essay in the background, then discards it
            break

    # The aborted turn's prompt is rolled back, so the session reads as if it never happened.
    assert len(chat.messages) == history_len
    answer = await chat.send('What is my name? One word.')
    assert 'Ada' in answer.text  # → continues cleanly from turn 1

    # --- session.abort(): server-side cancel of background work ---
    session = worker.chat()
    task = await session.detach('Please run a long task using slowWork.')
    assert task.snapshot_id
    await asyncio.sleep(2.0)  # let the tool start churning

    status = await session.abort()  # abort_signal fires inside slowWork; the snapshot settles ABORTED
    assert status == SnapshotStatus.ABORTED
    await asyncio.sleep(0.5)  # let the background task unwind

    # The stop is durable: read the snapshot back and it stays ABORTED.
    snap = await session.get_snapshot()
    assert snap and snap.status == SnapshotStatus.ABORTED
    # The snapshot holds history through this turn's user prompt but none of its
    # model output: a turn's model/tool messages are committed to the session in
    # one batch at turn end, and abort interrupts before that. So an aborted
    # snapshot is a clean pre-response checkpoint you can branch from.

    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
