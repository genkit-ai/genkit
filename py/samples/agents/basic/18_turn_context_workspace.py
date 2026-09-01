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

"""Name external resources after this turn's snapshot id — before the turn ends.

With a store-backed custom agent, each turn reserves its snapshot id up front and
hands it to your handler as TurnContext.snapshot_id. That means you can create a
worktree, scratch directory, or sandbox named after the snapshot *while the turn
is still running*, and the snapshot persisted at turn end reuses that same id.

Without this, the id only appears after save_snapshot mints it — too late to bind
external state to the resume handle your app stores. Here we keep a tiny on-disk
"workspace" per turn under .workspaces/<snapshotId>/, write a file during the
turn, then resume from that snapshot and find the same directory. Requires
GEMINI_API_KEY.
"""

from __future__ import annotations

from pathlib import Path

from genkit_google_genai import GoogleAI

from genkit import ActionRunContext, FinishReason, Genkit, Message
from genkit.agent import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    InMemorySessionStore,
    SessionRunner,
    TurnContext,
    TurnResult,
)

ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()
WORKSPACES = Path(__file__).resolve().parent / '.workspaces'


async def workspace_agent_fn(sess: SessionRunner, _: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput, turn_ctx: TurnContext) -> TurnResult | None:
        # Reserved before this handler ran — same id the store will persist under.
        assert turn_ctx.snapshot_id is not None
        work = WORKSPACES / turn_ctx.snapshot_id
        work.mkdir(parents=True, exist_ok=True)
        note = work / 'turn.txt'
        prompt = ''
        if inp.message and inp.message.content:
            root = inp.message.content[0].root
            prompt = getattr(root, 'text', '') or ''
        note.write_text(f'parent={turn_ctx.parent_snapshot_id}\nprompt={prompt}\n', encoding='utf-8')

        history = await sess.get_messages()
        messages = [Message(m) for m in history] if history else None
        res = await ai.generate(
            model=GoogleAI.gemini_model('gemini-flash-latest'),
            system=(
                'You are a terse assistant. Mention that you wrote notes into a '
                f'workspace directory named after snapshot {turn_ctx.snapshot_id}.'
            ),
            messages=messages,
        )
        if res.message:
            await sess.add_messages([res.message])

        fr = AgentFinishReason.STOP if res.finish_reason == FinishReason.STOP else AgentFinishReason.UNKNOWN
        return TurnResult(finish_reason=fr)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='workspaceAgent', fn=workspace_agent_fn, store=store)


async def main() -> None:
    WORKSPACES.mkdir(exist_ok=True)
    chat = agent.chat()

    # → handler creates .workspaces/<snapshotId>/turn.txt *during* the turn;
    #   the response's snapshot_id matches that directory name.
    res1 = await chat.send('Draft a one-line plan.')
    assert res1.snapshot_id is not None
    workspace = WORKSPACES / res1.snapshot_id
    print('reserved+persisted snapshot:', res1.snapshot_id)
    print('workspace dir:', workspace)
    print('workspace note:\n', (workspace / 'turn.txt').read_text(encoding='utf-8'))

    # Resume from that exact snapshot — external dir is still findable by id.
    resumed = await agent.load_chat(snapshot_id=res1.snapshot_id)
    res2 = await resumed.send('What snapshot workspace did we use?')
    assert res2.snapshot_id is not None
    assert res2.snapshot_id != res1.snapshot_id  # new turn, new reserved id
    print('follow-up snapshot:', res2.snapshot_id)
    print('follow-up workspace:', WORKSPACES / res2.snapshot_id)


if __name__ == '__main__':
    ai.run_main(main())
