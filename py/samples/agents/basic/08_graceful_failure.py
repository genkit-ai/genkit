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

"""A failing turn fails gracefully instead of crashing the chat.

One turn succeeds; the next raises inside the agent. Rather than bubbling up and
killing the process, the turn resolves with finish_reason FAILED, so your app can
surface the error and keep the session alive. Crucially, the failed turn doesn't
advance the session: it leaves the resume handle pinned to the last successful
snapshot, so the next send picks up from that last good parent — the failure is
a dead end, not a new branch point.
"""

from __future__ import annotations

from genkit_google_genai import GoogleAI

from genkit import ActionRunContext, Genkit, GenkitError, Message, Part, TextPart
from genkit.agent import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    InMemorySessionStore,
    SessionRunner,
    TurnResult,
)

ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()


async def flaky_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput) -> TurnResult | None:
        text = ''
        if inp.message:
            for part in inp.message.content or []:
                root = getattr(part, 'root', part)
                if isinstance(root, TextPart) and root.text:
                    text += root.text
        if 'fail' in text.lower():
            raise GenkitError(status='INTERNAL', message='Simulated turn failure')
        msgs = await sess.get_messages()
        await sess.set_messages(msgs + [Message(role='model', content=[Part(TextPart(text='OK'))])])
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='flakyAgent', fn=flaky_fn, store=store)


async def main() -> None:
    chat = agent.chat()

    # A normal turn succeeds and becomes the session's last good parent.
    out_ok = await chat.send('hello').response
    assert out_ok.finish_reason == AgentFinishReason.STOP
    last_good_parent = chat.snapshot_id

    # This turn raises inside the agent — but the failure is contained.
    out_fail = await chat.send('please fail now').response
    # → resolves as FAILED instead of throwing; the session stays usable
    assert out_fail.finish_reason == AgentFinishReason.FAILED
    # → the failure didn't advance the session: the resume handle is still the
    #   last successful snapshot, so the next turn won't build on the failure.
    assert chat.snapshot_id == last_good_parent

    # The next send picks up from that last good parent, as if the failure never
    # branched the conversation.
    out_ok2 = await chat.send('hello again').response
    assert out_ok2.finish_reason == AgentFinishReason.STOP


if __name__ == '__main__':
    ai.run_main(main())
