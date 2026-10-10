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

One turn succeeds; the next raises inside the agent. The chat returns that
turn with finish_reason failed and the why on res.error, and the session
stays usable: the failed turn doesn't advance the resume handle — it stays
pinned to the last successful snapshot — so the next send picks up from
that last good parent. The failure is a dead end, not a new branch point.

Because a call that fails before a turn result is produced (such as a
dropped connection or an init the server rejects as misuse) still raises
AgentError, production code still wraps chat.send() in try/except AgentError
for transport drops, while inspecting res.error for in-turn failures.
"""

from __future__ import annotations

from genkit_google_genai import GoogleAI

from genkit import ActionRunContext, GenkitError, Message, Part
from genkit.exp import Genkit
from genkit.exp.agent import (
    AgentError,
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


async def flaky_fn(sess: SessionRunner, _: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput, _: TurnContext) -> TurnResult | None:
        text = ''
        if inp.message:
            for part in inp.message.content or []:
                if part.text:
                    text += part.text
        if 'fail' in text.lower():
            raise GenkitError(status='INTERNAL', message='Simulated turn failure')
        msgs = await sess.get_messages()
        await sess.set_messages(msgs + [Message(role='model', content=[Part.from_text('OK')])])
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='flakyAgent', fn=flaky_fn, store=store)


async def main() -> None:
    chat = agent.chat()

    # 1. A normal turn succeeds and becomes the session's last good parent.
    out_ok = await chat.send('hello')
    assert out_ok.finish_reason == AgentFinishReason.STOP
    last_good_parent = chat.snapshot_id

    # 2. Turn execution failure: the turn runs, but raises inside the agent.
    # The chat returns it with res.error set. A try/except AgentError still
    # wraps the call to catch transport drops or pre-turn rejects.
    try:
        res = await chat.send('please fail now')
        if res.error:
            assert res.finish_reason == AgentFinishReason.FAILED
            assert 'Simulated turn failure' in res.error.message
            # The failure didn't advance the session: the resume handle is still the
            # last successful snapshot, so the next turn won't build on the failure.
            assert res.snapshot_id == last_good_parent
            assert chat.snapshot_id == last_good_parent
    except AgentError as err:
        raise AssertionError('turn ran, expected res.error rather than AgentError') from err

    # 3. The next send picks up from that last good parent, as if the failure never
    # branched the conversation.
    out_ok2 = await chat.send('hello again')
    assert out_ok2.finish_reason == AgentFinishReason.STOP

    # 4. Pre-turn failure: a call that never produces a turn result (such as
    # passing a snapshot_id to an agent with no store) still raises AgentError.
    client_agent = ai.define_custom_agent(name='clientAgent', fn=flaky_fn, store=None)
    bad_chat = client_agent.chat(snapshot_id='unsupported')
    try:
        await bad_chat.send('hi')
        raise AssertionError('expected AgentError for pre-turn misuse')
    except AgentError as err:
        assert err.status == 'FAILED_PRECONDITION'


if __name__ == '__main__':
    ai.run_main(main())
