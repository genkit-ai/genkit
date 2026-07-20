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

"""Three ways a turn can end early, and what each does to history and the store.

A turn that doesn't land a normal reply can get there three different ways, and
they are genuinely different — especially once you reload the session from the
store, which is the source of truth:

  1. turn.abort() — a *client-side* detach. You stop listening, but the server
     turn keeps running and completes. The store shows a finished turn with no
     trace of the abort, while your in-memory chat is left holding an unanswered
     prompt. The client view drifts ahead of (really, behind) the store.

  2. task.abort() — a *server-side* cancel of a detached turn. The snapshot
     settles ABORTED and never becomes the session's resume point. Your chat is
     left holding the optimistic prompt (drift again), so you reload from the
     store to resync — which skips the dead aborted leaf back to the last
     completed turn.

  3. a real server error (e.g. the model is exhausted) — the turn settles FAILED,
     the optimistic prompt is rolled back, and the resume handle stays pinned to
     the last good turn, so the next send picks up from there.

And a fourth, related point: a detached turn that *succeeds* still never streams
its reply back to your in-memory chat, so before continuing you reload from the
store to resync — otherwise the next turn builds on a view that's missing the
reply.

Uses a no-model custom agent and a linear (parent-retaining) store, so it runs
deterministically with no API key.
"""

from __future__ import annotations

import asyncio

from genkit import ActionRunContext, Genkit, GenkitError, Message, Part, TextPart
from genkit.agent import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    InMemorySessionStore,
    SessionRunner,
    SnapshotStatus,
    TurnResult,
)

ai = Genkit()
store = InMemorySessionStore()


def _text(content: list[Part] | None) -> str:
    return ''.join(
        root.text for p in (content or []) if isinstance((root := getattr(p, 'root', p)), TextPart) and root.text
    )


async def flaky_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput) -> TurnResult | None:
        text = _text(inp.message.content if inp.message else None).lower()
        if 'fail' in text:
            raise GenkitError(status='INTERNAL', message='model exhausted')
        if 'slow' in text:
            await asyncio.sleep(1.0)  # leaves a window to abort while the turn runs
        msgs = await sess.get_messages()
        await sess.set_messages(msgs + [Message(role='model', content=[Part(TextPart(text='reply'))])])
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='flakyAgent', fn=flaky_fn, store=store)


def turns(chat: object) -> list[str]:
    """A compact 'text/role' view of the chat's running history."""
    return [f'{_text(m.content or [])}/{m.role}' for m in chat.messages]  # type: ignore[attr-defined]


async def client_side_abort() -> None:
    """turn.abort(): the server finishes anyway; only the client detaches."""
    chat = agent.chat()
    await chat.send('q1').response
    session_id = chat.session_id

    turn = chat.send('slow q2')
    await asyncio.sleep(0.2)
    await turn.abort()

    # The client stopped listening, so the prompt sits in history unanswered.
    assert turns(chat) == ['q1/user', 'reply/model', 'slow q2/user']

    # The server turn never knew about the abort — give it a beat to finish, then
    # read the session back from the store. The turn completed normally, reply and
    # all: there is no record of an abort anywhere in the durable state.
    await asyncio.sleep(1.3)
    reloaded = await agent.load_chat(session_id=session_id)
    assert turns(reloaded) == ['q1/user', 'reply/model', 'slow q2/user', 'reply/model']


async def server_side_task_abort() -> None:
    """task.abort(): the snapshot settles ABORTED and is not a resume point."""
    chat = agent.chat()
    await chat.send('a1').response
    session_id = chat.session_id

    task = await chat.detach('slow a2')
    # detach optimistically appends the prompt to the local view.
    assert turns(chat) == ['a1/user', 'reply/model', 'slow a2/user']

    status = await task.abort()
    assert status == SnapshotStatus.ABORTED
    # Aborting drops the optimistic 'slow a2' prompt the chat was holding for the
    # killed turn, so the local view rolls back to the last completed turn.
    assert turns(chat) == ['a1/user', 'reply/model']

    # Still reload from the store before continuing — it's the authoritative
    # state, and a detached turn's work never streams back to this chat object,
    # so load_chat is the way to pick up whatever actually landed server-side.
    chat = await agent.load_chat(session_id=session_id)
    assert turns(chat) == ['a1/user', 'reply/model']

    out = await chat.send('a3').response
    assert out.finish_reason == AgentFinishReason.STOP
    assert turns(chat) == ['a1/user', 'reply/model', 'a3/user', 'reply/model']


async def server_side_failure() -> None:
    """A real server error: FAILED, prompt rolled back, resume stays on last good."""
    chat = agent.chat()
    await chat.send('b1').response
    last_good = chat.snapshot_id

    out = await chat.send('please fail').response
    assert out.finish_reason == AgentFinishReason.FAILED
    # No reply landed, so the prompt is dropped and the resume handle holds.
    assert turns(chat) == ['b1/user', 'reply/model']
    assert chat.snapshot_id == last_good

    # The next turn picks up from that last good parent, as if the failure never
    # branched the conversation.
    out2 = await chat.send('b2').response
    assert out2.finish_reason == AgentFinishReason.STOP
    assert turns(chat) == ['b1/user', 'reply/model', 'b2/user', 'reply/model']


async def detached_turn_reload_to_resync() -> None:
    """A detached turn that succeeds: its reply lands in the store, not the chat."""
    chat = agent.chat()
    await chat.send('c1').response
    session_id = chat.session_id

    # Run a turn in the background and let it finish.
    task = await chat.detach('c2')
    snap = await task.wait()
    assert snap.status == SnapshotStatus.COMPLETED

    # The reply streamed server-side, so the in-memory chat only holds the
    # optimistic prompt — it never saw the model's answer.
    assert turns(chat) == ['c1/user', 'reply/model', 'c2/user']

    # Reload from the store to pick up the authoritative history (reply included)
    # before continuing; sending on the stale chat would build on a view missing
    # the detached turn's reply.
    chat = await agent.load_chat(session_id=session_id)
    assert turns(chat) == ['c1/user', 'reply/model', 'c2/user', 'reply/model']

    out = await chat.send('c3').response
    assert out.finish_reason == AgentFinishReason.STOP
    assert turns(chat) == ['c1/user', 'reply/model', 'c2/user', 'reply/model', 'c3/user', 'reply/model']


async def main() -> None:
    await client_side_abort()
    await server_side_task_abort()
    await server_side_failure()
    await detached_turn_reload_to_resync()


if __name__ == '__main__':
    ai.run_main(main())
