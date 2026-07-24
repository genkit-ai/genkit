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

"""Shape what the client sees with state_transform / chunk_transform.

Two hooks run at the egress boundary; the stored/server-side state is never
touched, only the client's view:

  - state_transform: reshape or redact session state before it leaves. It shapes
    snapshot reads, client-managed output, and the baseline for streamed custom
    patches.
  - chunk_transform: reshape or drop each stream chunk in flight. Return None to
    drop it.

Here the agent keeps an api_key in its custom state and emits an internal
artifact, but the client sees neither: state strips the key, chunk drops the
artifact. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit_google_genai import GoogleAI

from genkit import ActionRunContext, FinishReason, Genkit, Message, Part, TextPart
from genkit.agent import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    AgentStreamChunk,
    Artifact,
    SessionRunner,
    SessionState,
    TurnResult,
)

ai = Genkit(plugins=[GoogleAI()])


async def guarded_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput) -> TurnResult | None:
        # Server-side state carries a secret the client should never see.
        await sess.update_custom(lambda c: {'answers': (c or {}).get('answers', 0) + 1, 'api_key': 'sk-super-secret'})
        # An internal artifact the client shouldn't receive either.
        await sess.add_artifacts(Artifact(name='debug', parts=[Part(TextPart(text='internal trace'))]))

        history = await sess.get_messages()
        messages = [Message(m) for m in history] if history else None
        stream_resp = ai.generate_stream(
            model='googleai/gemini-flash-latest',
            system='Answer in one short sentence.',
            messages=messages,
        )
        async for chunk in stream_resp.stream:
            ctx.send_chunk(AgentStreamChunk(model_chunk=chunk))

        res = await stream_resp.response
        if res.message:
            await sess.add_messages(res.message)

        fr = AgentFinishReason.STOP if res.finish_reason == FinishReason.STOP else AgentFinishReason.UNKNOWN
        return TurnResult(finish_reason=fr)

    await sess.run(handle_turn)
    return await sess.result()


def redact_state(state: SessionState) -> SessionState:
    # Strip the secret; keep everything else. The state hook must return a state
    # (to hide everything you'd return an explicitly cleared one, not None).
    custom = dict(state.custom or {})
    custom.pop('api_key', None)
    return state.model_copy(update={'custom': custom})


def drop_artifacts(chunk: AgentStreamChunk) -> AgentStreamChunk | None:
    # Keep internal artifacts server-side: drop artifact chunks, pass the rest.
    return None if chunk.artifact is not None else chunk


# No store → client-managed, so the (transformed) state ships inline on the output.
agent = ai.define_custom_agent(
    name='guardedAgent',
    fn=guarded_fn,
    state_transform=redact_state,
    chunk_transform=drop_artifacts,
)


async def main() -> None:
    chat = agent.chat()
    turn = chat.send('Say hello.')

    saw_artifact_chunk = False
    async for chunk in turn.stream:
        if chunk.artifact is not None:
            saw_artifact_chunk = True
        if chunk.custom is not None:
            # Streamed custom patches ride on the state hook's output too.
            assert 'api_key' not in chunk.custom

    res = await turn.response
    # chunk hook dropped every artifact chunk before it reached us.
    assert not saw_artifact_chunk
    # state hook stripped the secret but kept the public counter.
    assert res.state is not None
    assert res.state.custom.get('api_key') is None
    assert res.state.custom.get('answers') == 1
    print(f'client sees custom={res.state.custom}, {len(res.artifacts)} artifact(s)')


if __name__ == '__main__':
    ai.run_main(main())
