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

"""Same weather agent, but the client owns the conversation — no server store.

Drop the store and state management flips: nothing is kept server-side, so each
turn hands back the whole session blob and the caller passes it straight back on
the next turn. Multi-turn and tool-calling work exactly the same; the only
difference is who holds the history.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from typing import Any

from _ai import ai
from pydantic import BaseModel
from weather_agent import get_weather  # reuse the same tool

from genkit import ActionRunContext

# No store → client-managed. The turn returns the full state; the caller echoes
# it back next time. Nothing about the conversation lives on the server.
weather_agent_stateless = ai.define_agent(
    name='weatherAgentStateless',
    system='You are a helpful weather assistant. Use the getWeather tool to look up weather. Be concise.',
    tools=[get_weather],
)


class StatelessTurn(BaseModel):
    # The state returned by the previous turn; omit on the first call.
    state: Any | None = None
    text: str = 'What is the weather in Tokyo?'


@ai.flow()
async def test_weather_agent_stateless(input: StatelessTurn, ctx: ActionRunContext) -> dict[str, Any]:
    """Resume from the client-held state (or start fresh), then hand it back."""
    chat = weather_agent_stateless.chat(state=input.state) if input.state else weather_agent_stateless.chat()
    turn = chat.send(input.text)
    async for chunk in turn:
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res = await turn
    # The updated blob to round-trip on the next turn — this is the whole session.
    state = res.raw.state.model_dump(by_alias=True, mode='json') if res.raw.state else None
    return {'state': state, 'text': res.text}


if __name__ == '__main__':
    import asyncio

    ai.run_main(asyncio.sleep(0))
