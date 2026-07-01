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

"""Persist a session, then resume it later from just a snapshot id.

Run a turn, save the snapshot id, and drop the chat. Later — after a client
reconnect or a server restart — rehydrate the whole conversation from that id and
keep going; the agent still remembers turn 1. The store is the source of truth, so
your app only has to hold onto a string.

The same ``render`` view drives the fresh chat and the rehydrated one: a UI only
ever needs the chat handle, because everything it shows rides on the streamed
chunks and the settled response. With a store, session_id is minted server-side
and arrives when the first turn completes. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import random
from typing import TypeVar

from pydantic import BaseModel

from genkit import Genkit
from genkit.agent import AgentChat, AgentResponse, InMemorySessionStore
from genkit.plugins.google_genai import GoogleAI

StateT = TypeVar('StateT')


class WeatherInput(BaseModel):
    location: str


class WeatherOutput(BaseModel):
    weather: str
    temperature: str


ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()


@ai.tool(name='getWeather', description='Get weather for a city.')
async def get_weather(input: WeatherInput) -> WeatherOutput:
    return WeatherOutput(
        weather=f'{random.choice(["Sunny", "Cloudy", "Rainy"])} in {input.location}',
        temperature=f'{random.randint(5, 34)}°C',
    )


agent = ai.define_agent(
    name='weatherAgent',
    model='googleai/gemini-flash-latest',
    system='Weather assistant. Use getWeather for weather questions.',
    tools=[get_weather],
    store=store,
)


async def render(chat: AgentChat[StateT], prompt: str) -> AgentResponse[StateT]:
    """Render one turn the way a UI would — handed nothing but the chat handle.

    Generic over the chat's state, so it's the one view a dumb component needs:
    pass any agent's handle and the turn's typed state flows straight through to
    the AgentResponse. A view binds to two things, both reachable from the handle:
    the streaming chunk (accumulated_text is the reply so far; tool_requests are
    the calls in flight) and the settled response (text, finish_reason,
    snapshot_id, media, data). Identical whether the chat is new or rehydrated.
    """
    turn = chat.send(prompt)
    async for chunk in turn:
        for call in chunk.tool_requests:
            print(f'  → {call.tool_request.name}')  # tools light up as they're called
        if chunk.text:
            print(chunk.accumulated_text, end='\r', flush=True)  # the frame a UI re-renders
    res = await turn
    res.assert_valid()  # raises if the turn was blocked or produced no message
    print(f'\n{res.text}\n')
    return res


async def main() -> None:
    chat = agent.chat()
    res = await render(chat, 'Weather in Paris?')
    # With a store the server mints these, and they arrive on the settled turn.
    assert res.session_id and res.snapshot_id

    # Hold onto snapshot_id — it's the resume handle after disconnect/restart.
    checkpoint = res.snapshot_id
    await chat.close()

    # Same view, now driving a chat rehydrated from just that one string.
    resumed = await agent.load_chat(snapshot_id=checkpoint)
    # → answers "Paris" — the resumed session still has turn 1's context
    await render(resumed, 'What city did I ask about? One word.')


if __name__ == '__main__':
    ai.run_main(main())
