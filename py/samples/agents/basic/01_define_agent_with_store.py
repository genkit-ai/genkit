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

With a store, session_id and snapshot_id are minted server-side and arrive when
the first turn completes. To resume the exact conversation state later,
you just need the snapshot_id. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import random

from genkit_google_genai import GoogleAI
from pydantic import BaseModel

from genkit import Genkit
from genkit.agent import InMemorySessionStore


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


async def main() -> None:
    chat = agent.chat()
    turn = chat.send('Weather in Paris?')

    # Two ways to consume a turn:
    #   await chat.send(msg).response          output only, skip the stream
    #   async for chunk in turn.stream: ...    stream chunks, then await turn.response
    async for chunk in turn.stream:
        for call in chunk.tool_requests:
            print(f'  → {call.tool_request.name}')  # tools light up as they're called
        if chunk.text:
            print(chunk.accumulated_text, end='\r', flush=True)

    res = await turn.response
    assert res.text
    print(f'\n{res.text}\n')

    # With a store the server mints these, and they arrive on the settled turn.
    assert res.session_id and res.snapshot_id

    # Hold onto snapshot_id — it's the resume handle after disconnect/restart.
    checkpoint = res.snapshot_id

    # Rehydrate chat directly from that snapshot string.
    resumed = await agent.load_chat(snapshot_id=checkpoint)
    # → answers "Paris" — the resumed session still has turn 1's context
    res2 = await resumed.send('What city did I ask about? One word.').response
    print(f'{res2.text}\n')


if __name__ == '__main__':
    ai.run_main(main())
