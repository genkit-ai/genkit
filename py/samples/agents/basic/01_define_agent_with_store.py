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

"""The store is the source of truth. Your app holds a snapshot id.

Run a turn, keep ``snapshot_id``, drop the chat. After a reconnect or a process
restart, ``load_chat(snapshot_id=...)`` rehydrates the conversation and the
agent still remembers turn 1.

These samples use ``InMemorySessionStore`` so they run tonight. The same
``store=`` slot takes ``FirestoreSessionStore`` from ``genkit-google-cloud``
when you ship. Requires GEMINI_API_KEY.
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
# In-memory so this file runs without GCP. Swap in FirestoreSessionStore from
# genkit-google-cloud when the session has to survive a deploy.
store = InMemorySessionStore()


@ai.tool(name='getWeather', description='Get weather for a city.')
async def get_weather(input: WeatherInput) -> WeatherOutput:
    return WeatherOutput(
        weather=f'{random.choice(["Sunny", "Cloudy", "Rainy"])} in {input.location}',
        temperature=f'{random.randint(5, 34)}°C',
    )


# Same object you mount with serve_agent(agent) under /api/weatherAgent.
agent = ai.define_agent(
    name='weatherAgent',
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    system='Weather assistant. Use getWeather for weather questions.',
    tools=[get_weather],
    store=store,
)


async def main() -> None:
    chat = agent.chat()
    turn = chat.send_stream('Weather in Paris?')

    # await chat.send(msg) when you only need the final text.
    # send_stream when the UI should paint tokens (then await turn.response).
    async for chunk in turn.stream:
        for call in chunk.tool_requests:
            print(f'  → {call.tool_request.name}')
        if chunk.text:
            print(chunk.accumulated_text, end='\r', flush=True)

    res = await turn.response
    assert res.text
    print(f'\n{res.text}\n')

    # The store mints these; they show up on the settled turn.
    assert res.session_id and res.snapshot_id

    # Persist snapshot_id with the user. That string is the resume handle.
    checkpoint = res.snapshot_id

    resumed = await agent.load_chat(snapshot_id=checkpoint)
    # Follow-up, not a new chat: turn 1 is still in the store.
    res2 = await resumed.send('What city did I ask about? One word.')
    print(f'{res2.text}\n')


if __name__ == '__main__':
    ai.run_main(main())
