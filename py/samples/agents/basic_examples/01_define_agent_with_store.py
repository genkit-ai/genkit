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

Run a turn, save the snapshot id, and drop the session. Later — after a client
reconnect or a server restart — rehydrate the whole conversation from that id and
keep going; the agent still remembers turn 1. The store is the source of truth, so
your app only has to hold onto a string. With a store, session_id is minted
server-side and arrives after the first turn completes. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import random

from pydantic import BaseModel

from genkit import Genkit
from genkit.agent import InMemoryLinearSessionStore
from genkit.plugins.google_genai import GoogleAI


class WeatherInput(BaseModel):
    location: str


class WeatherOutput(BaseModel):
    weather: str
    temperature: str


ai = Genkit(plugins=[GoogleAI()])
store = InMemoryLinearSessionStore()


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
    session = agent.chat()
    turn = session.send('Weather in Paris?')
    # session_id is None until this turn finishes — the store mints it server-side
    await turn.output
    # → session_id and snapshot_id are both set now

    # Hold onto snapshot_id — it's the resume handle after disconnect/restart.
    checkpoint = session.snapshot_id
    assert checkpoint
    await session.close()

    # load_chat restores session_id too — it's already there before you send.
    resumed = await agent.load_chat(checkpoint)
    # → answers "Paris" — the resumed session still has turn 1's context
    await resumed.send('What city did I ask about? One word.')


if __name__ == '__main__':
    ai.run_main(main())
