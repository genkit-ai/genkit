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

"""Backend: define_agent + store — two turns in a persistent session."""

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
    # --- 1. START A NEW SESSION & RUN TURN 1 ---
    print('--- STARTING A NEW SESSION ---')
    session = agent.chat()

    print('Sending: Weather in Paris?')
    out1 = await session.send('Weather in Paris?').output
    print('Response:', out1.message.content if out1.message else '')

    # Capture the snapshot ID to resume this conversation later!
    saved_snapshot_id = session.snapshot_id
    assert saved_snapshot_id is not None
    print(f'[Client] Saving snapshot ID for later: {saved_snapshot_id}')
    await session.close()

    print('\n======================================================')
    print('   Simulating client disconnect / server restart...   ')
    print('======================================================\n')

    # --- 2. RESUME THE SESSION LATER & RUN TURN 2 ---
    print(f'--- RESUMING SESSION: {saved_snapshot_id} ---')
    resumed_session = await agent.load_chat(saved_snapshot_id)

    print('Sending: What city did I ask about? One word.')
    out2 = await resumed_session.send('What city did I ask about? One word.').output
    print('Response:', out2.message.content if out2.message else '')
    await resumed_session.close()


if __name__ == '__main__':
    ai.run_main(main())
