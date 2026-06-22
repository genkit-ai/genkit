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
from uuid import uuid4

from pydantic import BaseModel

from genkit import Genkit
from genkit.agent import AgentInit, InMemoryLinearSessionStore
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
    session_id = str(uuid4())

    session = agent.connect(AgentInit(session_id=session_id))
    # Turn 1
    print('--- SENDING TURN 1 ---')
    turn1 = session.send('Weather in Paris?')
    async for chunk in turn1.stream:
        print('turn1 chunk:', chunk)
    out1 = await turn1.output
    print('turn1 output:', out1)

    # Turn 2
    print('--- SENDING TURN 2 ---')
    turn2 = session.send('What city did I ask about? One word.')
    async for chunk in turn2.stream:
        print('turn2 chunk:', chunk)
    out2 = await turn2.output
    print('turn2 output:', out2)
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
