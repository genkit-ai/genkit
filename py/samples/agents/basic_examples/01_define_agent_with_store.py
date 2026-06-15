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

"""Backend: define_agent + store — two invocations, same session_id.

Pattern:
  conn = await agent.stream_bidi(AgentInit(session_id=...))
  await conn.send_text(...)
  await conn.close()
  async for chunk in conn.receive(): ...
  out = await conn.output()
"""

from __future__ import annotations

import asyncio
import random
from uuid import uuid4

from pydantic import BaseModel

from genkit import Genkit
from genkit.agent import AgentInit, InMemorySessionStore
from genkit.plugins.google_genai import GoogleAI


class WeatherInput(BaseModel):
    location: str


class WeatherOutput(BaseModel):
    weather: str
    temperature: str


async def main() -> None:
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

    session_id = str(uuid4())

    conn = await agent.stream_bidi(AgentInit(session_id=session_id))
    await conn.send_text('Weather in Paris?')
    await conn.close()

    async for chunk in conn.receive():
        print('turn1 chunk:', chunk.model_dump(by_alias=True, exclude_none=True))

    out1 = await conn.output()
    print('turn1 output:', out1.model_dump(by_alias=True, exclude_none=True))

    conn2 = await agent.stream_bidi(AgentInit(session_id=session_id))
    await conn2.send_text('What city did I ask about? One word.')
    await conn2.close()

    async for chunk in conn2.receive():
        print('turn2 chunk:', chunk.model_dump(by_alias=True, exclude_none=True))

    out2 = await conn2.output()
    print('turn2 output:', out2.model_dump(by_alias=True, exclude_none=True))


if __name__ == '__main__':
    asyncio.run(main())
