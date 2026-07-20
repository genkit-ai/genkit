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

"""The hello-world agent: a tool, a store, and streaming multi-turn chat.

``weatherAgent`` keeps its history in a file-backed store, so the server owns the
conversation and a caller only ever needs a session id to pick it back up. The
``test_weather_agent`` flow is what you click Run on in the Dev UI — it drives the
agent exactly the way real code does, through ``agent.chat()``.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import random

from _ai import ai
from pydantic import BaseModel

from genkit import ActionRunContext
from genkit.agent import FileSessionStore


class WeatherInput(BaseModel):
    location: str


class WeatherOutput(BaseModel):
    weather: str
    temperature: str


@ai.tool(name='getWeather', description='Get the current weather for a given location.')
async def get_weather(input: WeatherInput) -> WeatherOutput:
    return WeatherOutput(
        weather=f'{random.choice(["Sunny", "Cloudy", "Rainy"])} in {input.location}',
        temperature=f'{random.randint(5, 34)}°C',
    )


# A store makes this server-managed: history lives on disk, so a client resumes a
# conversation with nothing but its session id (no state round-tripped over the wire).
weather_agent = ai.define_agent(
    name='weatherAgent',
    system='You are an assistant helping with weather information. Use the getWeather tool.',
    tools=[get_weather],
    store=FileSessionStore('./.snapshots'),
)


@ai.flow()
async def test_weather_agent(text: str, ctx: ActionRunContext) -> str:
    """One streamed turn. Tools light up as they're called; text streams as it lands."""
    chat = weather_agent.chat()
    turn = chat.send(text or 'What is the weather like in London?')
    async for chunk in turn:
        for call in chunk.tool_requests:
            ctx.send_chunk(f'[tool] {call.tool_request.name}')
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res = await turn
    return res.text


@ai.flow()
async def test_weather_agent_stream(text: str, ctx: ActionRunContext) -> str:
    """Multi-turn: one chat carries history across turns, so the follow-up just knows."""
    chat = weather_agent.chat()

    turn = chat.send(text or 'What is the weather like in Paris?')
    async for chunk in turn:
        if chunk.text:
            ctx.send_chunk(chunk.text)
    await turn

    followup = chat.send('now say that in French')
    async for chunk in followup:
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res = await followup
    return res.text


if __name__ == '__main__':
    # Lets you run this one agent on its own in the Dev UI:
    #   genkit start -- uv run testapp/weather_agent.py
    import asyncio

    ai.run_main(asyncio.sleep(0))
