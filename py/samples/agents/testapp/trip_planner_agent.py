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

"""A trip planner that calls tools for attractions and flights, with a store.

A domain assistant wired from a system prompt plus two mock data tools. It keeps
history in a file store so a planning conversation survives a reload, and streams
its itinerary as it goes. The shape you'd reach for when building any
"assistant over your own data/APIs".

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from _ai import ai
from pydantic import BaseModel

from genkit import ActionRunContext
from genkit.agent import FileSessionStore


class CityInput(BaseModel):
    city: str


class FlightInput(BaseModel):
    from_city: str
    to_city: str


_ATTRACTIONS = {
    'paris': ['Eiffel Tower — iconic iron tower', 'Louvre — world-renowned art museum'],
    'tokyo': ['Senso-ji — ancient Buddhist temple', 'Shibuya Crossing — famous intersection'],
}


@ai.tool(name='getAttractions', description='Get popular tourist attractions for a city.')
async def get_attractions(input: CityInput) -> dict[str, list[str]]:
    key = input.city.lower()
    return {'attractions': _ATTRACTIONS.get(key, [f'{input.city} Central Park', f'{input.city} History Museum'])}


@ai.tool(name='getFlightInfo', description='Get mock flights between two cities.')
async def get_flight_info(input: FlightInput) -> dict[str, list[str]]:
    return {'flights': ['SkyAir 08:00→11:30 $350', 'GlobalJet 14:15→17:45 $420']}


trip_planner_agent = ai.define_agent(
    name='tripPlannerAgent',
    system=(
        'You are a friendly trip planner. Use getAttractions to suggest things to do and '
        'getFlightInfo when the user asks about getting there. Keep it concise and organized.'
    ),
    tools=[get_attractions, get_flight_info],
    store=FileSessionStore('./.snapshots-trip'),
)


@ai.flow()
async def test_trip_planner_agent(text: str, ctx: ActionRunContext) -> str:
    chat = trip_planner_agent.chat()
    turn = chat.send(text or 'I want to plan a trip to Paris. What should I see there?')
    async for chunk in turn:
        for call in chunk.tool_requests:
            ctx.send_chunk(f'[tool] {call.tool_request.name}')
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res = await turn
    return res.text


if __name__ == '__main__':
    import asyncio

    ai.run_main(asyncio.sleep(0))
