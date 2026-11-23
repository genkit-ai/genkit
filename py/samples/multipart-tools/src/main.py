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

"""Multipart tools and streaming tool progress chunks."""

import asyncio

from genkit_google_genai import GoogleAI
from pydantic import BaseModel

from genkit import FlowContext, Genkit, MultipartToolResponse, Part, ToolContext

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))

# 1x1 PNG so the tool message has real media without a camera.
_PNG = Part.from_media(
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
    content_type='image/png',
)


class WeatherInput(BaseModel):
    city: str
    unit: str = 'C'


class PrepProgress(BaseModel):
    stage_name: str
    percent: int


# A tool takes one input; several fields go on one model the model fills in.
@ai.tool()
async def weather(input: WeatherInput) -> str:
    """Look up the weather. A bare return is wrapped as output-only."""
    return f'Sunny and 22{input.unit} in {input.city}'


@ai.tool()
async def screenshot(label: str) -> MultipartToolResponse:
    """Take a screenshot. Return ``MultipartToolResponse`` to send the PNG with the output."""
    return MultipartToolResponse(output={'ok': True, 'label': label}, content=[_PNG], metadata={'src': 'lab-cam'})


@ai.tool()
async def prepare_order(dish: str, ctx: ToolContext[PrepProgress]) -> str:
    """Prepare a dish and emit live progress updates before returning."""
    ctx.send_chunk(PrepProgress(stage_name='searing', percent=40))
    await asyncio.sleep(0.2)
    ctx.send_chunk(PrepProgress(stage_name='plating', percent=90))
    await asyncio.sleep(0.2)
    return f'{dish} ready'


@ai.flow()
async def ask_about_the_lab() -> str:
    res = await ai.generate(
        prompt='What is the weather in Austin, and take a screenshot of the lab camera.',
        tools=[weather, screenshot],
    )
    return res.text


@ai.flow()
async def prep_with_progress(dish: str, ctx: FlowContext[PrepProgress]) -> str:
    stream = ai.generate_stream(prompt=f'Prepare {dish}.', tools=[prepare_order])
    async for chunk in stream:
        for part in chunk.tool_responses:
            if part.is_partial and part.tool_response is not None:
                ctx.send_chunk(PrepProgress.model_validate(part.tool_response.output))
    return (await stream.response).text


async def main() -> None:
    print(await ask_about_the_lab())

    stream = prep_with_progress.stream('smoked salmon tartine')
    async for progress in stream:
        print(f'[{progress.percent}%] {progress.stage_name}')
    print(await stream.response)


if __name__ == '__main__':
    ai.run_main(main())
