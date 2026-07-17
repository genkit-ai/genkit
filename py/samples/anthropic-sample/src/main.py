# Copyright 2025 Google LLC
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

"""Anthropic's Claude Opus models generation samples."""

from genkit_anthropic import Anthropic
from pydantic import BaseModel, Field

from genkit import ActionRunContext, Genkit, ModelResponse, ReasoningPart

ai = Genkit(plugins=[Anthropic()], model='anthropic/claude-opus-4-8')


class TopicInput(BaseModel):
    """Input for a plain-text generation."""

    topic: str = Field(default='coding', description='Topic for the haiku')


class CatInput(BaseModel):
    """Input for a structured generation."""

    name: str = Field(default='Mittens', description='Name of the cat to invent')


class Cat(BaseModel):
    """Structured cat profile — proves output=['text','json'] + constrained."""

    name: str
    breed: str
    age: int
    personality: str


class WeatherInput(BaseModel):
    """Input for the weather tool and thinking round-trip flow."""

    city: str = Field(default='Reykjavik', description='City to look up')


@ai.tool()
async def current_weather(input: WeatherInput) -> str:
    """Return mocked weather data for tool-calling demos."""
    return f'The weather in {input.city} is 3C, windy, and clear.'


def _thinking_summary(response: ModelResponse) -> dict[str, object]:
    """Summarize reasoning parts across the full generate transcript."""
    reasoning_parts: list[str] = []
    signature_present: list[bool] = []
    content_types: list[str] = []

    for message in response.messages:
        for part in message.content:
            root = part.root
            if root.text is not None:
                content_types.append('text')
            elif root.tool_request is not None:
                content_types.append('tool_request')
            elif root.tool_response is not None:
                content_types.append('tool_response')
            elif isinstance(root, ReasoningPart):
                content_types.append('reasoning')
                reasoning_parts.append(root.reasoning)
                signature_present.append(bool(root.metadata and root.metadata.get('thoughtSignature')))
            elif root.custom is not None:
                content_types.append('custom')
            else:
                content_types.append(type(root).__name__)

    return {
        'content_types': content_types,
        'reasoning_parts': len(reasoning_parts),
        'reasoning_preview': ''.join(reasoning_parts)[:1000],
        'thinking_signatures_present': signature_present,
    }


# --- claude-opus-4-7 -------------------------------------------------------


@ai.flow()
async def haiku_opus_4_7(data: TopicInput) -> str:
    """Plain-text generate against claude-opus-4-7."""
    response = await ai.generate(
        model='anthropic/claude-opus-4-7',
        prompt=f'Write a haiku about {data.topic}.',
    )
    return response.text


@ai.flow()
async def cat_opus_4_7(data: CatInput) -> Cat:
    """Structured/JSON generate against claude-opus-4-7."""
    response = await ai.generate(
        model='anthropic/claude-opus-4-7',
        prompt=f'Invent a cat named {data.name}.',
        output_format='json',
        output_schema=Cat,
    )
    return response.output


# --- claude-opus-4-8 -------------------------------------------------------


@ai.flow()
async def haiku_opus_4_8(data: TopicInput) -> str:
    """Plain-text generate against claude-opus-4-8."""
    response = await ai.generate(
        model='anthropic/claude-opus-4-8',
        prompt=f'Write a haiku about {data.topic}.',
    )
    return response.text


@ai.flow()
async def cat_opus_4_8(data: CatInput) -> Cat:
    """Structured/JSON generate against claude-opus-4-8."""
    response = await ai.generate(
        model='anthropic/claude-opus-4-8',
        prompt=f'Invent a cat named {data.name}.',
        output_format='json',
        output_schema=Cat,
    )
    return response.output


@ai.flow(name='thinking_tool_round_trip')
async def thinking_tool_round_trip(data: WeatherInput, ctx: ActionRunContext) -> dict[str, object]:
    """Dev UI check for Anthropic thinking streaming and signature round-trip."""
    # Opus 4.7+ accept only adaptive thinking; older models use type=enabled with a budget.
    stream_response = ai.generate_stream(
        model='anthropic/claude-opus-4-8',
        prompt=(
            f'You must call the current_weather tool exactly once for {data.city}. '
            'Think through the request before and after the tool call, then answer in one concise sentence.'
        ),
        tools=['current_weather'],
        config={
            'thinking': {'type': 'adaptive', 'display': 'summarized'},
            'max_tokens': 4096,
        },
        max_turns=3,
    )

    streamed_reasoning: list[str] = []
    streamed_text: list[str] = []
    async for chunk in stream_response.stream:
        for part in chunk.content:
            root = part.root
            if isinstance(root, ReasoningPart):
                streamed_reasoning.append(root.reasoning)
                ctx.send_chunk(f'[thinking] {root.reasoning}')
        if chunk.text:
            streamed_text.append(chunk.text)
            ctx.send_chunk(chunk.text)

    response = await stream_response.response
    summary = _thinking_summary(response)
    return {
        **summary,
        'final_text': response.text,
        'streamed_reasoning_chunks': len(streamed_reasoning),
        'streamed_reasoning_preview': ''.join(streamed_reasoning)[:1000],
        'streamed_text': ''.join(streamed_text),
    }


async def main() -> None:
    """Run the lightweight flows once from the CLI."""
    try:
        print(await haiku_opus_4_7(TopicInput()))  # noqa: T201
        print(await cat_opus_4_7(CatInput()))  # noqa: T201
        print(await haiku_opus_4_8(TopicInput()))  # noqa: T201
        print(await cat_opus_4_8(CatInput()))  # noqa: T201
    except Exception as error:
        print(f'Set ANTHROPIC_API_KEY to a valid value before running this sample.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
