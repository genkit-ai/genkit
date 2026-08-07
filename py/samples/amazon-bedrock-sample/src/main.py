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

"""Amazon Bedrock samples for the non-streaming Converse path.

Needs AWS credentials and a region (``AWS_REGION`` or ``~/.aws/config``) with
model access granted for the three models below. Streaming is not wired up yet,
so every flow here is a single Converse call.
"""

from genkit_amazon_bedrock import Bedrock, ModelDefinition
from pydantic import BaseModel, Field

from genkit import Genkit, ModelResponse, ReasoningPart

NOVA = 'bedrock/us.amazon.nova-lite-v1:0'
LLAMA = 'bedrock/us.meta.llama3-3-70b-instruct-v1:0'
DEEPSEEK = 'bedrock/us.deepseek.r1-v1:0'
CLAUDE = 'bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0'

# Declaring models is optional — resolve() serves any Bedrock model ID or ARN
# on demand — but declared ones show up in the Dev UI model list.
ai = Genkit(
    plugins=[
        Bedrock(
            models=[
                ModelDefinition(name='us.amazon.nova-lite-v1:0'),
                ModelDefinition(name='us.meta.llama3-3-70b-instruct-v1:0'),
                ModelDefinition(name='us.deepseek.r1-v1:0'),
                ModelDefinition(name='us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
            ]
        )
    ],
    model=NOVA,
)


class TopicInput(BaseModel):
    """Input for a plain-text generation."""

    topic: str = Field(default='coding', description='Topic for the haiku')


class CatInput(BaseModel):
    """Input for a structured generation."""

    name: str = Field(default='Mittens', description='Name of the cat to invent')


class Cat(BaseModel):
    """Structured cat profile."""

    name: str
    breed: str
    age: int
    personality: str


class CityInput(BaseModel):
    """Input for the weather tool."""

    city: str = Field(default='Lagos', description='City to look up')


@ai.tool()
async def current_weather(city_input: CityInput) -> str:
    """Return mocked weather data for tool-calling demos."""
    return f'The weather in {city_input.city} is 31C and humid.'


@ai.flow()
async def haiku(data: TopicInput) -> str:
    """Plain-text generate through Converse."""
    response = await ai.generate(prompt=f'Write a haiku about {data.topic}.')
    return response.text


@ai.flow()
async def cat_profile(data: CatInput) -> Cat:
    """Structured output, carried by prompt instructions.

    Bedrock has no constrained-decoding mode, and the core's json format only
    injects the schema when ``output_instructions`` is set, so it is required
    here. Model choice matters too: the Nova models answer in prose often
    enough to fail extraction, so this uses Llama 3.3.
    """
    response = await ai.generate(
        model=LLAMA,
        prompt=f'Invent a cat named {data.name}.',
        output_format='json',
        output_schema=Cat,
        output_instructions=True,
        config={'maxTokens': 1024},
    )
    return response.output


@ai.flow()
async def weather_report(data: CityInput) -> str:
    """Tool calling: the model calls the tool, then answers from its output."""
    response = await ai.generate(
        prompt=f'What is the weather in {data.city}? Use the tool, then answer in one sentence.',
        tools=['current_weather'],
    )
    return response.text


@ai.flow()
async def reasoning(data: TopicInput) -> dict[str, object]:
    """Reasoning parts parsed off a Converse response.

    DeepSeek R1 reasons on every turn, so no thinking config is needed. Its
    reasoning carries no signature, which is why ``signatures_present`` is
    false here: signatures are Anthropic-specific and gate replay.
    """
    response = await ai.generate(
        model=DEEPSEEK,
        prompt=f'What is 17 * 23? Think it through, then state the answer. Mention {data.topic} once.',
        config={'maxTokens': 2048},
    )
    return _reasoning_summary(response)


@ai.flow()
async def thinking(data: TopicInput) -> dict[str, object]:
    """Claude extended thinking: signed reasoning that survives replay.

    Unlike DeepSeek, Claude signs its reasoning, so ``signatures_present`` is
    true here and the parts are replayed verbatim on multi-turn follow-ups.
    """
    response = await ai.generate(
        model=CLAUDE,
        prompt=f'What is 17 * 23? Think it through, then state the answer. Mention {data.topic} once.',
        config={
            'maxTokens': 4096,
            # Bedrock requires budget_tokens >= 1024, below maxTokens.
            'additionalModelRequestFields': {'thinking': {'type': 'enabled', 'budget_tokens': 1024}},
        },
    )
    return _reasoning_summary(response)


def _reasoning_summary(response: ModelResponse) -> dict[str, object]:
    """Summarize the reasoning parts on a response."""
    reasoning_text: list[str] = []
    signed: list[bool] = []
    for message in response.messages:
        for part in message.content:
            root = part.root
            if isinstance(root, ReasoningPart):
                reasoning_text.append(root.reasoning)
                signed.append(bool(root.metadata and root.metadata.get('bedrockReasoningSignature')))
    return {
        'answer': response.text,
        'reasoning_parts': len(reasoning_text),
        'reasoning_preview': ''.join(reasoning_text)[:500],
        'signatures_present': signed,
    }


async def main() -> None:
    """Run the lightweight flows once from the CLI."""
    try:
        print(await haiku(TopicInput()))  # noqa: T201
        print(await weather_report(CityInput()))  # noqa: T201
    except Exception as error:
        # Printed, not raised: in dev mode the Dev UI stays up either way.
        print(  # noqa: T201
            f'Set AWS credentials and a region, and grant model access for {NOVA}, {LLAMA}, and {DEEPSEEK}, '
            f'before running this sample.\n{error}'
        )


if __name__ == '__main__':
    ai.run_main(main())
