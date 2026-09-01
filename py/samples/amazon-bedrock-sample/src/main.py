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

"""Same generate() as Gemini, on the AWS profile you already have."""

from genkit_amazon_bedrock import Bedrock, ModelDefinition
from pydantic import BaseModel

from genkit import Genkit

# Declaring a model costs nothing at startup. A missing model-access
# grant only shows up when you call generate().
ai = Genkit(
    plugins=[
        Bedrock(
            models=[ModelDefinition(name='us.amazon.nova-lite-v1:0')],
            embedders=['amazon.titan-embed-text-v2:0'],
        )
    ],
    model='bedrock/us.amazon.nova-lite-v1:0',
)


class CityInput(BaseModel):
    city: str


@ai.tool()
async def current_weather(input: CityInput) -> str:
    return f'The weather in {input.city} is 31C and humid.'


async def main() -> None:
    haiku = await ai.generate(prompt='Write a haiku about coding.')
    print(haiku.text)

    weather = await ai.generate(
        prompt='What is the weather in Lagos? Use the tool, then answer in one sentence.',
        tools=['current_weather'],
    )
    print(weather.text)

    embeddings = await ai.embed(
        embedder='bedrock/amazon.titan-embed-text-v2:0',
        content='Bedrock hosts models from several providers.',
    )
    print(f'dimensions={len(embeddings[0].embedding)}')


if __name__ == '__main__':
    ai.run_main(main())
