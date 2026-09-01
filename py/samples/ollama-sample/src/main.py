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

"""Local chat, tools, and embeddings. Same generate() as the cloud plugins."""

import os

from genkit_ollama import EmbeddingDefinition, ModelDefinition, Ollama, OllamaConnectionError
from pydantic import BaseModel

from genkit import Genkit, GenkitError

chat_model = os.getenv('OLLAMA_CHAT_MODEL', 'llama3.2')
embedder_model = os.getenv('OLLAMA_EMBEDDER_MODEL', 'nomic-embed-text')

ai = Genkit(
    plugins=[
        Ollama(
            models=[ModelDefinition(name=chat_model)],
            embedders=[EmbeddingDefinition(name=embedder_model)],
            server_address=os.getenv('OLLAMA_HOST'),
        )
    ],
    model=f'ollama/{chat_model}',
)


class WeatherInput(BaseModel):
    city: str


@ai.tool()
async def current_weather(input: WeatherInput) -> str:
    return f'The weather in {input.city} is 18C and partly cloudy.'


async def main() -> None:
    try:
        response = await ai.generate(prompt='Write a two-sentence pitch for local AI development.')
        print(response.text)

        # Ollama streams text on the chunks and returns an empty final
        # message, so join the chunks instead of reading response.text.
        stream = ai.generate_stream(prompt='One sentence on why local models matter.')
        async for chunk in stream.stream:
            if chunk.text:
                print(chunk.text, end='', flush=True)
        print()
        await stream.response

        weather = await ai.generate(
            prompt='Use current_weather to tell me the weather in London.',
            tools=['current_weather'],
        )
        print(weather.text)

        embeddings = await ai.embed(embedder=f'ollama/{embedder_model}', content='Local models stay on your laptop.')
        print(f'dimensions={len(embeddings[0].embedding)}')
    except GenkitError as error:
        # Genkit wraps provider failures, so unwrap .cause to tell
        # "Ollama is not running" from a real bug.
        if not isinstance(error.cause, OllamaConnectionError):
            raise
        print(
            'Start Ollama and pull the sample models first:\n'
            f'  ollama pull {chat_model}\n'
            f'  ollama pull {embedder_model}\n\n'
            f'{error.cause}'
        )
        raise SystemExit(1) from error


if __name__ == '__main__':
    ai.run_main(main())
