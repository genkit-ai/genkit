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

"""Ollama sample for local chat, streaming, tools, and embeddings.

Run the default exercise once:

    uv run src/main.py

Or open the Dev UI and pick a flow:

    genkit start -- uv run src/main.py
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from genkit import Genkit, GenkitError
from genkit.plugins.ollama import (  # pyright: ignore[reportMissingImports]
    EmbeddingDefinition,
    ModelDefinition,
    Ollama,
    OllamaConnectionError,
)

CHAT_MODEL = os.getenv('OLLAMA_CHAT_MODEL', 'llama3.2')
EMBEDDER_MODEL = os.getenv('OLLAMA_EMBEDDER_MODEL', 'nomic-embed-text')


ai = Genkit(
    plugins=[
        Ollama(
            models=[ModelDefinition(name=CHAT_MODEL)],
            embedders=[EmbeddingDefinition(name=EMBEDDER_MODEL)],
            server_address=os.getenv('OLLAMA_HOST'),
        )
    ],
    model=f'ollama/{CHAT_MODEL}',
)


class PromptInput(BaseModel):
    """Prompt input for chat examples."""

    prompt: str = Field(default='Write a two-sentence pitch for local AI development.', description='Prompt to send')


class WeatherInput(BaseModel):
    """Input for the weather tool."""

    city: str = Field(default='London', description='City to look up')


class EmbedInput(BaseModel):
    """Input for embedding examples."""

    text: str = Field(default='Local models are useful for private development.', description='Text to embed')


@ai.tool()
async def current_weather(input: WeatherInput) -> str:
    """Return mocked weather data for tool-calling demos."""
    return f'The weather in {input.city} is 18C and partly cloudy.'


@ai.flow(name='chat')
async def chat(input: PromptInput) -> str:
    """Generate a single response with the default Ollama chat model."""
    response = await ai.generate(prompt=input.prompt)
    return response.text


@ai.flow(name='chat_stream')
async def chat_stream(input: PromptInput) -> dict[str, str | int]:
    """Stream a response and return the final text plus chunk count."""
    stream_response = ai.generate_stream(prompt=input.prompt)
    # Ollama streams the text via chunks and returns an empty final message,
    # so accumulate the chunk text instead of reading response.text.
    chunks: list[str] = []
    async for chunk in stream_response.stream:
        chunks.append(chunk.text or '')

    await stream_response.response
    return {
        'chunks': len(chunks),
        'text': ''.join(chunks),
    }


@ai.flow(name='tool_assistant')
async def tool_assistant(input: WeatherInput) -> str:
    """Let the model call a local tool."""
    response = await ai.generate(
        prompt=f'Use the current_weather tool to tell me the weather in {input.city}.',
        tools=['current_weather'],
    )
    return response.text


@ai.flow(name='embed_text')
async def embed_text(input: EmbedInput) -> dict[str, int]:
    """Embed text with Ollama and report vector dimensions."""
    embeddings = await ai.embed(embedder=f'ollama/{EMBEDDER_MODEL}', content=input.text)
    if not embeddings:
        raise RuntimeError('Ollama embedder returned no embeddings for a non-empty input.')
    return {'dimensions': len(embeddings[0].embedding)}


async def main() -> None:
    """Run the fast Ollama demos once."""
    try:
        print(await chat(PromptInput()))
        print(await embed_text(EmbedInput()))
    except GenkitError as error:
        # Genkit wraps provider failures in GenkitError, so unwrap `.cause` to
        # tell a "server not running" setup problem from a real bug.
        if not isinstance(error.cause, OllamaConnectionError):
            raise
        print(
            'Start Ollama and pull the sample models before running this sample directly:\n'
            f'  ollama pull {CHAT_MODEL}\n'
            f'  ollama pull {EMBEDDER_MODEL}\n\n'
            f'{error.cause}'
        )
        raise SystemExit(1) from error


if __name__ == '__main__':
    ai.run_main(main())
