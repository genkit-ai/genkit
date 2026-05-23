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
# pyright: reportMissingImports=false

"""Live integration tests against a real Ollama server.

These tests are skipped by default. To run them:

    OLLAMA_LIVE_TEST=1 uv run pytest tests/live_integration_test.py

Optional environment variables:

- ``OLLAMA_HOST`` — server URL (default: ``http://127.0.0.1:11434``)
- ``OLLAMA_LIVE_CHAT_MODEL`` — chat model name (default: ``llama3.2``)
- ``OLLAMA_LIVE_EMBEDDER_MODEL`` — embedder name (default: ``nomic-embed-text``)

Both models must be pulled (``ollama pull <name>``) before running.
"""

from __future__ import annotations

import os

import pytest

from genkit import Genkit
from genkit.plugins.ollama import EmbeddingDefinition, ModelDefinition, Ollama

LIVE_ENABLED = os.getenv('OLLAMA_LIVE_TEST') == '1'
CHAT_MODEL = os.getenv('OLLAMA_LIVE_CHAT_MODEL', 'llama3.2')
EMBEDDER_MODEL = os.getenv('OLLAMA_LIVE_EMBEDDER_MODEL', 'nomic-embed-text')

pytestmark = [
    pytest.mark.skipif(not LIVE_ENABLED, reason='set OLLAMA_LIVE_TEST=1 to enable'),
    pytest.mark.asyncio,
]


def _plugin() -> Ollama:
    kwargs: dict[str, object] = {
        'models': [ModelDefinition(name=CHAT_MODEL)],
        'embedders': [EmbeddingDefinition(name=EMBEDDER_MODEL)],
    }
    if host := os.getenv('OLLAMA_HOST'):
        kwargs['server_address'] = host
    return Ollama(**kwargs)


async def test_chat_returns_text() -> None:
    ai = Genkit(plugins=[_plugin()], model=f'ollama/{CHAT_MODEL}')
    response = await ai.generate(prompt='Reply with the single word: pong.')
    assert response.text
    assert isinstance(response.text, str)


async def test_streaming_emits_chunks() -> None:
    ai = Genkit(plugins=[_plugin()], model=f'ollama/{CHAT_MODEL}')
    stream = ai.generate_stream(prompt='Count 1 to 3, one per line.')
    chunks: list[str] = []
    async for chunk in stream.stream:
        if chunk.text:
            chunks.append(chunk.text)
    final = await stream.response
    assert chunks, 'expected at least one streamed chunk'
    assert final.text


async def test_embeddings_have_dimensions() -> None:
    ai = Genkit(plugins=[_plugin()], model=f'ollama/{CHAT_MODEL}')
    embeddings = await ai.embed(embedder=f'ollama/{EMBEDDER_MODEL}', content='hello world')
    assert embeddings, 'expected at least one embedding'
    assert len(embeddings[0].embedding) > 0
