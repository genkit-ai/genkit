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

"""Integration tests for the Valkey plugin.

Requires a running Valkey instance with valkey-search module:
    docker run -d --name valkey -p 6379:6379 valkey/valkey:8-alpine

Run:
    pytest tests/test_valkey.py -v
"""

from __future__ import annotations

import os

import pytest

from genkit import Document, Genkit
from genkit._ai._retriever import RetrieverResponse
from genkit._core._typing import EmbedRequest, EmbedResponse, Embedding
from genkit.plugins.valkey import Valkey, ValkeyConfig


VALKEY_HOST = os.environ.get('VALKEY_HOST', 'localhost')
VALKEY_PORT = int(os.environ.get('VALKEY_PORT', '6379'))

# Skip all tests if VALKEY_TEST env var is not set
pytestmark = pytest.mark.skipif(
    os.environ.get('VALKEY_TEST') != '1',
    reason='Set VALKEY_TEST=1 to run Valkey integration tests',
)


# --- Fake embedder that returns known vectors ---

# Coffee docs get vectors near [1, 0, 0]; pastry near [0, 0, 1]
FAKE_VECTORS: dict[str, list[float]] = {
    'espresso is strong coffee': [1.0, 0.0, 0.0],
    'latte has steamed milk': [0.9, 0.1, 0.0],
    'croissant is a pastry': [0.0, 0.0, 1.0],
}


async def fake_embed(req: EmbedRequest) -> EmbedResponse:
    """Fake embedder that returns pre-registered vectors."""
    embeddings = []
    for doc in req.input:
        # Extract text from document
        text = ''
        for part in doc.content:
            part_data = part.root if hasattr(part, 'root') else part
            text_val = getattr(part_data, 'text', None)
            if isinstance(text_val, str):
                text += text_val

        vec = FAKE_VECTORS.get(text, [0.5, 0.5, 0.0])
        embeddings.append(Embedding(embedding=vec))
    return EmbedResponse(embeddings=embeddings)


@pytest.fixture
async def genkit_app():
    """Create a Genkit instance with the Valkey plugin and fake embedder."""
    cfg = ValkeyConfig(
        index_name='test-genkit-valkey-py',
        embedder='fake/embedder',
        dimension=3,
        host=VALKEY_HOST,
        port=VALKEY_PORT,
    )

    ai = Genkit(plugins=[Valkey(configs=[cfg])])
    ai.define_embedder('fake/embedder', fake_embed)

    # Ensure plugins are initialized
    await ai.registry.initialize_all_plugins()

    yield ai

    # Cleanup: drop the test index to avoid stale data on repeated runs.
    plugin = ai.registry.lookup_plugin('valkey')
    if plugin and hasattr(plugin, '_clients'):
        client = plugin._clients.get(cfg.index_name)
        if client:
            try:
                from glide import ft as glide_ft
                await glide_ft.dropindex(client, cfg.index_name)
            except Exception:
                pass
        await plugin.close()


@pytest.mark.asyncio
async def test_index_and_retrieve(genkit_app):
    """Test indexing documents and retrieving them via KNN search."""
    ai = genkit_app

    docs = [
        Document.from_text('espresso is strong coffee'),
        Document.from_text('latte has steamed milk'),
        Document.from_text('croissant is a pastry'),
    ]

    # Index documents
    await ai.index(indexer='valkey/test-genkit-valkey-py', documents=docs)

    # Retrieve — query vector is near coffee cluster
    resp = await ai.retrieve(
        retriever='valkey/test-genkit-valkey-py',
        query=Document.from_text('espresso is strong coffee'),
        options={'k': 2},
    )

    assert isinstance(resp, RetrieverResponse)
    assert len(resp.documents) == 2

    # Both results should be coffee-related
    texts = [d.text for d in resp.documents]
    for text in texts:
        assert 'croissant' not in text, f'Unexpected pastry in results: {text}'

