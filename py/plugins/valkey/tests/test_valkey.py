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
    VALKEY_TEST=1 pytest tests/test_valkey.py -v

Note: These tests use asyncio.run() directly (sync test functions) because
the valkey-glide client's Rust-backed tokio runtime deadlocks under
pytest-asyncio's managed event loop.
"""

from __future__ import annotations

import asyncio
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

FAKE_VECTORS: dict[str, list[float]] = {
    'espresso is strong coffee': [1.0, 0.0, 0.0],
    'latte has steamed milk': [0.9, 0.1, 0.0],
    'croissant is a pastry': [0.0, 0.0, 1.0],
}


async def fake_embed(req: EmbedRequest) -> EmbedResponse:
    """Fake embedder that returns pre-registered vectors."""
    embeddings = []
    for doc in req.input:
        text = ''
        for part in doc.content:
            part_data = part.root if hasattr(part, 'root') else part
            text_val = getattr(part_data, 'text', None)
            if isinstance(text_val, str):
                text += text_val
        vec = FAKE_VECTORS.get(text, [0.5, 0.5, 0.0])
        embeddings.append(Embedding(embedding=vec))
    return EmbedResponse(embeddings=embeddings)


async def _run_index_and_retrieve() -> None:
    """Index documents and retrieve them via KNN search."""
    # Ensure the reflection server does not start.
    os.environ.pop('GENKIT_ENV', None)

    cfg = ValkeyConfig(
        index_name='test-genkit-valkey-py',
        embedder='fake/embedder',
        dimension=3,
        host=VALKEY_HOST,
        port=VALKEY_PORT,
    )

    ai = Genkit(plugins=[Valkey(configs=[cfg])])
    ai.define_embedder('fake/embedder', fake_embed)
    await ai.registry.initialize_all_plugins()

    try:
        docs = [
            Document.from_text('espresso is strong coffee'),
            Document.from_text('latte has steamed milk'),
            Document.from_text('croissant is a pastry'),
        ]

        await ai.index(indexer='valkey/test-genkit-valkey-py', documents=docs)

        resp = await ai.retrieve(
            retriever='valkey/test-genkit-valkey-py',
            query=Document.from_text('espresso is strong coffee'),
            options={'k': 2},
        )

        assert isinstance(resp, RetrieverResponse)
        assert len(resp.documents) == 2

        texts = [d.text for d in resp.documents]
        for text in texts:
            assert 'croissant' not in text, f'Unexpected pastry in results: {text}'
    finally:
        # Cleanup: drop the test index and close connections.
        for plugin in ai.registry._plugins.values():
            if hasattr(plugin, '_clients'):
                client = plugin._clients.get(cfg.index_name)
                if client:
                    from glide import ft as glide_ft
                    try:
                        await glide_ft.dropindex(client, cfg.index_name)
                    except Exception:
                        pass
                await plugin.close()


def test_index_and_retrieve():
    """Test indexing documents and retrieving them via KNN search."""
    asyncio.run(_run_index_and_retrieve())


def test_doc_id_cross_language_vector():
    """Cross-language test vector: must match Go and JS implementations.

    Canonical JSON: {"data":"hello","dataType":"text","metadata":null}
    MD5: 3c04c6b9f04e5e522404b4c567ad09b0
    """
    from genkit.plugins.valkey.plugin import _doc_id

    doc = Document.from_text('hello')
    assert _doc_id(doc) == '3c04c6b9f04e5e522404b4c567ad09b0'
