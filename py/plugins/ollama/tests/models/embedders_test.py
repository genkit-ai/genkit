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

"""Unit tests for Ollama embedders package."""

import unittest
from unittest.mock import AsyncMock, MagicMock

import ollama as ollama_api

from genkit import (
    Document,
    DocumentPart,
    Embedding,
    EmbedRequest,
    EmbedResponse,
    TextPart,
)
from genkit.plugins.ollama.embedders import EmbeddingDefinition, OllamaEmbedder


class TestOllamaEmbedderEmbed(unittest.IsolatedAsyncioTestCase):
    """Unit tests for OllamaEmbedder.embed method."""

    async def asyncSetUp(self) -> None:
        """Common setup."""
        self.mock_ollama_client_instance = AsyncMock()
        self.mock_ollama_client_factory = MagicMock(return_value=self.mock_ollama_client_instance)

        self.mock_embedding_definition = EmbeddingDefinition(name='test-embed-model', dimensions=1536)
        self.ollama_embedder = OllamaEmbedder(
            client=self.mock_ollama_client_factory, embedding_definition=self.mock_embedding_definition
        )

    async def test_embed_single_document_single_content(self) -> None:
        """Test embed with a single document containing single text content."""
        request = EmbedRequest(
            input=[
                Document.from_text(text='hello world'),
            ]
        )
        expected_ollama_embeddings = [[0.1, 0.2, 0.3]]
        self.mock_ollama_client_instance.embed.return_value = ollama_api.EmbedResponse(
            embeddings=expected_ollama_embeddings
        )

        response = await self.ollama_embedder.embed(request)

        # Assertions
        self.mock_ollama_client_instance.embed.assert_awaited_once_with(
            model='test-embed-model',
            input=['hello world'],
        )
        expected_genkit_embeddings = [Embedding(embedding=[0.1, 0.2, 0.3])]
        self.assertEqual(response, EmbedResponse(embeddings=expected_genkit_embeddings))

    async def test_embed_multiple_documents_multiple_content(self) -> None:
        """Test embed with multiple documents, each with multiple text contents."""
        request = EmbedRequest(
            input=[
                Document(
                    content=[
                        DocumentPart(root=TextPart(text='doc1_part1')),
                        DocumentPart(root=TextPart(text='doc1_part2')),
                    ]
                ),
                Document(content=[DocumentPart(root=TextPart(text='doc2_part1'))]),
            ]
        )
        expected_ollama_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        self.mock_ollama_client_instance.embed.return_value = ollama_api.EmbedResponse(
            embeddings=expected_ollama_embeddings
        )

        response = await self.ollama_embedder.embed(request)

        # Assertions
        self.mock_ollama_client_instance.embed.assert_awaited_once_with(
            model='test-embed-model',
            input=['doc1_part1', 'doc1_part2', 'doc2_part1'],
        )
        expected_genkit_embeddings = [
            Embedding(embedding=[0.1, 0.2]),
            Embedding(embedding=[0.3, 0.4]),
            Embedding(embedding=[0.5, 0.6]),
        ]
        self.assertEqual(response, EmbedResponse(embeddings=expected_genkit_embeddings))

    async def test_embed_empty_input(self) -> None:
        """Test embed with an empty input request."""
        request = EmbedRequest(input=[])
        self.mock_ollama_client_instance.embed.return_value = ollama_api.EmbedResponse(embeddings=[])

        response = await self.ollama_embedder.embed(request)

        # Assertions
        self.mock_ollama_client_instance.embed.assert_awaited_once_with(
            model='test-embed-model',
            input=[],
        )
        self.assertEqual(response, EmbedResponse(embeddings=[]))

    async def test_embed_api_raises_exception(self) -> None:
        """Test embed method handles exception from client.embed."""
        request = EmbedRequest(input=[Document(content=[DocumentPart(root=TextPart(text='error text'))])])
        self.mock_ollama_client_instance.embed.side_effect = Exception('Ollama Embed API Error')

        with self.assertRaisesRegex(Exception, 'Ollama Embed API Error'):
            await self.ollama_embedder.embed(request)

        self.mock_ollama_client_instance.embed.assert_awaited_once()

    async def test_embed_response_mismatch_input_count(self) -> None:
        """Test embed when client returns fewer embeddings than input texts (edge case)."""
        request = EmbedRequest(
            input=[
                Document(content=[DocumentPart(root=TextPart(text='text1'))]),
                Document(content=[DocumentPart(root=TextPart(text='text2'))]),
            ]
        )
        # Simulate Ollama returning only one embedding for two inputs
        expected_ollama_embeddings = [[1.0, 2.0]]
        self.mock_ollama_client_instance.embed.return_value = ollama_api.EmbedResponse(
            embeddings=expected_ollama_embeddings
        )

        response = await self.ollama_embedder.embed(request)

        # The current implementation will just use whatever embeddings are returned.
        # It's up to the caller or a higher layer to decide if this is an error.
        # This test ensures it doesn't crash and correctly maps the available embeddings.
        expected_genkit_embeddings = [Embedding(embedding=[1.0, 2.0])]
        self.assertEqual(response, EmbedResponse(embeddings=expected_genkit_embeddings))
        self.assertEqual(len(response.embeddings), 1)  # Confirm only one embedding was processed
