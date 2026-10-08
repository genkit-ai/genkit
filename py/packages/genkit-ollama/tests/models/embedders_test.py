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

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

import ollama as ollama_api
from genkit_ollama._embedders import EmbeddingDefinition, OllamaEmbedder
from pydantic import ValidationError

from genkit import Document, Embedding, GenkitError, Part
from genkit.embedder import EmbedRequest, EmbedResponse


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
                        Part.from_text('doc1_part1'),
                        Part.from_text('doc1_part2'),
                    ]
                ),
                Document(content=[Part.from_text('doc2_part1')]),
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
        request = EmbedRequest(input=[Document(content=[Part.from_text('error text')])])
        self.mock_ollama_client_instance.embed.side_effect = Exception('Ollama Embed API Error')

        with self.assertRaisesRegex(Exception, 'Ollama Embed API Error'):
            await self.ollama_embedder.embed(request)

        self.mock_ollama_client_instance.embed.assert_awaited_once()

    async def test_embed_classifies_server_http_status(self) -> None:
        """A rejected embed call carries the server's HTTP status."""
        request = EmbedRequest(input=[Document.from_text('Smoked salmon tartine')])
        cases = [
            (ollama_api.ResponseError('model "nomic-embed-text" not found, try pulling it first', 404), 'NOT_FOUND'),
            (ollama_api.ResponseError('input length exceeds the context length', 400), 'INVALID_ARGUMENT'),
            (ollama_api.ResponseError('llama runner process has terminated', 500), 'INTERNAL'),
        ]
        for error, status in cases:
            with self.subTest(status=status):
                self.mock_ollama_client_instance.embed.side_effect = error

                with self.assertRaises(GenkitError) as raised:
                    await self.ollama_embedder.embed(request)

                self.assertEqual(raised.exception.status, status)
                self.assertIs(raised.exception.__cause__, error)

    async def test_embed_leaves_missing_http_status_unclassified(self) -> None:
        """ResponseError with status_code -1 has no real status, so it stays raw."""
        request = EmbedRequest(input=[Document.from_text('Smoked salmon tartine')])
        error = ollama_api.ResponseError('unexpected end of stream')
        self.mock_ollama_client_instance.embed.side_effect = error

        with self.assertRaises(ollama_api.ResponseError) as raised:
            await self.ollama_embedder.embed(request)

        self.assertIs(raised.exception, error)

    async def test_embed_marks_malformed_response_internal(self) -> None:
        """A non-JSON body or a payload that fails validation is the server's fault."""
        request = EmbedRequest(input=[Document.from_text('Smoked salmon tartine')])
        cases: list[Exception] = [
            json.JSONDecodeError('Expecting value', '<html>502 Bad Gateway</html>', 0),
            ValidationError.from_exception_data('EmbedResponse', []),
        ]
        for error in cases:
            with self.subTest(error=type(error).__name__):
                self.mock_ollama_client_instance.embed.side_effect = error

                with self.assertRaises(GenkitError) as raised:
                    await self.ollama_embedder.embed(request)

                self.assertEqual(raised.exception.status, 'INTERNAL')
                self.assertIs(raised.exception.__cause__, error)

    async def test_embed_marks_non_numeric_vector_internal(self) -> None:
        """A vector our Embedding model rejects is a malformed response, not a bad request."""
        request = EmbedRequest(input=[Document.from_text('Smoked salmon tartine')])
        bad_response = MagicMock()
        bad_response.embeddings = [['not-a-float']]
        self.mock_ollama_client_instance.embed.side_effect = None
        self.mock_ollama_client_instance.embed.return_value = bad_response

        with self.assertRaises(GenkitError) as raised:
            await self.ollama_embedder.embed(request)

        self.assertEqual(raised.exception.status, 'INTERNAL')

    async def test_embed_response_mismatch_input_count(self) -> None:
        """Test embed when client returns fewer embeddings than input texts (edge case)."""
        request = EmbedRequest(
            input=[
                Document(content=[Part.from_text('text1')]),
                Document(content=[Part.from_text('text2')]),
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
