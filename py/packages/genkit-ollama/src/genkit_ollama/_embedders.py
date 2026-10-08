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


"""Ollama embedders."""

import json
from collections.abc import Callable

import ollama as ollama_api
from pydantic import ValidationError

from genkit import Embedding, GenkitError
from genkit.embedder import EmbedRequest, EmbedResponse
from genkit.plugin_api import wrap_http_error


class OllamaEmbedder:
    """Handles embedding requests using an Ollama embedding model.

    This class provides the necessary logic to interact with a specific
    Ollama embedding model, processing input text into vector embeddings.
    """

    def __init__(
        self,
        client: Callable,
        model: str,
    ) -> None:
        """Initializes the OllamaEmbedder.

        Sets up the client factory for communicating with the Ollama server and stores
        the embedding model name.

        Note: We store the client factory (not the client instance) to avoid async
        event loop binding issues. The client is created fresh per request to ensure
        it's bound to the correct event loop.

        Args:
            client: A callable that returns an asynchronous Ollama client instance.
            model: The Ollama embedding model name, e.g. ``nomic-embed-text``.
        """
        self._client_factory = client
        self.model = model

    def _get_client(self) -> ollama_api.AsyncClient:
        """Creates a fresh async client bound to the current event loop.

        Returns:
            A fresh Ollama async client instance.
        """
        return self._client_factory()

    async def embed(self, request: EmbedRequest, client: ollama_api.AsyncClient | None = None) -> EmbedResponse:
        """Generates embeddings for the provided input text.

        Converts the input documents from the Genkit EmbedRequest into a raw
        list of strings, sends them to the Ollama server for embedding, and then
        formats the response into a Genkit EmbedResponse.

        Args:
            request: The embedding request containing the input documents.
            client: An optional pre-resolved Ollama client (e.g. one built with
                per-request headers); falls back to the stored client factory.

        Returns:
            An EmbedResponse containing the generated vector embeddings.

        Raises:
            GenkitError: The server's HTTP status for a rejected call (e.g.
                NOT_FOUND for a model that is not pulled), or INTERNAL for a
                response that is not a valid embed response.
        """
        if client is None:
            client = self._get_client()
        input_raw: list[str] = []
        for doc in request.input:
            input_raw.extend([str(content.text) for content in doc.content if content.text is not None])
        try:
            response = await client.embed(
                model=self.model,
                input=input_raw,
            )
            return EmbedResponse(embeddings=[Embedding(embedding=list(embedding)) for embedding in response.embeddings])
        except ollama_api.ResponseError as e:
            # No real HTTP status (-1) is re-raised as-is by wrap_http_error.
            raise wrap_http_error(e, status_code=e.status_code) from e
        except (json.JSONDecodeError, ValidationError) as e:
            raise GenkitError(status='INTERNAL', message='ollama embed: malformed response from server', cause=e) from e
