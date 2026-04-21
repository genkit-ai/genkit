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

"""Test the Google-Genai embedder model."""

import pytest
from google import genai
from pytest_mock import MockerFixture

from genkit import (
    Document,
    EmbedRequest,
    EmbedResponse,
)
from genkit.plugins.google_genai.models.embedder import (
    Embedder,
    GeminiEmbeddingModels,
)


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [x for x in GeminiEmbeddingModels])
async def test_embedding(mocker: MockerFixture, version: GeminiEmbeddingModels) -> None:
    """Test the embedding method."""
    request_text = 'request text'
    embedding_values = [0.0017063986, -0.044727605, 0.043327782, 0.00044852644]

    request = EmbedRequest(input=[Document.from_text(request_text)])
    api_response = genai.types.EmbedContentResponse(embeddings=[genai.types.ContentEmbedding(values=embedding_values)])
    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.embed_content.return_value = api_response

    embedder = Embedder(version, googleai_client_mock)

    response = await embedder.generate(request)

    googleai_client_mock.assert_has_calls([
        mocker.call.aio.models.embed_content(
            model=version,
            contents=[genai.types.Content(parts=[genai.types.Part.from_text(text=request_text)])],
            config=None,
        )
    ])
    assert isinstance(response, EmbedResponse)
    assert len(response.embeddings) == 1
    assert response.embeddings[0].embedding == embedding_values
