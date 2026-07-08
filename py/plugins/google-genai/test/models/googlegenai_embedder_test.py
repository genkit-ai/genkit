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

import base64

import pytest
from google import genai
from pytest_mock import MockerFixture

from genkit import (
    Document,
    DocumentPart,
    EmbedRequest,
    EmbedResponse,
    Media,
    MediaPart,
    TextPart,
)
from genkit.plugins.google_genai.models.embedder import (
    Embedder,
    GeminiEmbeddingModels,
    get_embedder_options,
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


@pytest.mark.asyncio
async def test_embedding_forwards_media_parts(mocker: MockerFixture) -> None:
    """Multimodal docs forward media parts to the client alongside the text."""
    text = 'a photo'
    raw_image = b'fake-image-bytes'
    data_url = f'data:image/png;base64,{base64.b64encode(raw_image).decode()}'
    embedding_values = [0.0017063986, -0.044727605, 0.043327782, 0.00044852644]

    doc = Document(
        content=[
            DocumentPart(root=TextPart(text=text)),
            DocumentPart(root=MediaPart(media=Media(url=data_url, content_type='image/png'))),
        ]
    )
    request = EmbedRequest(input=[doc])
    api_response = genai.types.EmbedContentResponse(embeddings=[genai.types.ContentEmbedding(values=embedding_values)])
    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.embed_content.return_value = api_response

    embedder = Embedder(GeminiEmbeddingModels.GEMINI_EMBEDDING_2, googleai_client_mock)

    response = await embedder.generate(request)

    googleai_client_mock.assert_has_calls([
        mocker.call.aio.models.embed_content(
            model=GeminiEmbeddingModels.GEMINI_EMBEDDING_2,
            contents=[
                genai.types.Content(
                    parts=[
                        genai.types.Part.from_text(text=text),
                        genai.types.Part(inline_data=genai.types.Blob(mime_type='image/png', data=raw_image)),
                    ]
                )
            ],
            config=None,
        )
    ])
    assert isinstance(response, EmbedResponse)
    assert len(response.embeddings) == 1
    assert response.embeddings[0].embedding == embedding_values


@pytest.mark.asyncio
async def test_embedding_rejects_empty_input(mocker: MockerFixture) -> None:
    """Empty input must not call the API (avoids opaque BatchEmbedContents errors)."""
    googleai_client_mock = mocker.AsyncMock()
    embedder = Embedder(GeminiEmbeddingModels.GEMINI_EMBEDDING_001, googleai_client_mock)
    with pytest.raises(ValueError, match='Embed request input is empty'):
        await embedder.generate(EmbedRequest(input=[]))
    googleai_client_mock.aio.models.embed_content.assert_not_called()


def test_get_embedder_options_multimodal_and_fallback() -> None:
    """Gemini embedding 2 models are multimodal while unknown stays text-only."""
    options = get_embedder_options('gemini-embedding-2', 'Google AI - gemini-embedding-2')
    assert options.dimensions == 3072
    assert options.supports is not None
    assert options.supports.input == ['text', 'image', 'video']

    unknown_options = get_embedder_options('custom-embedder', 'Google AI - custom-embedder')
    assert unknown_options.dimensions is None
    assert unknown_options.supports is not None
    assert unknown_options.supports.input == ['text']


def test_get_embedder_options_supports_are_backend_scoped() -> None:
    """The same model advertises multimodal on Google AI but text-only on Vertex."""
    vertex_options = get_embedder_options('gemini-embedding-2', 'Vertex AI - gemini-embedding-2', is_vertex=True)
    assert vertex_options.supports is not None
    assert vertex_options.supports.input == ['text']
    assert vertex_options.dimensions == 3072
