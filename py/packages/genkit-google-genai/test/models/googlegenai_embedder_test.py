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

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest
from genkit_google_genai.models.embedder import (
    EMBED_CONCURRENCY_LIMIT,
    GOOGLEAI_EMBED_BATCH_SIZE,
    VERTEXAI_EMBED_BATCH_SIZE,
    Embedder,
    EmbeddingConfigSchema,
    EmbeddingTaskType,
    GeminiEmbeddingModels,
    VertexEmbeddingConfigSchema,
    get_embedder_info,
)
from google import genai
from pytest_mock import MockerFixture

from genkit import (
    Document,
    DocumentPart,
    EmbedRequest,
    EmbedResponse,
    GenkitError,
    Media,
    MediaPart,
    TextPart,
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
async def test_options_version_is_the_api_model(mocker: MockerFixture) -> None:
    """options.version overlays the action id as the model sent to embed_content."""
    request_text = 'request text'
    embedding_values = [0.1, 0.2]
    request = EmbedRequest(
        input=[Document.from_text(request_text)],
        options={'version': 'vertexai/text-embedding-005'},
    )
    api_response = genai.types.EmbedContentResponse(embeddings=[genai.types.ContentEmbedding(values=embedding_values)])
    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.embed_content.return_value = api_response

    embedder = Embedder('text-embedding-004', googleai_client_mock)
    await embedder.generate(request)

    googleai_client_mock.aio.models.embed_content.assert_called_once()
    assert googleai_client_mock.aio.models.embed_content.call_args.kwargs['model'] == 'text-embedding-005'


def _single_embedding_client(mocker: MockerFixture) -> AsyncMock:
    client = mocker.AsyncMock()
    client.aio.models.embed_content.return_value = genai.types.EmbedContentResponse(
        embeddings=[genai.types.ContentEmbedding(values=[0.1, 0.2])]
    )
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'options',
    [
        {'task_type': 'RETRIEVAL_QUERY', 'title': 'doc', 'output_dimensionality': 256},
        {'taskType': 'RETRIEVAL_QUERY', 'title': 'doc', 'outputDimensionality': 256},
    ],
    ids=['snake_case', 'camelCase'],
)
async def test_options_map_to_embed_content_config(mocker: MockerFixture, options: dict[str, object]) -> None:
    """snake_case and camelCase option keys both reach EmbedContentConfig."""
    client = _single_embedding_client(mocker)
    embedder = Embedder(GeminiEmbeddingModels.GEMINI_EMBEDDING_001, client)

    await embedder.generate(EmbedRequest(input=[Document.from_text('text')], options=options))

    config = client.aio.models.embed_content.call_args.kwargs['config']
    assert config == genai.types.EmbedContentConfig(task_type='RETRIEVAL_QUERY', title='doc', output_dimensionality=256)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'options',
    [
        {'autoTruncate': False, 'mimeType': 'text/plain'},
        VertexEmbeddingConfigSchema(auto_truncate=False, mime_type='text/plain'),
    ],
    ids=['dict', 'typed'],
)
async def test_vertex_only_options_map_to_embed_content_config(mocker: MockerFixture, options: object) -> None:
    """Vertex embedders forward autoTruncate and mimeType to EmbedContentConfig."""
    client = _single_embedding_client(mocker)
    embedder = Embedder('text-embedding-005', client, is_vertex=True)

    await embedder.generate(EmbedRequest(input=[Document.from_text('text')], options=options))

    config = client.aio.models.embed_content.call_args.kwargs['config']
    assert config == genai.types.EmbedContentConfig(auto_truncate=False, mime_type='text/plain')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'options',
    [
        {'taskType': 'CLUSTERING', 'autoTruncate': False, 'mimeType': 'text/plain'},
        VertexEmbeddingConfigSchema(
            task_type=EmbeddingTaskType.CLUSTERING, auto_truncate=False, mime_type='text/plain'
        ),
    ],
    ids=['dict', 'typed'],
)
async def test_vertex_only_options_are_not_forwarded_on_gemini_api(mocker: MockerFixture, options: object) -> None:
    """Gemini API embedders accept autoTruncate and mimeType but leave them out of EmbedContentConfig."""
    client = _single_embedding_client(mocker)
    embedder = Embedder(GeminiEmbeddingModels.GEMINI_EMBEDDING_001, client)

    await embedder.generate(EmbedRequest(input=[Document.from_text('text')], options=options))

    config = client.aio.models.embed_content.call_args.kwargs['config']
    assert config == genai.types.EmbedContentConfig(task_type='CLUSTERING')


@pytest.mark.asyncio
async def test_unknown_option_keys_are_tolerated(mocker: MockerFixture) -> None:
    """Unknown option keys do not raise and are not forwarded to the config."""
    client = _single_embedding_client(mocker)
    embedder = Embedder(GeminiEmbeddingModels.GEMINI_EMBEDDING_001, client)

    await embedder.generate(
        EmbedRequest(input=[Document.from_text('text')], options={'taskType': 'CLUSTERING', 'someFutureFlag': True})
    )

    config = client.aio.models.embed_content.call_args.kwargs['config']
    assert config == genai.types.EmbedContentConfig(task_type='CLUSTERING')


@pytest.mark.asyncio
async def test_typed_options_instance_is_accepted(mocker: MockerFixture) -> None:
    """An EmbeddingConfigSchema instance can be passed as the request options."""
    client = _single_embedding_client(mocker)
    embedder = Embedder(GeminiEmbeddingModels.GEMINI_EMBEDDING_001, client)

    await embedder.generate(
        EmbedRequest(
            input=[Document.from_text('text')],
            options=EmbeddingConfigSchema(task_type=EmbeddingTaskType.CLASSIFICATION),
        )
    )

    config = client.aio.models.embed_content.call_args.kwargs['config']
    assert config == genai.types.EmbedContentConfig(task_type='CLASSIFICATION')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('options', 'field'),
    [
        ({'outputDimensionality': 0}, 'outputDimensionality'),
        ({'output_dimensionality': -1}, 'output_dimensionality'),
        ({'taskType': 'NOT_A_TASK'}, 'taskType'),
    ],
    ids=['zero_dimensionality', 'negative_dimensionality', 'unknown_task_type'],
)
async def test_invalid_options_are_rejected(mocker: MockerFixture, options: dict[str, object], field: str) -> None:
    """Invalid option values raise INVALID_ARGUMENT naming the field, before any request is sent."""
    client = mocker.AsyncMock()
    embedder = Embedder(GeminiEmbeddingModels.GEMINI_EMBEDDING_001, client)

    with pytest.raises(GenkitError) as exc_info:
        await embedder.generate(EmbedRequest(input=[Document.from_text('text')], options=options))

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.original_message == f'gemini-embedding-001: invalid config field {field}'
    client.aio.models.embed_content.assert_not_called()


def test_embedding_task_type_includes_code_retrieval_query() -> None:
    """CODE_RETRIEVAL_QUERY is an accepted task type."""
    assert EmbeddingTaskType.CODE_RETRIEVAL_QUERY == 'CODE_RETRIEVAL_QUERY'
    assert EmbeddingConfigSchema.model_validate({'taskType': 'CODE_RETRIEVAL_QUERY'}).task_type == (
        EmbeddingTaskType.CODE_RETRIEVAL_QUERY
    )


def _indexed_embed_content(
    *, model: str, contents: list[genai.types.Content], config: object
) -> genai.types.EmbedContentResponse:
    """Return one embedding per content whose value is the text of that content."""
    values: list[float] = []
    for content in contents:
        part = (content.parts or [])[0]
        values.append(float(part.text or 'nan'))
    return genai.types.EmbedContentResponse(embeddings=[genai.types.ContentEmbedding(values=[v]) for v in values])


def _numbered_docs(count: int) -> list[Document]:
    return [Document.from_text(str(i)) for i in range(count)]


@pytest.mark.asyncio
async def test_googleai_text_embedding_batches_requests(mocker: MockerFixture) -> None:
    """Gemini API requests are split into batches of 100 and embeddings kept in input order."""
    count = GOOGLEAI_EMBED_BATCH_SIZE + 50
    client = mocker.AsyncMock()
    client.aio.models.embed_content.side_effect = _indexed_embed_content
    embedder = Embedder(GeminiEmbeddingModels.GEMINI_EMBEDDING_001, client)

    response = await embedder.generate(EmbedRequest(input=_numbered_docs(count)))

    calls = client.aio.models.embed_content.call_args_list
    assert [len(call.kwargs['contents']) for call in calls] == [GOOGLEAI_EMBED_BATCH_SIZE, 50]
    assert [e.embedding for e in response.embeddings] == [[float(i)] for i in range(count)]


@pytest.mark.asyncio
async def test_vertex_text_embedding_batches_requests(mocker: MockerFixture) -> None:
    """Vertex text-embedding requests are split into batches of 250."""
    count = VERTEXAI_EMBED_BATCH_SIZE + 1
    client = mocker.AsyncMock()
    client.aio.models.embed_content.side_effect = _indexed_embed_content
    embedder = Embedder('text-embedding-005', client, is_vertex=True)

    response = await embedder.generate(EmbedRequest(input=_numbered_docs(count)))

    calls = client.aio.models.embed_content.call_args_list
    assert [len(call.kwargs['contents']) for call in calls] == [VERTEXAI_EMBED_BATCH_SIZE, 1]
    assert [e.embedding for e in response.embeddings] == [[float(i)] for i in range(count)]


@pytest.mark.asyncio
async def test_vertex_gemini_embedding_sends_one_document_per_request(mocker: MockerFixture) -> None:
    """Vertex gemini-embedding models accept a single input per request."""
    client = mocker.AsyncMock()
    client.aio.models.embed_content.side_effect = _indexed_embed_content
    embedder = Embedder('gemini-embedding-001', client, is_vertex=True)

    response = await embedder.generate(EmbedRequest(input=_numbered_docs(3)))

    calls = client.aio.models.embed_content.call_args_list
    assert [len(call.kwargs['contents']) for call in calls] == [1, 1, 1]
    assert [e.embedding for e in response.embeddings] == [[0.0], [1.0], [2.0]]


@pytest.mark.asyncio
async def test_embedding_count_mismatch_raises(mocker: MockerFixture) -> None:
    """A response with a different number of embeddings than documents is an error."""
    client = mocker.AsyncMock()
    client.aio.models.embed_content.return_value = genai.types.EmbedContentResponse(embeddings=[])
    embedder = Embedder(GeminiEmbeddingModels.GEMINI_EMBEDDING_001, client)

    with pytest.raises(GenkitError, match='returned 0 embeddings for 1 documents') as exc_info:
        await embedder.generate(EmbedRequest(input=[Document.from_text('text')]))

    assert exc_info.value.status == 'INTERNAL'


def test_get_embedder_info_exposes_config_schema() -> None:
    """Gemini API embedders advertise the common camelCase options and no Vertex-only ones."""
    options = get_embedder_info('gemini-embedding-001', 'Google AI - gemini-embedding-001')

    assert options.config_schema is not None
    properties = set(options.config_schema['properties'])
    assert {'taskType', 'title', 'outputDimensionality', 'version'} <= properties
    assert properties.isdisjoint({'mimeType', 'autoTruncate', 'task_type'})


def test_get_embedder_info_exposes_vertex_config_schema() -> None:
    """Vertex embedders advertise the common options plus mimeType and autoTruncate."""
    options = get_embedder_info('text-embedding-005', 'Vertex AI - text-embedding-005', is_vertex=True)

    assert options.config_schema is not None
    properties = set(options.config_schema['properties'])
    assert {'taskType', 'title', 'outputDimensionality', 'version', 'mimeType', 'autoTruncate'} <= properties
    assert 'task_type' not in properties


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


def test_get_embedder_info_multimodal_and_fallback() -> None:
    """Gemini embedding 2 models are multimodal while unknown stays text-only."""
    options = get_embedder_info('gemini-embedding-2', 'Google AI - gemini-embedding-2')
    assert options.dimensions == 3072
    assert options.supports is not None
    assert options.supports.input == ['text', 'image', 'video']

    unknown_options = get_embedder_info('custom-embedder', 'Google AI - custom-embedder')
    assert unknown_options.dimensions is None
    assert unknown_options.supports is not None
    assert unknown_options.supports.input == ['text']


@pytest.mark.parametrize('model_name', ['multimodalembedding', 'multimodalembedding@001'])
def test_get_embedder_info_multimodalembedding_versioned_and_bare(model_name: str) -> None:
    """The multimodalembedding model is multimodal with or without the '@001' suffix."""
    options = get_embedder_info(model_name, f'Vertex AI - {model_name}', is_vertex=True)
    assert options.dimensions == 1408
    assert options.supports is not None
    assert options.supports.input == ['text', 'image', 'video']


def test_get_embedder_info_scopes_supports_per_backend() -> None:
    """Each backend advertises only its own multimodal models, defaulting to text."""
    on_vertex = get_embedder_info('gemini-embedding-2', 'Vertex AI - gemini-embedding-2', is_vertex=True)
    assert on_vertex.supports is not None
    assert on_vertex.supports.input == ['text']

    on_googleai = get_embedder_info('multimodalembedding@001', 'Google AI - multimodalembedding@001')
    assert on_googleai.supports is not None
    assert on_googleai.supports.input == ['text']


@pytest.mark.asyncio
async def test_multimodal_embedding_uses_predict(mocker: MockerFixture) -> None:
    """A multimodal embed routes through :predict, not the text embed_content path."""
    request = EmbedRequest(input=[Document.from_media('gs://bucket/cat.png', 'image/png')])
    predict_body = {'predictions': [{'imageEmbedding': [0.1, 0.2, 0.3]}]}
    client_mock = mocker.AsyncMock()
    http_response = mocker.Mock()
    http_response.body = json.dumps(predict_body)
    client_mock._api_client.async_request.return_value = http_response

    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    response = await embedder.generate(request)

    # The text embed_content path must not be used for multimodal models.
    client_mock.aio.models.embed_content.assert_not_called()

    call = client_mock._api_client.async_request.call_args
    assert call.kwargs['http_method'] == 'post'
    assert call.kwargs['path'] == 'publishers/google/models/multimodalembedding:predict'
    instances = call.kwargs['request_dict']['instances']
    assert instances == [{'image': {'gcsUri': 'gs://bucket/cat.png', 'mimeType': 'image/png'}}]

    assert isinstance(response, EmbedResponse)
    assert len(response.embeddings) == 1
    assert response.embeddings[0].embedding == [0.1, 0.2, 0.3]
    assert response.embeddings[0].metadata == {'embedType': 'imageEmbedding'}


@pytest.mark.asyncio
async def test_multimodal_embedding_concatenates_text_parts(mocker: MockerFixture) -> None:
    """Multiple text parts in one document are concatenated into a single instance text."""
    request = EmbedRequest(
        input=[
            Document(
                content=[
                    *Document.from_text('hello ').content,
                    *Document.from_text('world').content,
                ]
            )
        ]
    )
    predict_body = {'predictions': [{'textEmbedding': [0.1, 0.2]}]}
    client_mock = mocker.AsyncMock()
    http_response = mocker.Mock()
    http_response.body = json.dumps(predict_body)
    client_mock._api_client.async_request.return_value = http_response

    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    response = await embedder.generate(request)

    call = client_mock._api_client.async_request.call_args
    instances = call.kwargs['request_dict']['instances']
    assert instances[0] == {'text': 'hello world'}
    assert response.embeddings[0].embedding == [0.1, 0.2]
    assert response.embeddings[0].metadata == {'embedType': 'textEmbedding'}


@pytest.mark.asyncio
async def test_multimodal_embedding_allows_image_and_video_in_one_instance(mocker: MockerFixture) -> None:
    """Image and video in one document share a single instance (Vertex supports this)."""
    request = EmbedRequest(
        input=[
            Document(
                content=[
                    *Document.from_media('gs://bucket/cat.png', 'image/png').content,
                    *Document.from_media('gs://bucket/clip.mp4', 'video/mp4').content,
                ]
            )
        ]
    )
    predict_body = {
        'predictions': [
            {
                'imageEmbedding': [0.1, 0.2],
                'videoEmbeddings': [{'startOffsetSec': 0, 'endOffsetSec': 5, 'embedding': [0.3, 0.4]}],
            }
        ]
    }
    client_mock = mocker.AsyncMock()
    http_response = mocker.Mock()
    http_response.body = json.dumps(predict_body)
    client_mock._api_client.async_request.return_value = http_response

    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    response = await embedder.generate(request)

    call = client_mock._api_client.async_request.call_args
    instances = call.kwargs['request_dict']['instances']
    assert instances[0] == {
        'image': {'gcsUri': 'gs://bucket/cat.png', 'mimeType': 'image/png'},
        'video': {'gcsUri': 'gs://bucket/clip.mp4'},
    }
    assert len(response.embeddings) == 2
    assert response.embeddings[0].metadata == {'embedType': 'imageEmbedding'}
    assert response.embeddings[1].metadata is not None
    assert response.embeddings[1].metadata['embedType'] == 'videoEmbedding'
    # Video chunk offsets are preserved in the embedding metadata.
    assert response.embeddings[1].metadata['startOffsetSec'] == 0


@pytest.mark.asyncio
async def test_multimodal_embedding_rejects_multiple_images(mocker: MockerFixture) -> None:
    """A document with two images is rejected; Vertex accepts one image per instance."""
    request = EmbedRequest(
        input=[
            Document(
                content=[
                    *Document.from_media('gs://bucket/a.png', 'image/png').content,
                    *Document.from_media('gs://bucket/b.png', 'image/png').content,
                ]
            )
        ]
    )
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    with pytest.raises(ValueError, match='more than one image'):
        await embedder.generate(request)
    client_mock._api_client.async_request.assert_not_called()


@pytest.mark.asyncio
async def test_multimodal_embedding_rejects_multiple_videos(mocker: MockerFixture) -> None:
    """A document with two videos is rejected; Vertex accepts one video per instance."""
    request = EmbedRequest(
        input=[
            Document(
                content=[
                    *Document.from_media('gs://bucket/a.mp4', 'video/mp4').content,
                    *Document.from_media('gs://bucket/b.mp4', 'video/mp4').content,
                ]
            )
        ]
    )
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    with pytest.raises(ValueError, match='more than one video'):
        await embedder.generate(request)
    client_mock._api_client.async_request.assert_not_called()


@pytest.mark.asyncio
async def test_multimodal_embedding_rejects_http_url(mocker: MockerFixture) -> None:
    """http(s) media URLs are rejected; Vertex gcsUri only accepts gs:// (diverges from JS)."""
    request = EmbedRequest(input=[Document.from_media('https://example.com/cat.png', 'image/png')])
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    with pytest.raises(ValueError, match='http'):
        await embedder.generate(request)
    client_mock._api_client.async_request.assert_not_called()


@pytest.mark.asyncio
async def test_multimodal_embedding_sends_one_predict_request_per_document(mocker: MockerFixture) -> None:
    """Each document becomes its own single-instance :predict call; results keep document order."""
    request = EmbedRequest(
        input=[
            Document.from_media('gs://bucket/a.png', 'image/png'),
            Document.from_media('gs://bucket/b.png', 'image/png'),
        ],
        options={'outputDimensionality': 256},
    )
    responses = []
    for values in ([0.1, 0.2], [0.3, 0.4]):
        http_response = mocker.Mock()
        http_response.body = json.dumps({'predictions': [{'imageEmbedding': values}]})
        responses.append(http_response)
    client_mock = mocker.AsyncMock()
    client_mock._api_client.async_request.side_effect = responses

    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    response = await embedder.generate(request)

    calls = client_mock._api_client.async_request.call_args_list
    assert [call.kwargs['request_dict']['instances'] for call in calls] == [
        [{'image': {'gcsUri': 'gs://bucket/a.png', 'mimeType': 'image/png'}}],
        [{'image': {'gcsUri': 'gs://bucket/b.png', 'mimeType': 'image/png'}}],
    ]
    assert all(call.kwargs['request_dict']['parameters'] == {'dimension': 256} for call in calls)
    assert [e.embedding for e in response.embeddings] == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_multimodal_embedding_validates_all_documents_before_requesting(mocker: MockerFixture) -> None:
    """An invalid document anywhere in the batch fails before any :predict call is made."""
    request = EmbedRequest(
        input=[
            Document.from_media('gs://bucket/a.png', 'image/png'),
            Document.from_media('https://example.com/b.png', 'image/png'),
        ]
    )
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    with pytest.raises(ValueError, match='http'):
        await embedder.generate(request)
    client_mock._api_client.async_request.assert_not_called()


@pytest.mark.asyncio
async def test_multimodal_embedding_rejects_non_vertex_client(mocker: MockerFixture) -> None:
    """Multimodal embedding is Vertex-only; a Gemini API embedder fails fast, before any request."""
    request = EmbedRequest(input=[Document.from_media('gs://bucket/cat.png', 'image/png')])
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding@001', client_mock, is_vertex=False)
    with pytest.raises(ValueError, match='only available on Vertex AI'):
        await embedder.generate(request)
    client_mock._api_client.async_request.assert_not_called()


@pytest.mark.asyncio
async def test_multimodal_embedding_inlines_base64_data_url(mocker: MockerFixture) -> None:
    """A base64 data: URL is inlined as bytesBase64Encoded, without the data: prefix."""
    request = EmbedRequest(input=[Document.from_media('data:image/png;base64,AAAA', 'image/png')])
    predict_body = {'predictions': [{'imageEmbedding': [0.1, 0.2]}]}
    client_mock = mocker.AsyncMock()
    http_response = mocker.Mock()
    http_response.body = json.dumps(predict_body)
    client_mock._api_client.async_request.return_value = http_response

    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    await embedder.generate(request)

    call = client_mock._api_client.async_request.call_args
    instances = call.kwargs['request_dict']['instances']
    assert instances[0] == {'image': {'bytesBase64Encoded': 'AAAA', 'mimeType': 'image/png'}}


@pytest.mark.asyncio
async def test_multimodal_embedding_rejects_non_base64_data_url(mocker: MockerFixture) -> None:
    """A data: URL without a ';base64,' marker is rejected; Vertex requires base64 bytes."""
    request = EmbedRequest(input=[Document.from_media('data:image/png,rawbytes', 'image/png')])
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    with pytest.raises(ValueError, match='base64'):
        await embedder.generate(request)
    client_mock._api_client.async_request.assert_not_called()


@pytest.mark.asyncio
async def test_multimodal_embedding_maps_output_dimensionality(mocker: MockerFixture) -> None:
    """The output_dimensionality option maps to the :predict parameters.dimension field."""
    request = EmbedRequest(
        input=[Document.from_media('gs://bucket/cat.png', 'image/png')],
        options={'output_dimensionality': 512},
    )
    predict_body = {'predictions': [{'imageEmbedding': [0.1, 0.2]}]}
    client_mock = mocker.AsyncMock()
    http_response = mocker.Mock()
    http_response.body = json.dumps(predict_body)
    client_mock._api_client.async_request.return_value = http_response

    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    await embedder.generate(request)

    call = client_mock._api_client.async_request.call_args
    assert call.kwargs['request_dict']['parameters'] == {'dimension': 512}


@pytest.mark.asyncio
async def test_multimodal_embedding_omits_parameters_without_dimension(mocker: MockerFixture) -> None:
    """Options without output_dimensionality do not produce a parameters field."""
    request = EmbedRequest(
        input=[Document.from_media('gs://bucket/cat.png', 'image/png')],
        options={'task_type': 'RETRIEVAL_QUERY'},
    )
    predict_body = {'predictions': [{'imageEmbedding': [0.1, 0.2]}]}
    client_mock = mocker.AsyncMock()
    http_response = mocker.Mock()
    http_response.body = json.dumps(predict_body)
    client_mock._api_client.async_request.return_value = http_response

    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    await embedder.generate(request)

    call = client_mock._api_client.async_request.call_args
    assert 'parameters' not in call.kwargs['request_dict']


@pytest.mark.asyncio
async def test_multimodal_embedding_maps_video_segment_config(mocker: MockerFixture) -> None:
    """Document metadata video_segment_config is forwarded as the instance videoSegmentConfig."""
    segment_config = {'startOffsetSec': 0, 'endOffsetSec': 10, 'intervalSec': 5}
    request = EmbedRequest(
        input=[
            Document(
                content=Document.from_media('gs://bucket/clip.mp4', 'video/mp4').content,
                metadata={'video_segment_config': segment_config},
            )
        ]
    )
    predict_body = {
        'predictions': [{'videoEmbeddings': [{'startOffsetSec': 0, 'endOffsetSec': 5, 'embedding': [0.3, 0.4]}]}]
    }
    client_mock = mocker.AsyncMock()
    http_response = mocker.Mock()
    http_response.body = json.dumps(predict_body)
    client_mock._api_client.async_request.return_value = http_response

    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    await embedder.generate(request)

    call = client_mock._api_client.async_request.call_args
    instances = call.kwargs['request_dict']['instances']
    assert instances[0] == {'video': {'gcsUri': 'gs://bucket/clip.mp4', 'videoSegmentConfig': segment_config}}


@pytest.mark.asyncio
async def test_multimodal_embedding_guards_missing_private_transport(mocker: MockerFixture) -> None:
    """A client missing the private _api_client transport fails with an actionable error."""
    request = EmbedRequest(input=[Document.from_media('gs://bucket/cat.png', 'image/png')])
    client_mock = mocker.Mock(spec=[])  # no _api_client attribute at all

    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    with pytest.raises(RuntimeError, match='google-genai>=1.63.0'):
        await embedder.generate(request)


class _ConcurrencyTracker:
    """Records overlap and completion order of mocked embedding requests."""

    def __init__(self, yields: int = 3) -> None:
        """Initialize the tracker.

        Args:
            yields: Default number of times a tracked request yields to the
                event loop before finishing, so siblings can start.
        """
        self.in_flight = 0
        self.max_in_flight = 0
        self.completed: list[int] = []
        self._yields = yields

    async def track(self, index: int, yields: int | None = None) -> None:
        """Hold one request in flight across a few event loop iterations.

        Args:
            index: Index of the batch or document being requested.
            yields: Overrides the default number of yields for this request.
        """
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        for _ in range(self._yields if yields is None else yields):
            await asyncio.sleep(0)
        self.in_flight -= 1
        self.completed.append(index)


def _tracked_embed_content(
    tracker: _ConcurrencyTracker,
    yields_for: Callable[[int], int] | None = None,
    fail_for: Callable[[int], bool] | None = None,
) -> Callable[..., Awaitable[genai.types.EmbedContentResponse]]:
    """Build an embed_content side effect that tracks overlap and can fail."""

    async def side_effect(
        *, model: str, contents: list[genai.types.Content], config: object
    ) -> genai.types.EmbedContentResponse:
        index = int(((contents[0].parts or [])[0].text) or '-1')
        await tracker.track(index, None if yields_for is None else yields_for(index))
        if fail_for is not None and fail_for(index):
            raise RuntimeError(f'batch {index} failed')
        return _indexed_embed_content(model=model, contents=contents, config=config)

    return side_effect


def _numbered_media_docs(count: int) -> list[Document]:
    """Documents whose gcsUri carries their index, for the multimodal path."""
    return [Document.from_media(f'gs://bucket/{i}.png', 'image/png') for i in range(count)]


def _instance_index(request_dict: dict[str, Any]) -> int:
    """Document index carried in a single-instance :predict payload's gcsUri."""
    uri = str(request_dict['instances'][0]['image']['gcsUri'])
    return int(uri.rsplit('/', 1)[1].removesuffix('.png'))


def _tracked_async_request(
    tracker: _ConcurrencyTracker,
    mocker: MockerFixture,
    yields_for: Callable[[int], int] | None = None,
    fail_for: Callable[[int], bool] | None = None,
) -> Callable[..., Awaitable[object]]:
    """Build a :predict side effect that tracks overlap and can fail."""

    async def side_effect(*, http_method: str, path: str, request_dict: dict[str, Any]) -> object:
        index = _instance_index(request_dict)
        await tracker.track(index, None if yields_for is None else yields_for(index))
        if fail_for is not None and fail_for(index):
            raise RuntimeError(f'document {index} failed')
        http_response = mocker.Mock()
        http_response.body = json.dumps({'predictions': [{'imageEmbedding': [float(index)]}]})
        return http_response

    return side_effect


@pytest.mark.asyncio
async def test_text_embedding_batches_run_concurrently(mocker: MockerFixture) -> None:
    """Text embedding batches are in flight at the same time, not one after another."""
    count = 5
    tracker = _ConcurrencyTracker()
    client = mocker.AsyncMock()
    client.aio.models.embed_content.side_effect = _tracked_embed_content(tracker)
    embedder = Embedder('gemini-embedding-001', client, is_vertex=True)

    response = await embedder.generate(EmbedRequest(input=_numbered_docs(count)))

    assert client.aio.models.embed_content.call_count == count
    assert tracker.max_in_flight > 1
    assert [e.embedding for e in response.embeddings] == [[float(i)] for i in range(count)]


@pytest.mark.asyncio
async def test_text_embedding_concurrency_is_capped(mocker: MockerFixture) -> None:
    """No more than EMBED_CONCURRENCY_LIMIT text batches are in flight at once."""
    count = EMBED_CONCURRENCY_LIMIT * 3
    tracker = _ConcurrencyTracker()
    client = mocker.AsyncMock()
    client.aio.models.embed_content.side_effect = _tracked_embed_content(tracker)
    embedder = Embedder('gemini-embedding-001', client, is_vertex=True)

    response = await embedder.generate(EmbedRequest(input=_numbered_docs(count)))

    assert client.aio.models.embed_content.call_count == count
    assert tracker.max_in_flight == EMBED_CONCURRENCY_LIMIT
    assert [e.embedding for e in response.embeddings] == [[float(i)] for i in range(count)]


@pytest.mark.asyncio
async def test_text_embedding_keeps_input_order_when_batches_finish_first(mocker: MockerFixture) -> None:
    """Embeddings stay in input order when later batches complete before earlier ones."""
    count = 5
    tracker = _ConcurrencyTracker()
    client = mocker.AsyncMock()
    client.aio.models.embed_content.side_effect = _tracked_embed_content(
        tracker, yields_for=lambda index: (count - index) * 2
    )
    embedder = Embedder('gemini-embedding-001', client, is_vertex=True)

    response = await embedder.generate(EmbedRequest(input=_numbered_docs(count)))

    # The mock really did complete back to front, so input order is not
    # completion order here.
    assert tracker.completed == list(reversed(range(count)))
    assert [e.embedding for e in response.embeddings] == [[float(i)] for i in range(count)]


@pytest.mark.asyncio
async def test_text_embedding_failure_leaves_queued_batches_unsent(mocker: MockerFixture) -> None:
    """A failing batch cancels the batches still queued instead of billing them all."""
    count = EMBED_CONCURRENCY_LIMIT * 3
    tracker = _ConcurrencyTracker()
    client = mocker.AsyncMock()
    client.aio.models.embed_content.side_effect = _tracked_embed_content(
        tracker,
        yields_for=lambda index: 0 if index == 0 else 3,
        fail_for=lambda index: index == 0,
    )
    embedder = Embedder('gemini-embedding-001', client, is_vertex=True)

    with pytest.raises(RuntimeError, match='batch 0 failed'):
        await embedder.generate(EmbedRequest(input=_numbered_docs(count)))

    assert client.aio.models.embed_content.call_count < count


@pytest.mark.asyncio
async def test_text_embedding_reports_the_first_failure_in_input_order(mocker: MockerFixture) -> None:
    """The failure raised is the earliest failing batch, not the first one to fail."""
    tracker = _ConcurrencyTracker()
    client = mocker.AsyncMock()
    client.aio.models.embed_content.side_effect = _tracked_embed_content(
        tracker,
        yields_for=lambda index: 1 if index == 0 else 0,
        fail_for=lambda index: index in {0, 2},
    )
    embedder = Embedder('gemini-embedding-001', client, is_vertex=True)

    # Batch 2 fails first, batch 0 fails a loop iteration later.
    with pytest.raises(RuntimeError, match='batch 0 failed'):
        await embedder.generate(EmbedRequest(input=_numbered_docs(3)))


@pytest.mark.asyncio
async def test_multimodal_embedding_requests_run_concurrently(mocker: MockerFixture) -> None:
    """Multimodal :predict requests are in flight at the same time."""
    count = 5
    tracker = _ConcurrencyTracker()
    client_mock = mocker.AsyncMock()
    client_mock._api_client.async_request.side_effect = _tracked_async_request(tracker, mocker)
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)

    response = await embedder.generate(EmbedRequest(input=_numbered_media_docs(count)))

    assert client_mock._api_client.async_request.call_count == count
    assert tracker.max_in_flight > 1
    assert [e.embedding for e in response.embeddings] == [[float(i)] for i in range(count)]


@pytest.mark.asyncio
async def test_multimodal_embedding_concurrency_is_capped(mocker: MockerFixture) -> None:
    """No more than EMBED_CONCURRENCY_LIMIT :predict requests are in flight at once."""
    count = EMBED_CONCURRENCY_LIMIT * 3
    tracker = _ConcurrencyTracker()
    client_mock = mocker.AsyncMock()
    client_mock._api_client.async_request.side_effect = _tracked_async_request(tracker, mocker)
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)

    response = await embedder.generate(EmbedRequest(input=_numbered_media_docs(count)))

    assert client_mock._api_client.async_request.call_count == count
    assert tracker.max_in_flight == EMBED_CONCURRENCY_LIMIT
    assert [e.embedding for e in response.embeddings] == [[float(i)] for i in range(count)]


@pytest.mark.asyncio
async def test_multimodal_embedding_keeps_document_order_when_requests_finish_first(mocker: MockerFixture) -> None:
    """Multimodal embeddings stay in document order when later requests finish first."""
    count = 5
    tracker = _ConcurrencyTracker()
    client_mock = mocker.AsyncMock()
    client_mock._api_client.async_request.side_effect = _tracked_async_request(
        tracker, mocker, yields_for=lambda index: (count - index) * 2
    )
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)

    response = await embedder.generate(EmbedRequest(input=_numbered_media_docs(count)))

    assert tracker.completed == list(reversed(range(count)))
    assert [e.embedding for e in response.embeddings] == [[float(i)] for i in range(count)]


@pytest.mark.asyncio
async def test_multimodal_embedding_failure_leaves_queued_requests_unsent(mocker: MockerFixture) -> None:
    """A failing :predict cancels the documents still queued instead of billing them all."""
    count = EMBED_CONCURRENCY_LIMIT * 3
    tracker = _ConcurrencyTracker()
    client_mock = mocker.AsyncMock()
    client_mock._api_client.async_request.side_effect = _tracked_async_request(
        tracker,
        mocker,
        yields_for=lambda index: 0 if index == 0 else 3,
        fail_for=lambda index: index == 0,
    )
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)

    with pytest.raises(RuntimeError, match='document 0 failed'):
        await embedder.generate(EmbedRequest(input=_numbered_media_docs(count)))

    assert client_mock._api_client.async_request.call_count < count
