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
import json

import pytest
from genkit_google_genai._models._embedder import (
    Embedder,
    get_embedder_info,
)
from google import genai
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.genai.errors import APIError
from pytest_mock import MockerFixture

from genkit import Document, GenkitError, Part
from genkit.embedder import EmbedRequest, EmbedResponse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'version',
    ['gemini-embedding-2-preview', 'gemini-embedding-2', 'gemini-embedding-001', 'text-embedding-004'],
)
async def test_embedding(mocker: MockerFixture, version: str) -> None:
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


@pytest.mark.asyncio
async def test_embedding_forwards_media_parts(mocker: MockerFixture) -> None:
    """Multimodal docs forward media parts to the client alongside the text."""
    text = 'a photo'
    raw_image = b'fake-image-bytes'
    data_url = f'data:image/png;base64,{base64.b64encode(raw_image).decode()}'
    embedding_values = [0.0017063986, -0.044727605, 0.043327782, 0.00044852644]

    doc = Document(
        content=[
            Part.from_text(text),
            Part.from_media(data_url, content_type='image/png'),
        ]
    )
    request = EmbedRequest(input=[doc])
    api_response = genai.types.EmbedContentResponse(embeddings=[genai.types.ContentEmbedding(values=embedding_values)])
    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.embed_content.return_value = api_response

    embedder = Embedder('gemini-embedding-2', googleai_client_mock)

    response = await embedder.generate(request)

    googleai_client_mock.assert_has_calls([
        mocker.call.aio.models.embed_content(
            model='gemini-embedding-2',
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
    embedder = Embedder('gemini-embedding-001', googleai_client_mock)
    with pytest.raises(GenkitError, match='Embed request input is empty') as raised:
        await embedder.generate(EmbedRequest(input=[]))
    assert raised.value.status == 'INVALID_ARGUMENT'
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
    with pytest.raises(GenkitError, match='more than one image') as raised:
        await embedder.generate(request)
    assert raised.value.status == 'INVALID_ARGUMENT'
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
    with pytest.raises(GenkitError, match='more than one video') as raised:
        await embedder.generate(request)
    assert raised.value.status == 'INVALID_ARGUMENT'
    client_mock._api_client.async_request.assert_not_called()


@pytest.mark.asyncio
async def test_multimodal_embedding_rejects_http_url(mocker: MockerFixture) -> None:
    """http(s) media URLs are rejected; Vertex gcsUri only accepts gs:// (diverges from JS)."""
    request = EmbedRequest(input=[Document.from_media('https://example.com/cat.png', 'image/png')])
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    with pytest.raises(GenkitError, match='http') as raised:
        await embedder.generate(request)
    assert raised.value.status == 'INVALID_ARGUMENT'
    client_mock._api_client.async_request.assert_not_called()


@pytest.mark.asyncio
async def test_multimodal_embedding_rejects_multiple_documents(mocker: MockerFixture) -> None:
    """multimodalembedding@001 accepts one instance per request, so >1 document is rejected.

    Batching (e.g. embed_many) would otherwise send a multi-instance payload that
    Vertex rejects with an opaque error; fail fast until batching is implemented.
    """
    request = EmbedRequest(
        input=[
            Document.from_media('gs://bucket/a.png', 'image/png'),
            Document.from_media('gs://bucket/b.png', 'image/png'),
        ]
    )
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)
    with pytest.raises(GenkitError, match='one document per request') as raised:
        await embedder.generate(request)
    assert raised.value.status == 'INVALID_ARGUMENT'
    client_mock._api_client.async_request.assert_not_called()


@pytest.mark.asyncio
async def test_multimodal_embedding_rejects_non_vertex_client(mocker: MockerFixture) -> None:
    """Multimodal embedding is Vertex-only; a Gemini API embedder fails fast, before any request."""
    request = EmbedRequest(input=[Document.from_media('gs://bucket/cat.png', 'image/png')])
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding@001', client_mock, is_vertex=False)
    with pytest.raises(GenkitError, match='only available on Vertex AI') as raised:
        await embedder.generate(request)
    assert raised.value.status == 'INVALID_ARGUMENT'
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
    with pytest.raises(GenkitError, match='base64') as raised:
        await embedder.generate(request)
    assert raised.value.status == 'INVALID_ARGUMENT'
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
    with pytest.raises(GenkitError, match='google-genai>=1.63.0') as raised:
        await embedder.generate(request)
    assert raised.value.status == 'FAILED_PRECONDITION'


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('code', 'status'), [(429, 'RESOURCE_EXHAUSTED'), (503, 'UNAVAILABLE'), (400, 'INVALID_ARGUMENT')]
)
async def test_embed_content_api_error_is_classified(mocker: MockerFixture, code: int, status: str) -> None:
    """A quota or overload error while indexing a menu keeps its status so retry can act on it."""
    api_error = APIError(code, {'error': {'message': 'embed failed'}})
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.embed_content.side_effect = api_error
    embedder = Embedder('gemini-embedding-001', client_mock)

    with pytest.raises(GenkitError) as raised:
        await embedder.generate(EmbedRequest(input=[Document.from_text('Smoked salmon tartine')]))

    assert raised.value.status == status
    assert raised.value.cause is api_error


@pytest.mark.asyncio
async def test_embed_content_credential_failure_is_unauthenticated(mocker: MockerFixture) -> None:
    """Missing ADC on a Vertex embed is UNAUTHENTICATED, not a raw google.auth error."""
    no_adc = DefaultCredentialsError('Your default credentials were not found.')
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.embed_content.side_effect = no_adc
    embedder = Embedder('text-embedding-005', client_mock, is_vertex=True)

    with pytest.raises(GenkitError) as raised:
        await embedder.generate(EmbedRequest(input=[Document.from_text('Smoked salmon tartine')]))

    assert raised.value.status == 'UNAUTHENTICATED'
    assert raised.value.cause is no_adc


@pytest.mark.asyncio
async def test_embed_content_unknown_exception_stays_raw(mocker: MockerFixture) -> None:
    """A dropped connection has no known status and reaches the caller unchanged."""
    dropped = ConnectionResetError('Connection reset by peer')
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.embed_content.side_effect = dropped
    embedder = Embedder('gemini-embedding-001', client_mock)

    with pytest.raises(ConnectionResetError) as raised:
        await embedder.generate(EmbedRequest(input=[Document.from_text('Smoked salmon tartine')]))

    assert raised.value is dropped


@pytest.mark.asyncio
async def test_multimodal_embedding_api_error_is_classified(mocker: MockerFixture) -> None:
    """The :predict call classifies provider HTTP errors like embed_content does."""
    api_error = APIError(429, {'error': {'message': 'Quota exceeded'}})
    client_mock = mocker.AsyncMock()
    client_mock._api_client.async_request.side_effect = api_error
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)

    with pytest.raises(GenkitError) as raised:
        await embedder.generate(EmbedRequest(input=[Document.from_media('gs://menu/tartine.png', 'image/png')]))

    assert raised.value.status == 'RESOURCE_EXHAUSTED'
    assert raised.value.cause is api_error


@pytest.mark.asyncio
async def test_multimodal_embedding_credential_failure_is_unauthenticated(mocker: MockerFixture) -> None:
    """A revoked token on :predict is UNAUTHENTICATED."""
    client_mock = mocker.AsyncMock()
    client_mock._api_client.async_request.side_effect = RefreshError('invalid_grant')
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)

    with pytest.raises(GenkitError) as raised:
        await embedder.generate(EmbedRequest(input=[Document.from_media('gs://menu/tartine.png', 'image/png')]))

    assert raised.value.status == 'UNAUTHENTICATED'


@pytest.mark.asyncio
async def test_multimodal_embedding_non_json_body_is_internal(mocker: MockerFixture) -> None:
    """A 200 whose body isn't JSON is a malformed provider response."""
    client_mock = mocker.AsyncMock()
    http_response = mocker.Mock()
    http_response.body = '<html>502 Bad Gateway</html>'
    client_mock._api_client.async_request.return_value = http_response
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)

    with pytest.raises(GenkitError) as raised:
        await embedder.generate(EmbedRequest(input=[Document.from_media('gs://menu/tartine.png', 'image/png')]))

    assert raised.value.status == 'INTERNAL'
    assert isinstance(raised.value.cause, json.JSONDecodeError)


@pytest.mark.asyncio
async def test_multimodal_embedding_rejects_unsupported_content_type(mocker: MockerFixture) -> None:
    """A PDF is caller input the multimodal API cannot take."""
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)

    with pytest.raises(GenkitError, match='Unsupported contentType') as raised:
        await embedder.generate(EmbedRequest(input=[Document.from_media('gs://menu/menu.pdf', 'application/pdf')]))

    assert raised.value.status == 'INVALID_ARGUMENT'
    client_mock._api_client.async_request.assert_not_called()


@pytest.mark.asyncio
async def test_multimodal_embedding_rejects_empty_document(mocker: MockerFixture) -> None:
    """A document with nothing to embed is caller input, rejected before the call."""
    client_mock = mocker.AsyncMock()
    embedder = Embedder('multimodalembedding', client_mock, is_vertex=True)

    with pytest.raises(GenkitError, match='no text, image, or video') as raised:
        await embedder.generate(EmbedRequest(input=[Document(content=[Part.from_text('')])]))

    assert raised.value.status == 'INVALID_ARGUMENT'
