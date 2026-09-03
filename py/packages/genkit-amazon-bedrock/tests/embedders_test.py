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

"""Tests for the Bedrock embedders (no AWS involved)."""

import asyncio
import json
from typing import Any

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from genkit_amazon_bedrock.embedders import (
    COHERE_TEXT_BATCH_SIZE,
    EMBED_CONCURRENCY_LIMIT,
    EMBEDDER_INFO,
    BedrockEmbedder,
    decode_cohere_embeddings,
    document_text,
    get_embedder_info,
    image_from_document,
    is_embedding_model,
)
from genkit_amazon_bedrock.model_info import strip_inference_profile_prefix

from genkit import DocumentPart, Media, MediaPart, TextPart
from genkit._core._typing import DocumentData
from genkit.embedder import EmbedRequest
from genkit.plugin_api import GenkitError

TITAN_TEXT = 'amazon.titan-embed-text-v2:0'
TITAN_MM = 'amazon.titan-embed-image-v1'
COHERE = 'cohere.embed-english-v3'
NOVA = 'amazon.nova-2-multimodal-embeddings-v1:0'

PNG_B64 = 'iVBORw0KGgoAAAANSUhEUg=='
PNG_DATA_URL = f'data:image/png;base64,{PNG_B64}'


class FakeInvokeTransport:
    """Stands in for BedrockTransport; records the InvokeModel kwargs.

    Responses come either from a queue (consumed in call order) or from
    ``dispatch``, which maps a parsed request body to a response so tests that
    fan out do not depend on completion order.
    """

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
        dispatch: Any = None,
    ) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.dispatch = dispatch
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def bodies(self) -> list[dict[str, Any]]:
        """The parsed request bodies, in call order."""
        return [json.loads(call['body']) for call in self.calls]

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            # Yields so concurrent calls actually overlap, which is what makes
            # the observed peak deterministic.
            await asyncio.sleep(0)
            if self.error is not None:
                raise self.error
            if self.dispatch is not None:
                return self.dispatch(json.loads(kwargs['body']))
            return self.responses.pop(0)
        finally:
            self.in_flight -= 1


class ForbiddenTransport:
    """Fails the test if the embedder reaches the wire at all."""

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f'no InvokeModel call expected, got {kwargs}')


def text_doc(*texts: str) -> DocumentData:
    return DocumentData(content=[DocumentPart(root=TextPart(text=text)) for text in texts])


def media_doc(url: str, content_type: str | None = None) -> DocumentData:
    return DocumentData(content=[DocumentPart(root=MediaPart(media=Media(url=url, content_type=content_type)))])


def mixed_doc(text: str, url: str, content_type: str | None = None) -> DocumentData:
    return DocumentData(
        content=[
            DocumentPart(root=TextPart(text=text)),
            DocumentPart(root=MediaPart(media=Media(url=url, content_type=content_type))),
        ]
    )


def titan_response(vector: list[float] | None = None) -> dict[str, Any]:
    return {'embedding': vector if vector is not None else [0.1, 0.2]}


def nova_response(vector: list[float] | None = None) -> dict[str, Any]:
    return {'embeddings': [{'embedding': vector if vector is not None else [0.1, 0.2, 0.3]}]}


def cohere_response(count: int) -> dict[str, Any]:
    return {'embeddings': {'float': [[float(i), 0.5] for i in range(count)]}}


async def embed(model_id: str, transport: Any, documents: list[DocumentData], **kwargs: Any) -> list[list[float]]:
    embedder = BedrockEmbedder(model_id=model_id, transport=transport)
    response = await embedder.embed(EmbedRequest(input=documents, **kwargs))
    return [embedding.embedding for embedding in response.embeddings]


# ---- Routing and validation -------------------------------------------------


@pytest.mark.asyncio
async def test_empty_input_is_rejected() -> None:
    with pytest.raises(GenkitError, match='no documents') as excinfo:
        await embed(TITAN_TEXT, ForbiddenTransport(), [])
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.parametrize(
    ('model_id', 'routable'),
    [
        ('amazon.titan-embed-text-v1', True),
        ('amazon.titan-embed-text-v2:0', True),
        ('amazon.titan-embed-image-v1', True),
        ('cohere.embed-english-v3', True),
        ('cohere.embed-multilingual-v3', True),
        ('cohere.embed-v4:0', True),
        ('amazon.nova-2-multimodal-embeddings-v1:0', True),
        ('us.amazon.titan-embed-text-v2:0', True),
        # Bare 'cohere' routing would swallow these two.
        ('cohere.rerank-v3-5:0', False),
        ('cohere.command-r-v1:0', False),
        ('anthropic.claude-sonnet-4-5-20250929-v1:0', False),
        ('amazon.nova-lite-v1:0', False),
        ('amazon.titan-image-generator-v1', False),
    ],
)
def test_embedding_model_routing(model_id: str, routable: bool) -> None:
    assert is_embedding_model(model_id) is routable


@pytest.mark.asyncio
async def test_profile_prefixed_id_routes_but_the_wire_keeps_it() -> None:
    transport = FakeInvokeTransport([titan_response()])
    await embed(f'us.{TITAN_TEXT}', transport, [text_doc('hi')])
    assert transport.calls[0]['modelId'] == f'us.{TITAN_TEXT}'


@pytest.mark.asyncio
async def test_unroutable_model_fails_as_unimplemented() -> None:
    with pytest.raises(GenkitError, match='unsupported embedding model') as excinfo:
        await embed('anthropic.claude-sonnet-4-5-20250929-v1:0', ForbiddenTransport(), [text_doc('hi')])
    assert excinfo.value.status == 'UNIMPLEMENTED'


@pytest.mark.asyncio
async def test_cohere_v4_is_unimplemented_without_calling() -> None:
    transport = FakeInvokeTransport()
    with pytest.raises(GenkitError, match='Cohere Embed v4') as excinfo:
        await embed('cohere.embed-v4:0', transport, [text_doc('hi')])
    assert excinfo.value.status == 'UNIMPLEMENTED'
    assert transport.calls == []


@pytest.mark.parametrize('model_id', [TITAN_TEXT, TITAN_MM, COHERE, NOVA])
@pytest.mark.asyncio
async def test_empty_document_fails_before_any_call(model_id: str) -> None:
    with pytest.raises(GenkitError, match='document 0 has no') as excinfo:
        await embed(model_id, ForbiddenTransport(), [text_doc('   ')])
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_mixed_validity_names_the_first_bad_document_and_calls_nothing() -> None:
    # Upfront validation: a bad batch costs nothing rather than paying for
    # every document that happens to be fine.
    transport = FakeInvokeTransport()
    with pytest.raises(GenkitError, match='document 2 has no text content'):
        await embed(TITAN_TEXT, transport, [text_doc('a'), text_doc('b'), text_doc(''), text_doc('')])
    assert transport.calls == []


# ---- Titan text -------------------------------------------------------------


@pytest.mark.asyncio
async def test_titan_text_sends_the_documented_body() -> None:
    transport = FakeInvokeTransport([titan_response([1.0, 2.0])])
    vectors = await embed(TITAN_TEXT, transport, [text_doc('hello')])

    assert vectors == [[1.0, 2.0]]
    call = transport.calls[0]
    assert call['modelId'] == TITAN_TEXT
    assert call['contentType'] == 'application/json'
    assert call['accept'] == 'application/json'
    # Exactly one key: Titan text rejects anything else.
    assert json.loads(call['body']) == {'inputText': 'hello'}


@pytest.mark.asyncio
async def test_titan_text_embeds_each_document_in_order() -> None:
    transport = FakeInvokeTransport(dispatch=lambda body: titan_response([float(len(body['inputText']))]))
    vectors = await embed(TITAN_TEXT, transport, [text_doc('a'), text_doc('bb'), text_doc('ccc')])

    assert vectors == [[1.0], [2.0], [3.0]]
    assert len(transport.calls) == 3


@pytest.mark.asyncio
async def test_titan_text_empty_vector_is_internal() -> None:
    transport = FakeInvokeTransport([{'embedding': []}])
    with pytest.raises(GenkitError, match='empty embedding vector') as excinfo:
        await embed(TITAN_TEXT, transport, [text_doc('hi')])
    assert excinfo.value.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_runtime_errors_name_the_document_without_repeating_the_prefix() -> None:
    transport = FakeInvokeTransport(dispatch=lambda body: titan_response([] if body['inputText'] == 'bad' else [1.0]))
    with pytest.raises(GenkitError, match='document 1: model returned an empty embedding vector'):
        await embed(TITAN_TEXT, transport, [text_doc('ok'), text_doc('bad')])


@pytest.mark.parametrize(
    ('code', 'status'),
    [
        ('ThrottlingException', 'RESOURCE_EXHAUSTED'),
        ('ValidationException', 'INVALID_ARGUMENT'),
        ('AccessDeniedException', 'PERMISSION_DENIED'),
        ('SomethingNewException', 'UNKNOWN'),
    ],
)
@pytest.mark.asyncio
async def test_client_errors_map_to_genkit_statuses(code: str, status: str) -> None:
    error = ClientError({'Error': {'Code': code, 'Message': 'boom'}}, 'InvokeModel')
    transport = FakeInvokeTransport(error=error)
    with pytest.raises(GenkitError, match='invoke model failed') as excinfo:
        await embed(TITAN_TEXT, transport, [text_doc('hi')])
    assert excinfo.value.status == status


@pytest.mark.asyncio
async def test_botocore_errors_map_to_genkit_statuses() -> None:
    transport = FakeInvokeTransport(error=NoCredentialsError())
    with pytest.raises(GenkitError, match='invoke model failed') as excinfo:
        await embed(TITAN_TEXT, transport, [text_doc('hi')])
    assert excinfo.value.status == 'UNAUTHENTICATED'


# ---- Titan multimodal -------------------------------------------------------


@pytest.mark.asyncio
async def test_titan_multimodal_text_only_omits_the_image_key() -> None:
    transport = FakeInvokeTransport([titan_response()])
    await embed(TITAN_MM, transport, [text_doc('hello')])
    assert transport.bodies() == [{'inputText': 'hello'}]


@pytest.mark.asyncio
async def test_titan_multimodal_image_only_sends_raw_base64() -> None:
    transport = FakeInvokeTransport([titan_response()])
    await embed(TITAN_MM, transport, [media_doc(PNG_DATA_URL)])
    # Raw base64, not a data URI: that polarity is Cohere's, not Titan's.
    assert transport.bodies() == [{'inputImage': PNG_B64}]


@pytest.mark.asyncio
async def test_titan_multimodal_sends_text_and_image_in_one_call() -> None:
    transport = FakeInvokeTransport([titan_response()])
    await embed(TITAN_MM, transport, [mixed_doc('hello', PNG_DATA_URL)])
    assert transport.bodies() == [{'inputText': 'hello', 'inputImage': PNG_B64}]
    assert len(transport.calls) == 1


@pytest.mark.parametrize('mime', ['image/jpeg', 'image/jpg', 'image/png'])
@pytest.mark.asyncio
async def test_titan_multimodal_accepts_its_supported_mime_types(mime: str) -> None:
    transport = FakeInvokeTransport([titan_response()])
    await embed(TITAN_MM, transport, [media_doc(f'data:{mime};base64,{PNG_B64}')])
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_titan_multimodal_rejects_unsupported_mime_before_calling() -> None:
    transport = FakeInvokeTransport()
    with pytest.raises(GenkitError, match='not supported by Titan') as excinfo:
        await embed(TITAN_MM, transport, [media_doc(f'data:image/webp;base64,{PNG_B64}')])
    assert excinfo.value.status == 'INVALID_ARGUMENT'
    assert transport.calls == []


@pytest.mark.asyncio
async def test_titan_multimodal_names_the_document_holding_a_remote_url() -> None:
    documents = [media_doc(PNG_DATA_URL), media_doc('https://example.com/i.png', 'image/png')]
    with pytest.raises(GenkitError, match='document 1: remote URLs are not supported') as excinfo:
        await embed(TITAN_MM, ForbiddenTransport(), documents)
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_titan_multimodal_requires_some_content() -> None:
    with pytest.raises(GenkitError, match='no text or image content') as excinfo:
        await embed(TITAN_MM, ForbiddenTransport(), [text_doc('')])
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_titan_multimodal_raises_the_in_band_message() -> None:
    # Titan multimodal reports input problems on an HTTP 200, so dropping this
    # field surfaces "empty embedding vector" instead.
    transport = FakeInvokeTransport([{'message': 'image too large'}])
    with pytest.raises(GenkitError, match='image too large') as excinfo:
        await embed(TITAN_MM, transport, [text_doc('hi')])
    assert excinfo.value.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_titan_multimodal_success_carries_no_message() -> None:
    transport = FakeInvokeTransport([{'embedding': [1.0], 'message': None}])
    assert await embed(TITAN_MM, transport, [text_doc('hi')]) == [[1.0]]


# ---- Cohere -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_cohere_batches_texts_into_one_call() -> None:
    transport = FakeInvokeTransport([cohere_response(2)])
    vectors = await embed(COHERE, transport, [text_doc('a'), text_doc('b')])

    assert vectors == [[0.0, 0.5], [1.0, 0.5]]
    assert transport.bodies() == [
        {
            'texts': ['a', 'b'],
            'input_type': 'search_document',
            'truncate': 'END',
            'embedding_types': ['float'],
        }
    ]


@pytest.mark.asyncio
async def test_cohere_chunks_above_the_batch_limit() -> None:
    documents = [text_doc(f'doc {i}') for i in range(100)]
    transport = FakeInvokeTransport(dispatch=lambda body: cohere_response(len(body['texts'])))
    vectors = await embed(COHERE, transport, documents)

    # Chunks go out concurrently, so sort rather than depend on call order.
    chunks = sorted((body['texts'] for body in transport.bodies()), key=len, reverse=True)
    assert [len(chunk) for chunk in chunks] == [COHERE_TEXT_BATCH_SIZE, 4]
    assert chunks[1] == ['doc 96', 'doc 97', 'doc 98', 'doc 99']
    assert transport.max_in_flight == 2
    # Reassembled by original index, so the second chunk restarts at 0.0.
    assert vectors[96] == [0.0, 0.5]
    assert len(vectors) == 100


@pytest.mark.asyncio
async def test_cohere_arity_mismatch_is_internal() -> None:
    transport = FakeInvokeTransport([cohere_response(1)])
    with pytest.raises(GenkitError, match='returned 1 text embeddings for 2 inputs') as excinfo:
        await embed(COHERE, transport, [text_doc('a'), text_doc('b')])
    assert excinfo.value.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_a_failing_cohere_chunk_names_its_first_document() -> None:
    documents = [text_doc(f'doc {i}') for i in range(100)]
    # The second chunk holds documents 96-99 and comes back one embedding short.
    transport = FakeInvokeTransport(
        dispatch=lambda body: cohere_response(1 if len(body['texts']) == 4 else len(body['texts']))
    )
    with pytest.raises(GenkitError, match='document 96: cohere returned 1 text embeddings for 4 inputs'):
        await embed(COHERE, transport, documents)


@pytest.mark.asyncio
async def test_cohere_rejects_an_image_only_document_and_names_the_alternative() -> None:
    # Bedrock reports inputModalities [TEXT] for both Cohere v3 IDs, so an image
    # request fails with "Invalid parameter combination" however it is encoded.
    with pytest.raises(GenkitError, match='text-only.*titan-embed-image-v1') as excinfo:
        await embed(COHERE, ForbiddenTransport(), [media_doc(PNG_DATA_URL)])
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_cohere_ignores_a_media_part_alongside_text() -> None:
    transport = FakeInvokeTransport([cohere_response(1)])
    await embed(COHERE, transport, [mixed_doc('hello', PNG_DATA_URL)])
    # Only the text travels, and nothing image-shaped reaches the body.
    assert transport.bodies() == [
        {
            'texts': ['hello'],
            'input_type': 'search_document',
            'truncate': 'END',
            'embedding_types': ['float'],
        }
    ]


def test_decode_cohere_typed_shape() -> None:
    assert decode_cohere_embeddings({'embeddings': {'float': [[1.0, 2.0]]}}) == [[1.0, 2.0]]


def test_decode_cohere_flat_shape() -> None:
    assert decode_cohere_embeddings({'embeddings': [[1.0, 2.0]]}) == [[1.0, 2.0]]


def test_decode_cohere_coerces_integers() -> None:
    assert decode_cohere_embeddings({'embeddings': [[1, 2]]}) == [[1.0, 2.0]]


@pytest.mark.parametrize(
    ('payload', 'match'),
    [
        ({}, 'missing embeddings field'),
        ({'embeddings': None}, 'missing embeddings field'),
        ({'embeddings': {'int8': [[1]]}}, 'no float embeddings'),
        ({'embeddings': {'float': []}}, 'no float embeddings'),
        ({'embeddings': []}, 'returned no embeddings'),
    ],
)
def test_decode_cohere_rejects_unusable_payloads(payload: dict[str, Any], match: str) -> None:
    with pytest.raises(GenkitError, match=match) as excinfo:
        decode_cohere_embeddings(payload)
    assert excinfo.value.status == 'INTERNAL'


# ---- Nova-2 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_nova_sends_the_single_embedding_body() -> None:
    transport = FakeInvokeTransport([nova_response()])
    await embed(NOVA, transport, [text_doc('hello')])

    assert transport.bodies() == [
        {
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': {
                'embeddingPurpose': 'GENERIC_INDEX',
                'text': {'truncationMode': 'END', 'value': 'hello'},
            },
        }
    ]


@pytest.mark.asyncio
async def test_nova_reads_the_vector_off_the_embeddings_list() -> None:
    # A list of objects, not the bare vector Titan returns.
    transport = FakeInvokeTransport([nova_response([0.5, 0.6])])
    assert await embed(NOVA, transport, [text_doc('hi')]) == [[0.5, 0.6]]


@pytest.mark.asyncio
async def test_nova_embeds_each_document_in_order() -> None:
    transport = FakeInvokeTransport(
        dispatch=lambda body: nova_response([float(len(body['singleEmbeddingParams']['text']['value']))])
    )
    assert await embed(NOVA, transport, [text_doc('a'), text_doc('bb')]) == [[1.0], [2.0]]
    assert len(transport.calls) == 2


@pytest.mark.parametrize('payload', [{'embeddings': []}, {}, {'embeddings': [{}]}])
@pytest.mark.asyncio
async def test_nova_missing_embeddings_is_internal(payload: dict[str, Any]) -> None:
    transport = FakeInvokeTransport([payload])
    with pytest.raises(GenkitError, match='empty embedding vector') as excinfo:
        await embed(NOVA, transport, [text_doc('hi')])
    assert excinfo.value.status == 'INTERNAL'


# ---- Helpers ----------------------------------------------------------------


def test_document_text_joins_parts_with_newlines() -> None:
    assert document_text(text_doc('a', 'b')) == 'a\nb'


def test_document_text_skips_whitespace_only_parts() -> None:
    assert document_text(text_doc('a', '   ', 'b')) == 'a\nb'


def test_document_text_keeps_inner_padding_and_strips_only_the_result() -> None:
    # Parts are appended untrimmed; only the joined string is stripped.
    assert document_text(text_doc('  a  ', '  b  ')) == 'a  \n  b'


def test_document_text_ignores_media_parts() -> None:
    assert document_text(media_doc(PNG_DATA_URL)) == ''


@pytest.mark.parametrize(
    ('document', 'expected'),
    [
        (media_doc(PNG_DATA_URL), ('image/png', PNG_B64)),
        (media_doc(f'data:image/jpeg;base64,{PNG_B64}'), ('image/jpeg', PNG_B64)),
        # An explicit content type wins over the data-URL header.
        (media_doc(PNG_DATA_URL, 'image/jpeg'), ('image/jpeg', PNG_B64)),
        # Parameters are stripped and the type is lower-cased.
        (media_doc(PNG_DATA_URL, 'IMAGE/PNG; charset=binary'), ('image/png', PNG_B64)),
        # Bare base64, the other shape converters accepts.
        (media_doc(PNG_B64, 'image/png'), ('image/png', PNG_B64)),
        # A data URL with no payload to send is skipped, comma or not.
        (media_doc('data:image/png;base64'), ('', '')),
        (media_doc('data:image/png;base64,'), ('', '')),
        (media_doc('', 'image/png'), ('', '')),
        (media_doc(f'data:application/pdf;base64,{PNG_B64}'), ('', '')),
        # No content type and no data-URL header, so the type is unknown.
        (media_doc(PNG_B64), ('', '')),
        # Remote URLs only raise once the part is known to be an image.
        (media_doc('https://example.com/notes.pdf', 'application/pdf'), ('', '')),
        (text_doc('no media here'), ('', '')),
        (DocumentData(content=[]), ('', '')),
    ],
)
def test_image_from_document(document: DocumentData, expected: tuple[str, str]) -> None:
    assert image_from_document(document) == expected


@pytest.mark.parametrize(
    'url',
    [
        'https://example.com/i.png',
        'http://example.com/i.png',
        's3://bucket/i.png',
        # A comma in a remote URL used to pass for a data-URL separator, which
        # sent the fragment after it as the image payload.
        'https://cdn.example.com/i.png?tags=a,b',
        's3://bucket/a,b/i.png',
    ],
)
def test_image_from_document_rejects_remote_urls(url: str) -> None:
    with pytest.raises(GenkitError, match='remote URLs are not supported') as excinfo:
        image_from_document(media_doc(url, 'image/png'))
    assert excinfo.value.status == 'INVALID_ARGUMENT'


def test_image_from_document_returns_the_first_image() -> None:
    document = DocumentData(
        content=[
            DocumentPart(root=MediaPart(media=Media(url=f'data:application/pdf;base64,{PNG_B64}'))),
            DocumentPart(root=MediaPart(media=Media(url='data:image/png;base64,FIRST'))),
            DocumentPart(root=MediaPart(media=Media(url='data:image/png;base64,SECOND'))),
        ]
    )
    assert image_from_document(document) == ('image/png', 'FIRST')


# ---- Concurrency ------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_respects_the_concurrency_cap() -> None:
    documents = [text_doc(f'doc {i}') for i in range(25)]
    transport = FakeInvokeTransport(dispatch=lambda _body: titan_response([1.0]))
    await embed(TITAN_TEXT, transport, documents)

    assert len(transport.calls) == len(documents)
    # Exactly the cap, not merely under it: `<=` also passes if the fan-out
    # silently went sequential. gather queues every task in one loop iteration
    # and Semaphore.acquire does not yield while permits remain, so the peak is
    # deterministically min(len(documents), cap).
    assert transport.max_in_flight == EMBED_CONCURRENCY_LIMIT


@pytest.mark.asyncio
async def test_a_failing_batch_cancels_the_calls_that_have_not_started() -> None:
    documents = [text_doc('bad')] + [text_doc(f'doc {i}') for i in range(49)]
    transport = FakeInvokeTransport(dispatch=lambda body: titan_response([] if body['inputText'] == 'bad' else [1.0]))
    with pytest.raises(GenkitError, match='document 0'):
        await embed(TITAN_TEXT, transport, documents)

    # Without cancellation the semaphore drains and every remaining document
    # still bills a call.
    assert len(transport.calls) < len(documents)


def test_the_semaphore_is_built_per_call_not_per_module() -> None:
    # A module-level semaphore binds to the first loop that has to wait on it,
    # and the Dev UI reflection server runs a second one. This needs more
    # documents than the cap: at or below it no waiter is created, so nothing
    # binds and the regression escapes.
    documents = [text_doc(f'doc {i}') for i in range(EMBED_CONCURRENCY_LIMIT + 1)]

    def run_on_a_fresh_loop() -> None:
        transport = FakeInvokeTransport(dispatch=lambda _body: titan_response([1.0]))
        asyncio.run(embed(TITAN_TEXT, transport, documents))

    run_on_a_fresh_loop()
    run_on_a_fresh_loop()


@pytest.mark.asyncio
async def test_results_keep_input_order_when_calls_finish_out_of_order() -> None:
    delays = {'a': 0.03, 'b': 0.02, 'c': 0.01}

    class ReversingTransport:
        async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
            text = json.loads(kwargs['body'])['inputText']
            await asyncio.sleep(delays[text])
            return titan_response([float(ord(text))])

    vectors = await embed(TITAN_TEXT, ReversingTransport(), [text_doc('a'), text_doc('b'), text_doc('c')])
    assert vectors == [[97.0], [98.0], [99.0]]


@pytest.mark.asyncio
async def test_the_first_failure_to_land_wins_and_the_slower_one_is_cancelled() -> None:
    # The first error cancels the rest, so the earlier document's slower
    # failure never gets reported.
    class RacingTransport:
        async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
            text = json.loads(kwargs['body'])['inputText']
            if text == 'slow-bad':
                await asyncio.sleep(0.02)
            raise ClientError({'Error': {'Code': 'ValidationException', 'Message': text}}, 'InvokeModel')

    with pytest.raises(GenkitError, match='fast-bad'):
        await embed(TITAN_TEXT, RacingTransport(), [text_doc('slow-bad'), text_doc('fast-bad')])


@pytest.mark.asyncio
async def test_request_options_are_ignored() -> None:
    # No per-request config surface yet; input_type stays pinned.
    transport = FakeInvokeTransport([cohere_response(1)])
    await embed(COHERE, transport, [text_doc('hi')], options={'dimensions': 256, 'input_type': 'search_query'})
    assert transport.bodies()[0] == {
        'texts': ['hi'],
        'input_type': 'search_document',
        'truncate': 'END',
        'embedding_types': ['float'],
    }


# ---- Registry ---------------------------------------------------------------


@pytest.mark.parametrize(
    ('model_id', 'dimensions', 'inputs'),
    [
        ('amazon.titan-embed-text-v1', 1536, ['text']),
        ('amazon.titan-embed-text-v2:0', 1024, ['text']),
        ('amazon.titan-embed-image-v1', 1024, ['text', 'image']),
        ('cohere.embed-english-v3', 1024, ['text']),
        ('cohere.embed-multilingual-v3', 1024, ['text']),
        ('amazon.nova-2-multimodal-embeddings-v1:0', 3072, ['text']),
    ],
)
def test_registry_dimensions_and_modalities(model_id: str, dimensions: int, inputs: list[str]) -> None:
    options = get_embedder_info(model_id)
    assert options.dimensions == dimensions
    assert options.supports is not None and options.supports.input == inputs
    assert options.label == f'Amazon Bedrock - {model_id}'


@pytest.mark.parametrize(
    ('model_id', 'inputs'),
    [
        ('amazon.titan-embed-image-v9', ['text', 'image']),
        ('amazon.titan-embed-text-v9:0', ['text']),
        # A future Cohere v3 point release must not inherit image support.
        ('cohere.embed-english-v3-5', ['text']),
        ('amazon.nova-2-multimodal-embeddings-v9:0', ['text']),
    ],
)
def test_unregistered_but_routable_model_falls_back_to_family_defaults(model_id: str, inputs: list[str]) -> None:
    options = get_embedder_info(model_id)
    assert options.dimensions is None
    assert options.supports is not None and options.supports.input == inputs


def test_profile_prefixed_lookup_strips_but_the_label_keeps_the_prefix() -> None:
    options = get_embedder_info(f'us.{TITAN_TEXT}')
    assert options.dimensions == 1024
    assert options.label == f'Amazon Bedrock - us.{TITAN_TEXT}'


def test_all_registry_keys_are_base_ids() -> None:
    for model_id in EMBEDDER_INFO:
        assert strip_inference_profile_prefix(model_id) == model_id
