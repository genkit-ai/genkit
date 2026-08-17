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

"""Tests for the Bedrock rerank helper (no AWS involved)."""

import json
from typing import Any, cast

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from genkit_amazon_bedrock import Bedrock
from genkit_amazon_bedrock.rerank import (
    BedrockReranker,
    BedrockRerankOptions,
    RerankerRequest,
    RerankerResponse,
    build_rerank_body,
    build_rerank_response,
    coerce_rerank_options,
    is_rerank_model,
)
from genkit_amazon_bedrock.transport import BedrockTransport

from genkit import Document, DocumentPart, TextPart
from genkit._core._typing import DocumentData
from genkit.plugin_api import GenkitError

COHERE_RERANK = 'cohere.rerank-v3-5:0'
AMAZON_RERANK = 'amazon.rerank-v1:0'


class FakeInvokeTransport:
    """Stands in for BedrockTransport; records the InvokeModel kwargs."""

    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.response = response if response is not None else {}
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def bodies(self) -> list[dict[str, Any]]:
        """The parsed request bodies, in call order."""
        return [json.loads(call['body']) for call in self.calls]

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class ForbiddenTransport:
    """Fails the test if the reranker reaches the wire at all."""

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f'no InvokeModel call expected, got {kwargs}')


def text_doc(*texts: str, metadata: dict[str, Any] | None = None) -> DocumentData:
    return DocumentData(
        content=[DocumentPart(root=TextPart(text=text)) for text in texts],
        metadata=metadata,
    )


def scored(*results: tuple[int, float]) -> dict[str, Any]:
    """A rerank response body, in the order the service returned it."""
    return {'results': [{'index': index, 'relevance_score': score} for index, score in results]}


def ranked_texts(response: RerankerResponse) -> list[str]:
    """The text of each ranked document, in ranked order."""
    return [
        '\n'.join(part.root.text for part in document.content if isinstance(part.root, TextPart))
        for document in response.documents
    ]


def scores(response: RerankerResponse) -> list[float]:
    return [document.metadata.score for document in response.documents]


async def rerank(
    model_id: str,
    transport: Any,  # noqa: ANN401
    query: DocumentData,
    documents: list[DocumentData],
    options: Any = None,  # noqa: ANN401
) -> RerankerResponse:
    reranker = BedrockReranker(model_id=model_id, transport=transport)
    return await reranker.rerank(RerankerRequest(query=query, documents=documents, options=options))


# ---- Routing ----------------------------------------------------------------


@pytest.mark.parametrize(
    ('model_id', 'routable'),
    [
        (COHERE_RERANK, True),
        (AMAZON_RERANK, True),
        # Inference-profile prefixes and full foundation-model ARNs still route
        # as rerank models.
        ('us.amazon.rerank-v1:0', True),
        ('eu.cohere.rerank-v3-5:0', True),
        ('arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0', True),
        ('arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0', True),
        # Bare 'cohere' routing would swallow these two.
        ('cohere.embed-english-v3', False),
        ('cohere.command-r-v1:0', False),
        ('anthropic.claude-sonnet-4-5-20250929-v1:0', False),
    ],
)
def test_rerank_model_routing(model_id: str, routable: bool) -> None:
    assert is_rerank_model(model_id) is routable


# ---- Request body -----------------------------------------------------------


@pytest.mark.asyncio
async def test_cohere_rerank_sends_the_documented_body_and_maps_scores() -> None:
    transport = FakeInvokeTransport(scored((1, 0.94), (0, 0.42)))
    response = await rerank(
        COHERE_RERANK,
        transport,
        text_doc('query text'),
        [
            text_doc('first document', metadata={'id': 'first'}),
            text_doc('second document', metadata={'id': 'second'}),
        ],
        options=BedrockRerankOptions(top_n=1),
    )

    call = transport.calls[0]
    assert call['modelId'] == COHERE_RERANK
    assert call['contentType'] == 'application/json'
    assert call['accept'] == 'application/json'
    body = json.loads(call['body'])
    assert body == {
        'query': 'query text',
        'documents': ['first document', 'second document'],
        'top_n': 1,
        'api_version': 2,
    }

    # top_n only caps what the service returns; nothing is truncated here.
    assert len(response.documents) == 2
    assert ranked_texts(response) == ['second document', 'first document']
    assert scores(response) == [0.94, 0.42]
    # The input document's own metadata is deliberately dropped for the score.
    assert response.documents[0].metadata.model_dump() == {'score': 0.94}


@pytest.mark.asyncio
async def test_a_multi_part_query_is_joined_before_it_reaches_the_wire() -> None:
    # document_text itself is covered by embedders_test; this pins the reuse.
    transport = FakeInvokeTransport(scored())
    await rerank(COHERE_RERANK, transport, text_doc('a', 'b'), [text_doc('document')])
    assert transport.bodies()[0]['query'] == 'a\nb'


@pytest.mark.parametrize('model_id', [AMAZON_RERANK, 'us.amazon.rerank-v1:0'])
@pytest.mark.asyncio
async def test_amazon_rerank_sends_a_body_with_no_api_version(model_id: str) -> None:
    transport = FakeInvokeTransport(scored((0, 0.87)))
    response = await rerank(model_id, transport, text_doc('query text'), [text_doc('first document')])

    body = transport.bodies()[0]
    assert body == {'query': 'query text', 'documents': ['first document'], 'top_n': 1}
    # The Amazon schema rejects the key outright, so a null value would fail too.
    assert 'api_version' not in body
    assert scores(response) == [0.87]


@pytest.mark.asyncio
async def test_an_unrecognized_model_id_gets_the_cohere_body() -> None:
    # Not blocked, and Cohere's is the only documented InvokeModel rerank body.
    transport = FakeInvokeTransport(scored())
    await rerank('some.rerankifier-v9:0', transport, text_doc('query'), [text_doc('a')])
    assert transport.bodies()[0] == {
        'query': 'query',
        'documents': ['a'],
        'top_n': 1,
        'api_version': 2,
    }


def test_the_cohere_body_carries_api_version() -> None:
    assert build_rerank_body('q', ['a'], 1, include_api_version=True) == {
        'query': 'q',
        'documents': ['a'],
        'top_n': 1,
        'api_version': 2,
    }


def test_the_amazon_body_omits_api_version_entirely() -> None:
    body = build_rerank_body('q', ['a'], 1, include_api_version=False)
    assert body == {'query': 'q', 'documents': ['a'], 'top_n': 1}
    assert 'api_version' not in body


# ---- top_n ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_n_defaults_to_the_document_count() -> None:
    transport = FakeInvokeTransport(scored())
    await rerank(COHERE_RERANK, transport, text_doc('query'), [text_doc('a'), text_doc('b')])
    assert transport.bodies()[0]['top_n'] == 2


@pytest.mark.asyncio
async def test_top_n_above_the_document_count_is_clamped_down() -> None:
    # A float, not an int: that is what survives a JSON round-trip from a flow.
    transport = FakeInvokeTransport(scored())
    await rerank(
        COHERE_RERANK,
        transport,
        text_doc('query'),
        [text_doc('a'), text_doc('b')],
        options={'topN': 99.0},
    )
    assert transport.bodies()[0]['top_n'] == 2


@pytest.mark.parametrize('top_n', [0, -1])
@pytest.mark.asyncio
async def test_a_non_positive_top_n_asks_for_every_document(top_n: int) -> None:
    transport = FakeInvokeTransport(scored())
    await rerank(
        COHERE_RERANK,
        transport,
        text_doc('query'),
        [text_doc('a'), text_doc('b')],
        options={'topN': top_n},
    )
    assert transport.bodies()[0]['top_n'] == 2


# ---- Validation -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_whitespace_only_query_is_rejected() -> None:
    with pytest.raises(GenkitError, match='query has no text content') as excinfo:
        await rerank(COHERE_RERANK, ForbiddenTransport(), text_doc(' '), [text_doc('document')])
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.parametrize(
    ('texts', 'position'),
    [
        (('',), 0),
        # The index names the original position, not the count of good documents.
        (('good', ''), 1),
    ],
)
@pytest.mark.asyncio
async def test_an_empty_document_names_its_original_position(texts: tuple[str, ...], position: int) -> None:
    documents = [text_doc(text) for text in texts]
    with pytest.raises(GenkitError, match=f'document {position} has no text content') as excinfo:
        await rerank(COHERE_RERANK, ForbiddenTransport(), text_doc('query'), documents)
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_an_unsupported_options_type_is_rejected() -> None:
    with pytest.raises(GenkitError, match='unsupported rerank options type str') as excinfo:
        await rerank(
            COHERE_RERANK,
            ForbiddenTransport(),
            text_doc('query'),
            [text_doc('document')],
            options='bad options',
        )
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_no_documents_returns_early_without_calling() -> None:
    response = await rerank(COHERE_RERANK, ForbiddenTransport(), text_doc('query text'), [])
    assert response.documents == []


@pytest.mark.asyncio
async def test_no_documents_returns_before_the_options_are_coerced() -> None:
    # Nothing to rank outranks bad options; coercing first would raise here.
    response = await rerank(COHERE_RERANK, ForbiddenTransport(), text_doc('query text'), [], options='garbage')
    assert response.documents == []


# ---- Options coercion -------------------------------------------------------


def test_absent_options_coerce_to_nothing() -> None:
    assert coerce_rerank_options(None) is None


def test_typed_options_are_passed_through_unchanged() -> None:
    options = BedrockRerankOptions(top_n=3)
    assert coerce_rerank_options(options) is options


@pytest.mark.parametrize(
    ('options', 'top_n'),
    [
        ({'topN': 7.0}, 7),
        ({'top_n': 7}, 7),
        # Unknown keys are ignored.
        ({'somethingElse': 1}, None),
    ],
)
def test_mapping_options_are_coerced(options: dict[str, Any], top_n: int | None) -> None:
    coerced = coerce_rerank_options(options)
    assert coerced is not None
    assert coerced.top_n == top_n


@pytest.mark.parametrize(
    ('options', 'message'),
    [
        ('nonsense', 'unsupported rerank options type str'),
        ({'topN': 'many'}, 'invalid rerank options'),
        # bool subclasses int, so True would otherwise rank down to one document.
        ({'topN': True}, 'invalid rerank options'),
    ],
)
def test_unusable_options_are_invalid_argument(
    options: Any,  # noqa: ANN401
    message: str,
) -> None:
    with pytest.raises(GenkitError, match=message) as excinfo:
        coerce_rerank_options(options)
    assert excinfo.value.status == 'INVALID_ARGUMENT'


# ---- Response mapping -------------------------------------------------------


@pytest.mark.asyncio
async def test_the_services_order_is_preserved_verbatim() -> None:
    # Ascending scores: the API's order is trusted, never re-sorted.
    transport = FakeInvokeTransport(scored((0, 0.1), (1, 0.9)))
    response = await rerank(COHERE_RERANK, transport, text_doc('query'), [text_doc('a'), text_doc('b')])

    assert ranked_texts(response) == ['a', 'b']
    assert scores(response) == [0.1, 0.9]


@pytest.mark.parametrize('payload', [{'results': []}, {}, {'results': None}])
def test_an_empty_ranking_is_not_an_error(payload: dict[str, Any]) -> None:
    assert build_rerank_response(payload, [text_doc('a')]).documents == []


@pytest.mark.parametrize('index', [1, -1])
def test_an_out_of_range_index_is_internal(index: int) -> None:
    with pytest.raises(GenkitError, match=f'result index {index} out of range') as excinfo:
        build_rerank_response(scored((index, 0.9)), [text_doc('only document')])
    assert excinfo.value.status == 'INTERNAL'


@pytest.mark.parametrize(
    ('payload', 'message'),
    [
        ({'results': 'nope'}, 'response results is not a list'),
        ({'results': ['nope']}, 'result 0 is not an object'),
        ({'results': [{'relevance_score': 0.9}]}, 'result 0 has no integer index'),
        ({'results': [{'index': '0', 'relevance_score': 0.9}]}, 'result 0 has no integer index'),
        # bool subclasses int, so True would otherwise silently rank document 1.
        ({'results': [{'index': True, 'relevance_score': 0.9}]}, 'result 0 has no integer index'),
        ({'results': [{'index': 0}]}, 'result 0 has no numeric relevance_score'),
        ({'results': [{'index': 0, 'relevance_score': 'high'}]}, 'result 0 has no numeric relevance_score'),
        ({'results': [{'index': 0, 'relevance_score': True}]}, 'result 0 has no numeric relevance_score'),
        # The position names the result, not the document it points at.
        ({'results': [{'index': 0, 'relevance_score': 0.9}, 'nope']}, 'result 1 is not an object'),
    ],
)
def test_a_malformed_result_is_internal(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(GenkitError, match=message) as excinfo:
        build_rerank_response(payload, [text_doc('a'), text_doc('b')])
    assert excinfo.value.status == 'INTERNAL'


# ---- AWS failures -----------------------------------------------------------


@pytest.mark.asyncio
async def test_client_errors_map_to_genkit_statuses() -> None:
    error = ClientError({'Error': {'Code': 'ThrottlingException', 'Message': 'boom'}}, 'InvokeModel')
    transport = FakeInvokeTransport(error=error)
    with pytest.raises(GenkitError, match='invoke model failed') as excinfo:
        await rerank(COHERE_RERANK, transport, text_doc('query'), [text_doc('document')])
    assert excinfo.value.status == 'RESOURCE_EXHAUSTED'


@pytest.mark.asyncio
async def test_botocore_errors_map_to_genkit_statuses() -> None:
    transport = FakeInvokeTransport(error=NoCredentialsError())
    with pytest.raises(GenkitError, match='invoke model failed') as excinfo:
        await rerank(COHERE_RERANK, transport, text_doc('query'), [text_doc('document')])
    assert excinfo.value.status == 'UNAUTHENTICATED'


# ---- Plugin helper ----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_plugin_helper_requires_a_model_id() -> None:
    plugin = Bedrock(region='us-east-1')
    plugin._transport = cast(BedrockTransport, ForbiddenTransport())  # noqa: SLF001
    with pytest.raises(GenkitError, match='bedrock rerank: model ID required') as excinfo:
        await plugin.rerank('', query='query', documents=[text_doc('document')])
    assert excinfo.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_the_plugin_helper_delegates_to_the_shared_transport() -> None:
    plugin = Bedrock(region='us-east-1')
    transport = FakeInvokeTransport(scored((0, 0.77)))
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001

    # A plain str query pins the Document.from_text coercion, and a plain dict
    # of options pins the JSON path a flow or the Dev UI would take.
    response = await plugin.rerank(
        COHERE_RERANK,
        query='query text',
        documents=[text_doc('document text')],
        options={'topN': 1},
    )

    assert transport.bodies()[0] == {
        'query': 'query text',
        'documents': ['document text'],
        'top_n': 1,
        'api_version': 2,
    }
    assert ranked_texts(response) == ['document text']
    assert scores(response) == [0.77]


@pytest.mark.asyncio
async def test_the_plugin_helper_accepts_a_document_built_by_the_veneer() -> None:
    plugin = Bedrock(region='us-east-1')
    transport = FakeInvokeTransport(scored((0, 0.77)))
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001

    await plugin.rerank(COHERE_RERANK, query='query text', documents=[Document.from_text('document text')])

    assert transport.bodies()[0]['documents'] == ['document text']
