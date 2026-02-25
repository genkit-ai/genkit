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

"""Unittests for VertexAI Vector Search retrievers.

Defines tests for all the methods of the DocRetriever
implementations like BigQueryRetriever and FirestoreRetriever.
"""

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.cloud.aiplatform_v1 import (
    FindNeighborsRequest,
    FindNeighborsResponse,
    IndexDatapoint,
    MatchServiceAsyncClient,
    types,
)

from genkit.ai import Genkit
from genkit.blocks.document import Document, DocumentData
from genkit.core.typing import DocumentPart, Embedding
from genkit.plugins.vertex_ai.vector_search import BigQueryRetriever, FirestoreRetriever
from genkit.types import ActionRunContext, RetrieverRequest, TextPart


class FakeAI:
    """Fake AI instance for testing."""

    async def embed(
        self,
        embedder: str,
        content: object | None = None,
        metadata: object | None = None,
        options: object | None = None,
    ) -> list[Embedding]:
        """Mock embed method."""
        return [Embedding(embedding=[0.1, 0.2, 0.3])]


@pytest.fixture
def bq_retriever_instance() -> Any:  # noqa: ANN401
    """Common initialization of bq retriever.

    Note: Returns Any because tests mock methods on this instance,
    breaking type contracts that the type checker cannot represent.
    """
    return BigQueryRetriever(
        ai=cast(Genkit, FakeAI()),
        name='test',
        embedder='test/embedder',
        match_service_client_generator=MagicMock(),
        bq_client=MagicMock(),
        dataset_id='dataset_id',
        table_id='table_id',
    )


def test_bigquery_retriever__init__(bq_retriever_instance: Any) -> None:  # noqa: ANN401
    """Init test."""
    bq_retriever = bq_retriever_instance

    assert bq_retriever is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'options, top_k',
    [
        (
            {'limit': 10},
            10,
        ),
        (
            {},
            3,
        ),
        (
            None,
            3,
        ),
    ],
)
async def test_bigquery_retriever_retrieve(
    bq_retriever_instance: Any,  # noqa: ANN401
    options: dict[str, Any] | None,
    top_k: int,
) -> None:
    """Test retrieve method bq retriever."""
    # Mock _get_closest_documents
    mock__get_closest_documents_result = [
        Document.from_text(
            text='1',
            metadata={'distance': 0.0, 'id': 1},
        ),
        Document.from_text(
            text='2',
            metadata={'distance': 0.0, 'id': 2},
        ),
    ]

    bq_retriever_instance._get_closest_documents = AsyncMock(
        return_value=mock__get_closest_documents_result,
    )

    # Executes
    request = RetrieverRequest(
        query=DocumentData(
            content=[
                DocumentPart(root=TextPart(text='test-1')),
            ],
        ),
        options=options,
    )
    await bq_retriever_instance.retrieve(
        request,
        MagicMock(spec=ActionRunContext),
    )

    # Assert mocks
    bq_retriever_instance._get_closest_documents.assert_awaited_once_with(
        request=request,
        top_k=top_k,
        query_embeddings=Embedding(
            embedding=[0.1, 0.2, 0.3],
        ),
    )


@pytest.mark.asyncio
async def test_bigquery__get_closest_documents(
    bq_retriever_instance: Any,  # noqa: ANN401
) -> None:
    """Test bigquery retriever _get_closest_documents."""
    # Mock find_neighbors method
    mock_vector_search_client = MagicMock(spec=MatchServiceAsyncClient)

    # find_neighbors response
    mock_nn = MagicMock()
    mock_nn.neighbors = []

    mock_nn_response = MagicMock(spec=FindNeighborsResponse)
    mock_nn_response.nearest_neighbors = [
        mock_nn,
    ]

    mock_vector_search_client.find_neighbors = AsyncMock(
        return_value=mock_nn_response,
    )

    # find_neighbors call
    bq_retriever_instance._match_service_client_generator.return_value = mock_vector_search_client

    # Mock _retrieve_neighbors_data_from_db method
    mock__retrieve_neighbors_data_from_db_result = [
        Document.from_text(text='1', metadata={'distance': 0.0, 'id': 1}),
        Document.from_text(text='2', metadata={'distance': 0.0, 'id': 2}),
    ]

    bq_retriever_instance._retrieve_neighbors_data_from_db = AsyncMock(
        return_value=mock__retrieve_neighbors_data_from_db_result,
    )

    await bq_retriever_instance._get_closest_documents(
        request=RetrieverRequest(
            query=DocumentData(
                content=[DocumentPart(root=TextPart(text='test-1'))],
                metadata={
                    'index_endpoint_path': 'index_endpoint_path',
                    'api_endpoint': 'api_endpoint',
                    'deployed_index_id': 'deployed_index_id',
                },
            ),
            options={
                'limit': 10,
            },
        ),
        top_k=10,
        query_embeddings=Embedding(
            embedding=[0.1, 0.2, 0.3],
        ),
    )

    # Assert calls
    mock_vector_search_client.find_neighbors.assert_awaited_once_with(
        request=FindNeighborsRequest(
            index_endpoint='index_endpoint_path',
            deployed_index_id='deployed_index_id',
            queries=[
                FindNeighborsRequest.Query(
                    datapoint=IndexDatapoint(feature_vector=[0.1, 0.2, 0.3]),
                    neighbor_count=10,
                )
            ],
        )
    )

    bq_retriever_instance._retrieve_neighbors_data_from_db.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'metadata',
    [
        {
            'index_endpoint_path': 'index_endpoint_path',
        },
        {
            'api_endpoint': 'api_endpoint',
        },
        {
            'deployed_index_id': 'deployed_index_id',
        },
        {
            'index_endpoint_path': 'index_endpoint_path',
            'api_endpoint': 'api_endpoint',
        },
        {
            'index_endpoint_path': 'index_endpoint_path',
            'deployed_index_id': 'deployed_index_id',
        },
        {
            'api_endpoint': 'api_endpoint',
            'deployed_index_id': 'deployed_index_id',
        },
    ],
)
async def test_bigquery__get_closest_documents_fail(
    bq_retriever_instance: Any,  # noqa: ANN401
    metadata: dict[str, Any],
) -> None:
    """Test failures bigquery retriever _get_closest_documents."""
    with pytest.raises(AttributeError):
        await bq_retriever_instance._get_closest_documents(
            request=RetrieverRequest(
                query=DocumentData(
                    content=[DocumentPart(root=TextPart(text='test-1'))],
                    metadata=metadata,
                ),
                options={
                    'limit': 10,
                },
            ),
            top_k=10,
            query_embeddings=Embedding(
                embedding=[0.1, 0.2, 0.3],
            ),
        )


@pytest.mark.asyncio
async def test_bigquery__retrieve_neighbors_data_from_db(
    bq_retriever_instance: Any,  # noqa: ANN401
) -> None:
    """Test bigquery retriever _retrieve_neighbors_data_from_db."""
    # Mock query job result from bigquery query
    mock_bq_query_job = MagicMock()
    mock_bq_query_job.result.return_value = [
        {
            'id': 'doc1',
            'content': {'body': 'text for document 1'},
        },
        {'id': 'doc2', 'content': json.dumps({'body': 'text for document 2'}), 'metadata': {'date': 'today'}},
        {},  # should error without skipping first two rows
    ]

    bq_retriever_instance.bq_client.query.return_value = mock_bq_query_job

    # call the method
    result = await bq_retriever_instance._retrieve_neighbors_data_from_db(
        neighbors=[
            FindNeighborsResponse.Neighbor(
                datapoint=types.index.IndexDatapoint(datapoint_id='doc1'),
                distance=0.0,
                sparse_distance=0.0,
            ),
            FindNeighborsResponse.Neighbor(
                datapoint=types.index.IndexDatapoint(datapoint_id='doc2'),
                distance=0.0,
                sparse_distance=0.0,
            ),
        ]
    )

    # Assert results and calls
    expected = [
        Document.from_text(
            text=json.dumps(
                {
                    'body': 'text for document 1',
                },
            ),
            metadata={'id': 'doc1', 'distance': 0.0},
        ),
        Document.from_text(
            text=json.dumps(
                {
                    'body': 'text for document 2',
                },
            ),
            metadata={'id': 'doc2', 'distance': 0.0, 'date': 'today'},
        ),
    ]

    assert result == expected

    bq_retriever_instance.bq_client.query.assert_called_once()

    mock_bq_query_job.result.assert_called_once()


@pytest.mark.asyncio
async def test_bigquery_retrieve_neighbors_data_from_db_fail(
    bq_retriever_instance: Any,  # noqa: ANN401
) -> None:
    """Test bigquery retriever _retrieve_neighbors_data_from_db when fails."""
    # Mock exception from bigquery query
    bq_retriever_instance.bq_client.query.raises = AttributeError

    # call the method
    result = await bq_retriever_instance._retrieve_neighbors_data_from_db(
        neighbors=[
            FindNeighborsResponse.Neighbor(
                datapoint=types.index.IndexDatapoint(datapoint_id='doc1'),
                distance=0.0,
                sparse_distance=0.0,
            ),
            FindNeighborsResponse.Neighbor(
                datapoint=types.index.IndexDatapoint(datapoint_id='doc2'),
                distance=0.0,
                sparse_distance=0.0,
            ),
        ]
    )

    assert len(result) == 0

    bq_retriever_instance.bq_client.query.assert_called_once()


@pytest.fixture
def fs_retriever_instance() -> Any:  # noqa: ANN401
    """Common initialization of firestore retriever.

    Note: Returns Any because tests mock methods on this instance,
    breaking type contracts that the type checker cannot represent.
    """
    return FirestoreRetriever(
        ai=cast(Genkit, FakeAI()),
        name='test',
        embedder='test/embedder',
        match_service_client_generator=MagicMock(),
        firestore_client=MagicMock(),
        collection_name='collection_name',
    )


def test_firestore_retriever__init__(fs_retriever_instance: Any) -> None:  # noqa: ANN401
    """Init test."""
    assert fs_retriever_instance is not None


@pytest.mark.asyncio
async def test_firesstore__retrieve_neighbors_data_from_db(
    fs_retriever_instance: Any,  # noqa: ANN401
) -> None:
    """Test _retrieve_neighbors_data_from_db for firestore retriever."""
    # Mock storage of firestore
    storage = {
        'doc1': {
            'content': {'body': 'text for document 1'},
        },
        'doc2': {'content': json.dumps({'body': 'text for document 2'}), 'metadata': {'date': 'today'}},
        'doc3': {},
    }

    # Mock get from firestore
    class MockCollection:
        def document(self, document_id: str) -> MagicMock:
            doc_ref = MagicMock()
            doc_snapshot = MagicMock()

            doc_ref.get = AsyncMock(return_value=doc_snapshot)
            if storage.get(document_id) is not None:
                doc_snapshot.exists = True
                doc_snapshot.to_dict.return_value = storage.get(document_id)
            else:
                doc_snapshot.exists = False

            return doc_ref

    fs_retriever_instance.db.collection.return_value = MockCollection()

    # call the method
    result = await fs_retriever_instance._retrieve_neighbors_data_from_db(
        neighbors=[
            FindNeighborsResponse.Neighbor(
                datapoint=types.index.IndexDatapoint(datapoint_id='doc1'),
                distance=0.0,
                sparse_distance=0.0,
            ),
            FindNeighborsResponse.Neighbor(
                datapoint=types.index.IndexDatapoint(datapoint_id='doc2'),
                distance=0.0,
                sparse_distance=0.0,
            ),
            FindNeighborsResponse.Neighbor(
                datapoint=types.index.IndexDatapoint(datapoint_id='doc3'),
                distance=0.0,
                sparse_distance=0.0,
            ),
        ]
    )

    # Assert results and calls
    expected = [
        Document.from_text(
            text=json.dumps(
                {
                    'body': 'text for document 1',
                },
            ),
            metadata={'id': 'doc1', 'distance': 0.0},
        ),
        Document.from_text(
            text=json.dumps(
                {
                    'body': 'text for document 2',
                },
            ),
            metadata={'id': 'doc2', 'distance': 0.0, 'date': 'today'},
        ),
        Document.from_text(
            text='',
            metadata={
                'id': 'doc3',
                'distance': 0.0,
            },
        ),
    ]

    assert result == expected
