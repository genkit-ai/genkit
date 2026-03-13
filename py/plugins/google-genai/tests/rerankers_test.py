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

"""Tests for Vertex AI Rerankers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genkit.blocks.document import Document
from genkit.core.typing import TextPart
from genkit.plugins.google_genai.rerankers import (
    DEFAULT_MODEL_NAME,
    KNOWN_MODELS,
    VertexRerankerConfig,
    is_reranker_model_name,
)
from genkit.plugins.google_genai.rerankers.reranker import (
    RerankRequest,
    RerankRequestRecord,
    RerankResponse,
    RerankResponseRecord,
    VertexRerankerClientOptions,
    _from_rerank_response,
    _to_reranker_doc,
    get_vertex_rerank_url,
)


def test_default_model_name() -> None:
    """Test that DEFAULT_MODEL_NAME is set correctly."""
    assert DEFAULT_MODEL_NAME == 'semantic-ranker-default@latest'


def test_known_models_contains_expected_models() -> None:
    """Test that KNOWN_MODELS contains expected reranker models."""
    assert 'semantic-ranker-default@latest' in KNOWN_MODELS
    assert 'semantic-ranker-default-004' in KNOWN_MODELS
    assert 'semantic-ranker-fast-004' in KNOWN_MODELS
    assert 'semantic-ranker-default-003' in KNOWN_MODELS
    assert 'semantic-ranker-default-002' in KNOWN_MODELS


def test_is_reranker_model_name_valid() -> None:
    """Test is_reranker_model_name returns True for valid names."""
    assert is_reranker_model_name('semantic-ranker-default@latest') is True
    assert is_reranker_model_name('semantic-ranker-fast-004') is True


def test_is_reranker_model_name_invalid() -> None:
    """Test is_reranker_model_name returns False for invalid names."""
    assert is_reranker_model_name('gemini-2.0-flash') is False
    assert is_reranker_model_name('gemini-embedding-001') is False
    assert is_reranker_model_name(None) is False
    assert is_reranker_model_name('') is False


def test_vertex_reranker_config() -> None:
    """Test VertexRerankerConfig model."""
    config = VertexRerankerConfig(top_n=5)
    assert config.top_n == 5


def test_vertex_reranker_config_defaults() -> None:
    """Test VertexRerankerConfig default values."""
    config = VertexRerankerConfig()
    assert config.top_n is None
    assert config.location is None
    assert config.ignore_record_details_in_response is None


def test_vertex_reranker_config_with_aliases() -> None:
    """Test VertexRerankerConfig works with aliases."""
    # Use Python field names (populate_by_name=True allows both)
    config = VertexRerankerConfig(top_n=10, ignore_record_details_in_response=True)
    assert config.top_n == 10
    assert config.ignore_record_details_in_response is True


def test_vertex_reranker_client_options() -> None:
    """Test VertexRerankerClientOptions model."""
    options = VertexRerankerClientOptions(
        project_id='my-project',
        location='us-central1',
    )
    assert options.project_id == 'my-project'
    assert options.location == 'us-central1'


def test_vertex_reranker_client_options_default_location() -> None:
    """Test VertexRerankerClientOptions uses default location."""
    options = VertexRerankerClientOptions(project_id='test-project')
    assert options.project_id == 'test-project'
    assert options.location == 'global'


def test_rerank_request_record() -> None:
    """Test RerankRequestRecord model."""
    record = RerankRequestRecord(
        id='doc-1',
        title='Test Document',
        content='This is the document content.',
    )
    assert record.id == 'doc-1'
    assert record.title == 'Test Document'
    assert record.content == 'This is the document content.'


def test_rerank_request_record_no_title() -> None:
    """Test RerankRequestRecord without optional title."""
    record = RerankRequestRecord(id='1', content='Content only')
    assert record.id == '1'
    assert record.title is None
    assert record.content == 'Content only'


def test_rerank_request() -> None:
    """Test RerankRequest model."""
    records = [
        RerankRequestRecord(id='1', content='Doc 1'),
        RerankRequestRecord(id='2', content='Doc 2'),
    ]
    request = RerankRequest(
        query='What is machine learning?',
        records=records,
        model='semantic-ranker-default@latest',
        top_n=5,
    )
    assert request.query == 'What is machine learning?'
    assert len(request.records) == 2
    assert request.model == 'semantic-ranker-default@latest'
    assert request.top_n == 5


def test_rerank_response_record() -> None:
    """Test RerankResponseRecord model."""
    record = RerankResponseRecord(
        id='doc-1',
        score=0.95,
        content='Document content',
    )
    assert record.id == 'doc-1'
    assert record.score == 0.95
    assert record.content == 'Document content'


def test_rerank_response_record_minimal() -> None:
    """Test RerankResponseRecord with only required fields."""
    record = RerankResponseRecord(id='1', score=0.5)
    assert record.id == '1'
    assert record.score == 0.5
    assert record.content is None
    assert record.title is None


def test_rerank_response() -> None:
    """Test RerankResponse model."""
    records = [
        RerankResponseRecord(id='1', score=0.9),
        RerankResponseRecord(id='2', score=0.7),
    ]
    response = RerankResponse(records=records)
    assert len(response.records) == 2
    assert response.records[0].score == 0.9


def test_get_vertex_rerank_url() -> None:
    """Test get_vertex_rerank_url builds correct URL."""
    options = VertexRerankerClientOptions(
        project_id='my-project',
        location='us-central1',
    )

    url = get_vertex_rerank_url(options)

    assert 'my-project' in url
    assert 'us-central1' in url
    assert 'discoveryengine.googleapis.com' in url
    assert ':rank' in url


def test_get_vertex_rerank_url_different_location() -> None:
    """Test get_vertex_rerank_url with different location."""
    options = VertexRerankerClientOptions(
        project_id='test-project',
        location='europe-west1',
    )

    url = get_vertex_rerank_url(options)

    assert 'test-project' in url
    assert 'europe-west1' in url


def test_to_reranker_doc_from_document() -> None:
    """Test _to_reranker_doc converts Document to RerankRequestRecord."""
    from genkit.core.typing import DocumentPart

    doc = Document(content=[DocumentPart(root=TextPart(text='This is document content.'))])

    record = _to_reranker_doc(doc, 0)

    assert record.content == 'This is document content.'
    assert record.id == '0'


def test_to_reranker_doc_different_index() -> None:
    """Test _to_reranker_doc uses provided index."""
    from genkit.core.typing import DocumentPart

    doc = Document(content=[DocumentPart(root=TextPart(text='Content'))])

    record = _to_reranker_doc(doc, 5)

    assert record.id == '5'


def test_from_rerank_response_basic() -> None:
    """Test _from_rerank_response converts response to scored documents."""
    from genkit.core.typing import DocumentPart

    original_docs = [
        Document(content=[DocumentPart(root=TextPart(text='Doc 0'))]),
        Document(content=[DocumentPart(root=TextPart(text='Doc 1'))]),
        Document(content=[DocumentPart(root=TextPart(text='Doc 2'))]),
    ]

    response = RerankResponse(
        records=[
            RerankResponseRecord(id='1', score=0.9),
            RerankResponseRecord(id='0', score=0.7),
            RerankResponseRecord(id='2', score=0.5),
        ]
    )

    result = _from_rerank_response(response, original_docs)

    assert len(result) == 3
    for doc in result:
        assert doc.metadata is not None
        assert 'score' in doc.metadata


def test_from_rerank_response_preserves_content() -> None:
    """Test _from_rerank_response preserves document content."""
    from genkit.core.typing import DocumentPart

    original_docs = [
        Document(content=[DocumentPart(root=TextPart(text='Original content'))]),
    ]

    response = RerankResponse(records=[RerankResponseRecord(id='0', score=0.85)])

    result = _from_rerank_response(response, original_docs)

    assert len(result) == 1
    assert result[0].text() == 'Original content'
    assert result[0].metadata is not None
    assert result[0].metadata.get('score') == 0.85


def test_from_rerank_response_preserves_original_metadata() -> None:
    """Test _from_rerank_response preserves original document metadata."""
    from genkit.core.typing import DocumentPart

    original_docs = [
        Document(
            content=[DocumentPart(root=TextPart(text='Content'))],
            metadata={'custom_field': 'value'},
        ),
    ]

    response = RerankResponse(records=[RerankResponseRecord(id='0', score=0.85)])

    result = _from_rerank_response(response, original_docs)

    assert len(result) == 1
    assert result[0].metadata is not None
    assert result[0].metadata.get('custom_field') == 'value'
    assert result[0].metadata.get('score') == 0.85


def test_from_rerank_response_empty() -> None:
    """Test _from_rerank_response handles empty response."""
    response = RerankResponse(records=[])

    result = _from_rerank_response(response, [])

    assert result == []


@pytest.mark.asyncio
async def test_reranker_api_call_structure() -> None:
    """Test that reranker API call is structured correctly."""
    from genkit.plugins.google_genai.rerankers.reranker import reranker_rank

    mock_credentials = MagicMock()
    mock_credentials.token = 'mock-token'
    mock_credentials.expired = False

    mock_response_data = {
        'records': [
            {'id': '0', 'score': 0.9},
            {'id': '1', 'score': 0.7},
        ]
    }

    with patch('genkit.plugins.google_genai.rerankers.reranker.google_auth_default') as mock_auth:
        mock_auth.return_value = (mock_credentials, 'test-project')

        # Mock get_cached_client to return a mock client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch('genkit.plugins.google_genai.rerankers.reranker.get_cached_client', return_value=mock_client):
            request = RerankRequest(
                model='semantic-ranker-default@latest',
                query='test query',
                records=[
                    RerankRequestRecord(id='0', content='Doc 1'),
                    RerankRequestRecord(id='1', content='Doc 2'),
                ],
            )
            options = VertexRerankerClientOptions(project_id='test-project')

            result = await reranker_rank('semantic-ranker-default@latest', request, options)

            assert isinstance(result, RerankResponse)
            assert len(result.records) == 2
