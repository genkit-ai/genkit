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

"""Vertex AI Reranker implementation.

This module implements the Vertex AI Discovery Engine Ranking API for reranking
documents based on their semantic relevance to a query.

Architecture::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    Vertex AI Reranker Module                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Constants & Configuration                                              │
    │  ├── DEFAULT_LOCATION (global)                                          │
    │  ├── DEFAULT_MODEL_NAME (semantic-ranker-default@latest)                │
    │  └── KNOWN_MODELS (supported model registry)                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Request/Response Types (Pydantic)                                      │
    │  ├── VertexRerankerConfig - User-facing configuration                   │
    │  ├── VertexRerankerClientOptions - Internal client config               │
    │  ├── RerankRequest, RerankRequestRecord - API request types             │
    │  └── RerankResponse, RerankResponseRecord - API response types          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  API Client                                                             │
    │  ├── reranker_rank() - Async API call to Discovery Engine               │
    │  └── get_vertex_rerank_url() - URL builder for ranking endpoint         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Conversion Functions                                                   │
    │  ├── _to_reranker_doc() - Document → RerankRequestRecord                │
    │  └── _from_rerank_response() - Response → RankedDocument list           │
    └─────────────────────────────────────────────────────────────────────────┘

Implementation Notes:
    - Uses Google Cloud Application Default Credentials (ADC) for auth
    - Calls the Discovery Engine rankingConfigs:rank endpoint
    - Supports configurable location and top_n parameters
    - Returns RankedDocument instances with scores

Note:
    The actual reranker action registration is handled by the VertexAI plugin
    in google.py via the _resolve_reranker method, which uses the conversion
    functions and API client defined here.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, ClassVar

from google.auth import default as google_auth_default
from google.auth.transport.requests import Request
from pydantic import BaseModel, ConfigDict, Field

from genkit.blocks.document import Document
from genkit.blocks.model import text_from_content
from genkit.blocks.reranker import RankedDocument
from genkit.core.error import GenkitError
from genkit.core.http_client import get_cached_client
from genkit.core.typing import DocumentData

# Default location for Vertex AI Ranking API (global is recommended per docs)
DEFAULT_LOCATION = 'global'

# Default reranker model name
DEFAULT_MODEL_NAME = 'semantic-ranker-default@latest'

# Known reranker models
KNOWN_MODELS: dict[str, str] = {
    'semantic-ranker-default@latest': 'semantic-ranker-default@latest',
    'semantic-ranker-default-004': 'semantic-ranker-default-004',
    'semantic-ranker-fast-004': 'semantic-ranker-fast-004',
    'semantic-ranker-default-003': 'semantic-ranker-default-003',
    'semantic-ranker-default-002': 'semantic-ranker-default-002',
}


def is_reranker_model_name(value: str | None) -> bool:
    """Check if a value is a valid reranker model name.

    Args:
        value: The value to check.

    Returns:
        True if the value is a valid reranker model name.
    """
    return value is not None and value.startswith('semantic-ranker-')


class VertexRerankerConfig(BaseModel):
    """Configuration options for Vertex AI reranker.

    Attributes:
        top_n: Number of top documents to return. If not specified, all documents
            are returned with their scores.
        ignore_record_details_in_response: If True, the response will only contain
            record ID and score. Defaults to False.
        location: Google Cloud location (e.g., "us-central1"). If not specified,
            uses the default location from plugin options.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra='allow',
        populate_by_name=True,
    )

    top_n: int | None = Field(default=None, alias='topN')
    ignore_record_details_in_response: bool | None = Field(
        default=None,
        alias='ignoreRecordDetailsInResponse',
    )
    location: str | None = None


class RerankRequestRecord(BaseModel):
    """A record to be reranked.

    Attributes:
        id: Unique identifier for the record.
        title: Optional title of the record.
        content: The content of the record to be ranked.
    """

    id: str
    title: str | None = None
    content: str


class RerankRequest(BaseModel):
    """Request body for the rerank API.

    Attributes:
        model: The reranker model to use.
        query: The query to rank documents against.
        records: The records to be ranked.
        top_n: Number of top documents to return.
        ignore_record_details_in_response: If True, only return ID and score.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra='allow',
        populate_by_name=True,
    )

    model: str
    query: str
    records: list[RerankRequestRecord]
    top_n: int | None = Field(default=None, alias='topN')
    ignore_record_details_in_response: bool | None = Field(
        default=None,
        alias='ignoreRecordDetailsInResponse',
    )


class RerankResponseRecord(BaseModel):
    """A record in the rerank response.

    Attributes:
        id: The record ID.
        score: The relevance score (0-1).
        content: The record content (if not ignored).
        title: The record title (if present).
    """

    id: str
    score: float
    content: str | None = None
    title: str | None = None


class RerankResponse(BaseModel):
    """Response from the rerank API.

    Attributes:
        records: The ranked records with scores.
    """

    records: list[RerankResponseRecord]


class VertexRerankerClientOptions(BaseModel):
    """Client options for the Vertex AI reranker.

    Attributes:
        project_id: Google Cloud project ID.
        location: Google Cloud location (e.g., "us-central1").
    """

    project_id: str
    location: str = DEFAULT_LOCATION


async def reranker_rank(
    model: str,
    request: RerankRequest,
    client_options: VertexRerankerClientOptions,
) -> RerankResponse:
    """Call the Vertex AI Ranking API.

    Args:
        model: The reranker model name.
        request: The rerank request.
        client_options: Client options including project and location.

    Returns:
        The rerank response with scored records.

    Raises:
        GenkitError: If the API call fails.
    """
    url = get_vertex_rerank_url(client_options)

    # Get authentication token
    # Use asyncio.to_thread to avoid blocking the event loop during token refresh
    credentials, _ = google_auth_default()
    await asyncio.to_thread(credentials.refresh, Request())
    token = credentials.token

    if not token:
        raise GenkitError(
            message='Unable to authenticate your request. '
            'Please ensure you have valid Google Cloud credentials configured.',
            status='UNAUTHENTICATED',
        )

    headers = {
        'Authorization': f'Bearer {token}',
        'x-goog-user-project': client_options.project_id,
        'Content-Type': 'application/json',
    }

    # Prepare request body - only include non-None values
    request_body: dict[str, Any] = {
        'model': request.model,
        'query': request.query,
        'records': [r.model_dump(exclude_none=True) for r in request.records],
    }
    if request.top_n is not None:
        request_body['topN'] = request.top_n
    if request.ignore_record_details_in_response is not None:
        request_body['ignoreRecordDetailsInResponse'] = request.ignore_record_details_in_response

    # Use cached client for better connection reuse.
    # Note: Auth headers are passed per-request since tokens may expire.
    client = get_cached_client(
        cache_key='vertex-ai-reranker',
        timeout=60.0,
    )

    try:
        response = await client.post(
            url,
            headers=headers,
            json=request_body,
        )

        if response.status_code != 200:
            error_message = response.text
            try:
                error_json = response.json()
                if 'error' in error_json and 'message' in error_json['error']:
                    error_message = error_json['error']['message']
            except json.JSONDecodeError:  # noqa: S110
                # JSON parsing failed, use raw text
                pass

            raise GenkitError(
                message=f'Error calling Vertex AI Reranker API: [{response.status_code}] {error_message}',
                status='INTERNAL',
            )

        return RerankResponse.model_validate(response.json())

    except Exception as e:
        if isinstance(e, GenkitError):
            raise
        raise GenkitError(
            message=f'Failed to call Vertex AI Reranker API: {e}',
            status='UNAVAILABLE',
        ) from e


def get_vertex_rerank_url(client_options: VertexRerankerClientOptions) -> str:
    """Get the URL for the Vertex AI Ranking API.

    Args:
        client_options: Client options including project and location.

    Returns:
        The API endpoint URL.
    """
    return (
        f'https://discoveryengine.googleapis.com/v1/projects/{client_options.project_id}'
        f'/locations/{client_options.location}/rankingConfigs/default_ranking_config:rank'
    )


def _to_reranker_doc(doc: Document | DocumentData, idx: int) -> RerankRequestRecord:
    """Convert a document to a rerank request record.

    Args:
        doc: The document to convert.
        idx: The index of the document (used as ID).

    Returns:
        A rerank request record.
    """
    if isinstance(doc, Document):
        text = doc.text()
    else:
        # DocumentData - use text_from_content helper
        text = text_from_content(doc.content)

    return RerankRequestRecord(
        id=str(idx),
        content=text,
    )


def _from_rerank_response(
    response: RerankResponse,
    documents: list[Document],
) -> list[RankedDocument]:
    """Convert rerank response to ranked documents.

    Args:
        response: The rerank response.
        documents: The original documents.

    Returns:
        RankedDocument instances with scores, sorted by relevance.
    """
    ranked_docs: list[RankedDocument] = []
    for record in response.records:
        idx = int(record.id)
        original_doc = documents[idx]

        # Create RankedDocument with the score from the API response
        ranked_docs.append(
            RankedDocument(
                content=original_doc.content,
                metadata=original_doc.metadata,
                score=record.score,
            )
        )

    return ranked_docs
