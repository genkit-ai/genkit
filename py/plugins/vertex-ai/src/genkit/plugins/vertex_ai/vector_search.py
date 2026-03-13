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

"""Vertex AI Vector Search integration for Genkit.

This module provides retrievers for Vertex AI Vector Search with BigQuery and Firestore backends.
"""

import json
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import structlog
from google.cloud import bigquery, firestore
from google.cloud.aiplatform_v1 import (
    FindNeighborsRequest,
    FindNeighborsResponse,
    IndexDatapoint,
    MatchServiceAsyncClient,
)
from pydantic import BaseModel, Field, ValidationError

from genkit.ai import Genkit
from genkit.blocks.document import Document
from genkit.blocks.retriever import RetrieverOptions, retriever_action_metadata
from genkit.core.action.types import ActionKind
from genkit.core.schema import to_json_schema
from genkit.core.typing import (
    DocumentData,
    Embedding,
    RetrieverResponse,
)
from genkit.types import ActionRunContext, RetrieverRequest

logger = structlog.get_logger(__name__)

DEFAULT_LIMIT_NEIGHBORS: int = 3


class RetrieverOptionsSchema(BaseModel):
    """Schema for retriever options.

    Attributes:
        limit: Number of documents to retrieve.
    """

    limit: int | None = Field(title='Number of documents to retrieve', default=None)


class DocRetriever(ABC):
    """Abstract base class for Vertex AI Vector Search document retrieval.

    This class outlines the core workflow for retrieving relevant documents.
    Subclasses must implement the abstract methods to provide concrete retrieval logic.

    Attributes:
        ai: The Genkit instance used for embeddings.
        name: The name of this retriever instance.
        embedder: The embedder to use for query embeddings.
        embedder_options: Optional configuration to pass to the embedder.
        match_service_client_generator: Generator function for the Vertex AI client.
    """

    def __init__(
        self,
        ai: Genkit,
        name: str,
        embedder: str,
        match_service_client_generator: Callable,
        embedder_options: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the DocRetriever.

        Args:
            ai: The Genkit instance used for embeddings.
            name: The name of this retriever instance.
            embedder: The embedder to use for query embeddings.
            match_service_client_generator: Generator function for the Vertex AI client.
            embedder_options: Optional configuration to pass to the embedder.
        """
        self.ai = ai
        self.name = name
        self.embedder = embedder
        self.embedder_options = embedder_options
        self._match_service_client_generator = match_service_client_generator

    async def retrieve(self, request: RetrieverRequest, _: ActionRunContext) -> RetrieverResponse:
        """Retrieves documents based on a given query.

        Args:
            request: The retrieval request containing the query.
            _: The ActionRunContext (unused in this method).

        Returns:
            A RetrieverResponse object containing the retrieved documents.
        """
        document = Document.from_document_data(document_data=request.query)

        # Get query embedding
        embed_resp = await self.ai.embed(
            embedder=self.embedder,
            content=document,
            options=self.embedder_options,
        )
        if not embed_resp:
            raise ValueError('Embedder returned no embeddings for query')

        # Get limit from options
        limit_neighbors = DEFAULT_LIMIT_NEIGHBORS
        if isinstance(request.options, dict) and (limit_val := request.options.get('limit')) is not None:
            limit_neighbors = int(limit_val)

        docs = await self._get_closest_documents(
            request=request,
            top_k=limit_neighbors,
            query_embeddings=Embedding(embedding=embed_resp[0].embedding),
        )

        return RetrieverResponse(documents=typing.cast(list[DocumentData], docs))

    async def _get_closest_documents(
        self, request: RetrieverRequest, top_k: int, query_embeddings: Embedding
    ) -> list[Document]:
        """Retrieves the closest documents from the vector search index.

        Args:
            request: The retrieval request containing the query and metadata.
            top_k: The number of nearest neighbors to retrieve.
            query_embeddings: The embedding of the query.

        Returns:
            A list of Document objects representing the closest documents.

        Raises:
            AttributeError: If the request does not contain the necessary metadata.
        """
        metadata = request.query.metadata

        required_keys = ['index_endpoint_path', 'api_endpoint', 'deployed_index_id']

        if not metadata:
            raise AttributeError('Request metadata provides no data about index')

        for rkey in required_keys:
            if rkey not in metadata:
                raise AttributeError(f'Request metadata provides no data for {rkey}')

        api_endpoint = metadata['api_endpoint']
        index_endpoint_path = metadata['index_endpoint_path']
        deployed_index_id = metadata['deployed_index_id']

        client_options = {'api_endpoint': api_endpoint}

        vector_search_client = self._match_service_client_generator(
            client_options=client_options,
        )

        nn_request = FindNeighborsRequest(
            index_endpoint=index_endpoint_path,
            deployed_index_id=deployed_index_id,
            queries=[
                FindNeighborsRequest.Query(
                    datapoint=IndexDatapoint(feature_vector=query_embeddings.embedding),
                    neighbor_count=top_k,
                )
            ],
        )

        response = await vector_search_client.find_neighbors(request=nn_request)

        return await self._retrieve_neighbors_data_from_db(neighbors=response.nearest_neighbors[0].neighbors)

    @abstractmethod
    async def _retrieve_neighbors_data_from_db(self, neighbors: list[FindNeighborsResponse.Neighbor]) -> list[Document]:
        """Retrieves document data from the database based on neighbor information.

        This method must be implemented by subclasses.

        Args:
            neighbors: A list of Neighbor objects from the vector search index.

        Returns:
            A list of Document objects containing the data for the retrieved documents.
        """
        raise NotImplementedError


class BigQueryRetriever(DocRetriever):
    """Retrieves documents from a BigQuery table.

    This class extends DocRetriever to fetch document data from a specified BigQuery
    dataset and table based on nearest neighbor search results.

    Attributes:
        bq_client: The BigQuery client to use for querying.
        dataset_id: The ID of the BigQuery dataset.
        table_id: The ID of the BigQuery table.
    """

    def __init__(
        self,
        bq_client: bigquery.Client,
        dataset_id: str,
        table_id: str,
        ai: Genkit,
        name: str,
        embedder: str,
        match_service_client_generator: Callable,
        embedder_options: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the BigQueryRetriever.

        Args:
            bq_client: The BigQuery client to use for querying.
            dataset_id: The ID of the BigQuery dataset.
            table_id: The ID of the BigQuery table.
            ai: The Genkit instance used for embeddings.
            name: The name of this retriever instance.
            embedder: The embedder to use for query embeddings.
            match_service_client_generator: Generator function for the Vertex AI client.
            embedder_options: Optional configuration to pass to the embedder.
        """
        super().__init__(
            ai=ai,
            name=name,
            embedder=embedder,
            match_service_client_generator=match_service_client_generator,
            embedder_options=embedder_options,
        )
        self.bq_client = bq_client
        self.dataset_id = dataset_id
        self.table_id = table_id

    async def _retrieve_neighbors_data_from_db(self, neighbors: list[FindNeighborsResponse.Neighbor]) -> list[Document]:
        """Retrieves document data from the BigQuery table for the given neighbors.

        Args:
            neighbors: A list of Neighbor objects representing the nearest neighbors.

        Returns:
            A list of Document objects containing the retrieved document data.
        """
        ids = [n.datapoint.datapoint_id for n in neighbors if n.datapoint and n.datapoint.datapoint_id]

        distance_by_id = {
            n.datapoint.datapoint_id: n.distance for n in neighbors if n.datapoint and n.datapoint.datapoint_id
        }

        if not ids:
            return []

        # dataset_id and table_id are from trusted config, ids are parameterized
        query = f"""
            SELECT * FROM `{self.dataset_id}.{self.table_id}`
            WHERE id IN UNNEST(@ids)
        """  # noqa: S608 - parameterized query, table names from trusted config

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter('ids', 'STRING', ids)],
        )

        try:
            query_job = self.bq_client.query(query, job_config=job_config)
            rows = query_job.result()
        except Exception as e:
            await logger.aerror('Failed to execute BigQuery query: %s', e)
            return []

        documents: list[Document] = []

        for row in rows:
            try:
                id = row['id']

                content = row['content']
                content = json.dumps(content) if isinstance(content, dict) else str(content)

                metadata = row.get('metadata', {})
                metadata['id'] = id
                metadata['distance'] = distance_by_id[id]

                documents.append(Document.from_text(content, metadata))
            except (ValidationError, json.JSONDecodeError, Exception) as error:
                doc_id = row.get('id', '<unknown>')
                await logger.awarning('Failed to parse document data for document with ID %s: %s', doc_id, error)

        return documents


class FirestoreRetriever(DocRetriever):
    """Retrieves documents from a Firestore collection.

    This class extends DocRetriever to fetch document data from a specified Firestore
    collection based on nearest neighbor search results.

    Attributes:
        db: The Firestore client.
        collection_name: The name of the Firestore collection.
    """

    def __init__(
        self,
        firestore_client: firestore.AsyncClient,
        collection_name: str,
        ai: Genkit,
        name: str,
        embedder: str,
        match_service_client_generator: Callable,
        embedder_options: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the FirestoreRetriever.

        Args:
            firestore_client: The Firestore client to use for querying.
            collection_name: The name of the Firestore collection.
            ai: The Genkit instance used for embeddings.
            name: The name of this retriever instance.
            embedder: The embedder to use for query embeddings.
            match_service_client_generator: Generator function for the Vertex AI client.
            embedder_options: Optional configuration to pass to the embedder.
        """
        super().__init__(
            ai=ai,
            name=name,
            embedder=embedder,
            match_service_client_generator=match_service_client_generator,
            embedder_options=embedder_options,
        )
        self.db = firestore_client
        self.collection_name = collection_name

    async def _retrieve_neighbors_data_from_db(self, neighbors: list[FindNeighborsResponse.Neighbor]) -> list[Document]:
        """Retrieves document data from the Firestore collection for the given neighbors.

        Args:
            neighbors: A list of Neighbor objects representing the nearest neighbors.

        Returns:
            A list of Document objects containing the retrieved document data.
        """
        documents: list[Document] = []

        for neighbor in neighbors:
            doc_ref = self.db.collection(self.collection_name).document(document_id=neighbor.datapoint.datapoint_id)
            # Typed as Any to bypass verification issues with google.cloud.firestore.DocumentSnapshot
            # which might be treated as a coroutine by the type checker in some contexts.
            doc: Any = await doc_ref.get()

            if doc.exists:
                doc_data = doc.to_dict() or {}

                content = doc_data.get('content', '')
                content = json.dumps(content) if isinstance(content, dict) else str(content)

                metadata = doc_data.get('metadata', {})
                metadata['id'] = neighbor.datapoint.datapoint_id
                metadata['distance'] = neighbor.distance

                try:
                    documents.append(
                        Document.from_text(
                            content,
                            metadata,
                        )
                    )
                except ValidationError as e:
                    await logger.awarning(
                        'Failed to parse document data for ID %s: %s',
                        neighbor.datapoint.datapoint_id,
                        e,
                    )

        return documents


def vertexai_vector_search_name(name: str) -> str:
    """Create a vertex AI vector search action name.

    Args:
        name: Base name for the action

    Returns:
        str: Vertex AI vector search action name.
    """
    return f'vertexai/{name}'


def define_vertex_vector_search_big_query(
    ai: Genkit,
    *,
    name: str,
    embedder: str,
    embedder_options: dict[str, Any] | None = None,
    bq_client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    match_service_client_generator: Callable | None = None,
) -> str:
    """Define and register a Vertex AI Vector Search retriever with BigQuery backend.

    Args:
        ai: The Genkit instance to register the retriever with.
        name: Name of the retriever.
        embedder: The embedder to use (e.g., 'vertexai/gemini-embedding-001').
        embedder_options: Optional configuration to pass to the embedder.
        bq_client: The BigQuery client to use for querying.
        dataset_id: The ID of the BigQuery dataset.
        table_id: The ID of the BigQuery table.
        match_service_client_generator: Optional generator for the Vertex AI client.
            Defaults to MatchServiceAsyncClient.

    Returns:
        The registered retriever name.
    """
    if match_service_client_generator is None:
        match_service_client_generator = MatchServiceAsyncClient

    retriever = BigQueryRetriever(
        ai=ai,
        name=name,
        embedder=embedder,
        embedder_options=embedder_options,
        match_service_client_generator=match_service_client_generator,
        bq_client=bq_client,
        dataset_id=dataset_id,
        table_id=table_id,
    )

    ai.registry.register_action(
        kind=ActionKind.RETRIEVER,
        name=name,
        fn=retriever.retrieve,
        metadata=retriever_action_metadata(
            name=name,
            options=RetrieverOptions(
                label=name,
                config_schema=to_json_schema(RetrieverOptionsSchema),
            ),
        ).metadata,
    )

    return name


def define_vertex_vector_search_firestore(
    ai: Genkit,
    *,
    name: str,
    embedder: str,
    embedder_options: dict[str, Any] | None = None,
    firestore_client: firestore.AsyncClient,
    collection_name: str,
    match_service_client_generator: Callable | None = None,
) -> str:
    """Define and register a Vertex AI Vector Search retriever with Firestore backend.

    Args:
        ai: The Genkit instance to register the retriever with.
        name: Name of the retriever.
        embedder: The embedder to use (e.g., 'vertexai/gemini-embedding-001').
        embedder_options: Optional configuration to pass to the embedder.
        firestore_client: The Firestore client to use for querying.
        collection_name: The name of the Firestore collection.
        match_service_client_generator: Optional generator for the Vertex AI client.
            Defaults to MatchServiceAsyncClient.

    Returns:
        The registered retriever name.
    """
    if match_service_client_generator is None:
        match_service_client_generator = MatchServiceAsyncClient

    retriever = FirestoreRetriever(
        ai=ai,
        name=name,
        embedder=embedder,
        embedder_options=embedder_options,
        match_service_client_generator=match_service_client_generator,
        firestore_client=firestore_client,
        collection_name=collection_name,
    )

    ai.registry.register_action(
        kind=ActionKind.RETRIEVER,
        name=name,
        fn=retriever.retrieve,
        metadata=retriever_action_metadata(
            name=name,
            options=RetrieverOptions(
                label=name,
                config_schema=to_json_schema(RetrieverOptionsSchema),
            ),
        ).metadata,
    )

    return name
