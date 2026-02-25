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

"""Reranker type definitions for the Genkit framework.

Rerankers and Two-Stage Retrieval
=================================

A **reranking model** (also known as a cross-encoder) is a type of model that,
given a query and document, outputs a similarity score. This score is used to
reorder documents by relevance to the query.

Reranker APIs take a list of documents (e.g., the output of a retriever) and
reorder them based on their relevance to the query. This step can be useful
for fine-tuning results and ensuring the most pertinent information is used
in the prompt provided to a generative model.

Two-Stage Retrieval
-------------------

In a typical RAG (Retrieval-Augmented Generation) pipeline:

1. **Stage 1 - Retrieval**: A retriever fetches a large set of candidate
   documents using fast vector similarity search.
2. **Stage 2 - Reranking**: A reranker scores and reorders these candidates
   using more expensive but accurate cross-encoder models.

This two-stage approach balances speed and accuracy:
- Retrievers are fast but may not perfectly rank results
- Rerankers are slower but provide superior relevance scoring

Usage Example
-------------

Using an existing reranker (e.g., Vertex AI):

.. code-block:: python

    from genkit.ai import Genkit

    ai = Genkit(plugins=[...])


    @ai.flow()
    async def rerank_flow(query: str):
        documents = [
            Document.from_text('pythagorean theorem'),
            Document.from_text('quantum mechanics'),
            Document.from_text('pizza'),
        ]

        reranked = await ai.rerank(
            reranker='vertexai/semantic-ranker-512',
            query=query,
            documents=documents,
        )

        return [{'text': doc.text(), 'score': doc.score} for doc in reranked]

Custom Rerankers
----------------

You can define custom rerankers for specific use cases:

.. code-block:: python

    from genkit.ai import Genkit
    from genkit.core.typing import (
        RerankerResponse,
        RankedDocumentData,
        RankedDocumentMetadata,
    )

    ai = Genkit()


    async def custom_reranker_fn(query, documents, options):
        # Your custom reranking logic here
        # Example: score by keyword overlap
        query_words = set(query.text().lower().split())
        scored = []
        for doc in documents:
            doc_words = set(doc.text().lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            scored.append((doc, score))

        # Sort by score descending and take top k
        k = options.get('k', 3) if options else 3
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:k]

        return RerankerResponse(
            documents=[
                RankedDocumentData(content=doc.content, metadata=RankedDocumentMetadata(score=score))
                for doc, score in top_k
            ]
        )


    ai.define_reranker('custom/keyword-reranker', custom_reranker_fn)


    # Use it in a flow
    @ai.flow()
    async def search_flow(query: str):
        docs = await ai.retrieve(retriever='my-retriever', query=query)
        return await ai.rerank(reranker='custom/keyword-reranker', query=query, documents=docs, options={'k': 5})
"""

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, TypeVar, cast

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit.blocks.document import Document
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind
from genkit.core.registry import Registry
from genkit.core.schema import to_json_schema
from genkit.core.typing import (
    DocumentData,
    DocumentPart,
    RankedDocumentData,
    RerankerRequest,
    RerankerResponse,
)

T = TypeVar('T')

# Type alias for reranker function
RerankerFn = Callable[[Document, list[Document], T], Awaitable[RerankerResponse]]


class RankedDocument(Document):
    """A document with a relevance score from reranking.

    This class extends Document to include a score property that represents
    the document's relevance to a query as determined by a reranker.
    """

    def __init__(
        self,
        content: list[DocumentPart],
        metadata: dict[str, Any] | None = None,
        score: float | None = None,
    ) -> None:
        """Initializes a RankedDocument object.

        Args:
            content: A list of DocumentPart objects representing the document's content.
            metadata: An optional dictionary containing metadata about the document.
            score: The relevance score from reranking.
        """
        md = metadata.copy() if metadata else {}
        if score is not None:
            md['score'] = score
        super().__init__(content=content, metadata=md)

    @property
    def score(self) -> float | None:
        """Returns the relevance score of the document.

        Returns:
            The relevance score as a float, or None if not set.
        """
        if self.metadata and 'score' in self.metadata:
            return self.metadata['score']
        return None

    @staticmethod
    def from_ranked_document_data(data: RankedDocumentData) -> 'RankedDocument':
        """Constructs a RankedDocument from RankedDocumentData.

        Args:
            data: The RankedDocumentData containing content, metadata with score.

        Returns:
            A new RankedDocument instance.
        """
        return RankedDocument(
            content=data.content,
            metadata=data.metadata.model_dump(),
            score=data.metadata.score,
        )


class RerankerSupports(BaseModel):
    """Reranker capability support."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True)

    media: bool | None = None


class RerankerInfo(BaseModel):
    """Information about a reranker's capabilities."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True)

    label: str | None = None
    supports: RerankerSupports | None = None


class RerankerOptions(BaseModel):
    """Configuration options for a reranker."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True, alias_generator=to_camel)

    config_schema: dict[str, Any] | None = None
    label: str | None = None
    supports: RerankerSupports | None = None


class RerankerRef(BaseModel):
    """Reference to a reranker with configuration.

    Used to reference a reranker by name with optional configuration
    and version information.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True)

    name: str
    config: Any | None = None
    version: str | None = None
    info: RerankerInfo | None = None


def reranker_action_metadata(
    name: str,
    options: RerankerOptions | None = None,
) -> ActionMetadata:
    """Creates action metadata for a reranker.

    Args:
        name: The name of the reranker.
        options: Optional configuration options for the reranker.

    Returns:
        An ActionMetadata instance for the reranker.
    """
    options = options if options is not None else RerankerOptions()
    reranker_metadata_dict: dict[str, Any] = {'reranker': {}}

    if options.label:
        reranker_metadata_dict['reranker']['label'] = options.label

    if options.supports:
        reranker_metadata_dict['reranker']['supports'] = options.supports.model_dump(exclude_none=True, by_alias=True)

    reranker_metadata_dict['reranker']['customOptions'] = options.config_schema if options.config_schema else None

    return ActionMetadata(
        kind=cast(ActionKind, ActionKind.RERANKER),
        name=name,
        input_json_schema=to_json_schema(RerankerRequest),
        output_json_schema=to_json_schema(RerankerResponse),
        metadata=reranker_metadata_dict,
    )


def create_reranker_ref(
    name: str,
    config: dict[str, Any] | None = None,
    version: str | None = None,
    info: RerankerInfo | None = None,
) -> RerankerRef:
    """Creates a RerankerRef instance.

    Args:
        name: The name of the reranker.
        config: Optional configuration for the reranker.
        version: Optional version string.
        info: Optional RerankerInfo with capability information.

    Returns:
        A RerankerRef instance.
    """
    return RerankerRef(name=name, config=config, version=version, info=info)


def define_reranker(
    registry: Registry,
    name: str,
    fn: RerankerFn[Any],
    options: RerankerOptions | None = None,
    description: str | None = None,
) -> Action:
    """Defines and registers a reranker action.

    Creates a reranker action from the provided function and registers it
    in the given registry.

    Args:
        registry: The registry to register the reranker in.
        name: The name of the reranker.
        fn: The reranker function that implements the reranking logic.
        options: Optional configuration options for the reranker.
        description: Optional description for the reranker action.

    Returns:
        The registered Action instance.

    Example:
        >>> async def my_reranker(query, documents, options):
        ...     # Score and sort documents
        ...     scored = [(doc, score_doc(query, doc)) for doc in documents]
        ...     scored.sort(key=lambda x: x[1], reverse=True)
        ...     return RerankerResponse(
        ...         documents=[
        ...             RankedDocumentData(content=doc.content, metadata=RankedDocumentMetadata(score=score))
        ...             for doc, score in scored
        ...         ]
        ...     )
        >>> define_reranker(registry, 'my-reranker', my_reranker)
    """
    metadata = reranker_action_metadata(name, options)

    async def wrapper(
        request: RerankerRequest,
        _ctx: Any,  # noqa: ANN401
    ) -> RerankerResponse:
        query_doc = Document.from_document_data(request.query)
        documents = [Document.from_document_data(d) for d in request.documents]
        return await fn(query_doc, documents, request.options)

    return registry.register_action(
        kind=cast(ActionKind, ActionKind.RERANKER),
        name=name,
        fn=wrapper,
        metadata=metadata.metadata,
        span_metadata={'genkit:metadata:reranker:name': name},
        description=description,
    )


# Type for reranker argument (can be action, reference, or string name)
RerankerArgument = Action | RerankerRef | str


class RerankerParams(BaseModel):
    """Parameters for the rerank function.

    Attributes:
        reranker: The reranker to use (action, reference, or name string).
        query: The query to rank documents against.
        documents: The list of documents to rerank.
        options: Optional configuration options for this rerank call.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True, arbitrary_types_allowed=True)

    reranker: RerankerArgument
    query: str | DocumentData
    documents: list[DocumentData]
    options: Any | None = None


async def rerank(
    registry: Registry,
    params: RerankerParams | dict[str, Any],
) -> list[RankedDocument]:
    """Reranks documents based on the provided query using a reranker.

    This function takes a query and a list of documents, and returns the
    documents reordered by relevance to the query as determined by the
    specified reranker.

    Args:
        registry: The registry to look up the reranker in.
        params: Parameters for the rerank operation + including the reranker,
            query, documents, and optional configuration.

    Returns:
        A list of RankedDocument objects sorted by relevance.

    Raises:
        ValueError: If the reranker cannot be resolved.

    Example:
        >>> ranked_docs = await rerank(
        ...     registry,
        ...     {
        ...         'reranker': 'my-reranker',
        ...         'query': 'What is machine learning?',
        ...         'documents': [doc1, doc2, doc3],
        ...     },
        ... )
        >>> for doc in ranked_docs:
        ...     print(f'Score: {doc.score}, Text: {doc.text()}')
    """
    # Convert dict to RerankerParams if needed
    if isinstance(params, dict):
        params = RerankerParams(**params)

    # Resolve the reranker action
    reranker_action = None

    if isinstance(params.reranker, str):
        reranker_action = await registry.resolve_reranker(params.reranker)
    elif isinstance(params.reranker, RerankerRef):
        reranker_action = await registry.resolve_reranker(params.reranker.name)
    elif isinstance(params.reranker, Action):  # pyright: ignore[reportUnnecessaryIsInstance]
        reranker_action = params.reranker

    if reranker_action is None:
        raise ValueError(f'Unable to resolve reranker: {params.reranker}')

    # Convert query to DocumentData if it's a string
    query_data: DocumentData
    query_data = Document.from_text(params.query) if isinstance(params.query, str) else params.query

    # Build the request
    request = RerankerRequest(
        query=query_data,
        documents=params.documents,
        options=params.options,
    )

    # Call the reranker
    action_response = await reranker_action.arun(request)
    response: RerankerResponse = action_response.response

    # Convert response to RankedDocument list
    return [RankedDocument.from_ranked_document_data(doc) for doc in response.documents]
