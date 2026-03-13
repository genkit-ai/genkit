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

"""Vertex AI Rerankers for the Genkit framework.

This module provides reranking functionality using the Vertex AI Discovery Engine
Ranking API. Rerankers improve RAG (Retrieval-Augmented Generation) quality by
re-scoring documents based on their relevance to a query.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Reranker            │ A "second opinion" scorer that re-orders your     │
    │                     │ search results by relevance. Like asking an expert │
    │                     │ to sort your library books by importance.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Semantic Ranker     │ Uses AI to understand meaning, not just keywords. │
    │                     │ Knows "car" and "automobile" mean the same thing. │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ top_n               │ How many top results to return after reranking.   │
    │                     │ "Give me the 5 most relevant documents."          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Score               │ A number (0-1) showing how relevant a document is.│
    │                     │ Higher = more relevant to your query.             │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      RAG WITH RERANKING                                 │
    │                                                                         │
    │   User Query: "How do neural networks learn?"                          │
    │        │                                                                │
    │        ▼                                                                │
    │   ┌─────────────┐                                                       │
    │   │  Retriever  │  ◄── Fast initial search, returns ~100 docs          │
    │   └──────┬──────┘                                                       │
    │          │  [doc1, doc2, doc3, ... doc100]                             │
    │          ▼                                                              │
    │   ┌─────────────┐                                                       │
    │   │  Reranker   │  ◄── AI-powered relevance scoring                    │
    │   │  (Vertex)   │                                                       │
    │   └──────┬──────┘                                                       │
    │          │  [doc47: 0.95, doc3: 0.87, doc12: 0.82, ...]                │
    │          ▼                                                              │
    │   ┌─────────────┐                                                       │
    │   │   Model     │  ◄── Uses top-k most relevant docs                   │
    │   │  (Gemini)   │                                                       │
    │   └──────┬──────┘                                                       │
    │          ▼                                                              │
    │   High-quality answer with accurate citations                          │
    └─────────────────────────────────────────────────────────────────────────┘

Overview:
    Vertex AI offers semantic rerankers that use machine learning to score
    documents based on their semantic similarity to a query. This is typically
    used after initial retrieval to improve the quality of the top-k results.

    Reranking is a two-stage retrieval pattern:
    1. **Fast retrieval**: Get many candidates quickly (e.g., 100 docs)
    2. **Quality reranking**: Score candidates by relevance, keep top-k

Available Models:
    +--------------------------------+-----------------------------------------+
    | Model                          | Description                             |
    +--------------------------------+-----------------------------------------+
    | semantic-ranker-default@latest | Latest default semantic ranker          |
    | semantic-ranker-default-004    | Semantic ranker version 004             |
    | semantic-ranker-fast-004       | Fast variant (lower latency, less acc.) |
    | semantic-ranker-default-003    | Semantic ranker version 003             |
    | semantic-ranker-default-002    | Semantic ranker version 002             |
    +--------------------------------+-----------------------------------------+

Example:
    Basic reranking:

        >>> from genkit import Genkit
        >>> from genkit.plugins.google_genai import VertexAI
        >>>
        >>> ai = Genkit(plugins=[VertexAI(project='my-project')])
        >>>
        >>> # Rerank documents after retrieval
        >>> ranked_docs = await ai.rerank(
        ...     reranker='vertexai/semantic-ranker-default@latest',
        ...     query='What is machine learning?',
        ...     documents=retrieved_docs,
        ...     options={'top_n': 5},
        ... )

    Full RAG pipeline with reranking:

        >>> # 1. Retrieve initial candidates
        >>> candidates = await ai.retrieve(
        ...     retriever='my-retriever',
        ...     query='How do neural networks learn?',
        ...     options={'limit': 50},
        ... )
        >>>
        >>> # 2. Rerank for quality
        >>> ranked = await ai.rerank(
        ...     reranker='vertexai/semantic-ranker-default@latest',
        ...     query='How do neural networks learn?',
        ...     documents=candidates,
        ...     options={'top_n': 5},
        ... )
        >>>
        >>> # 3. Generate with top results
        >>> response = await ai.generate(
        ...     model='vertexai/gemini-2.0-flash',
        ...     prompt='Explain how neural networks learn.',
        ...     docs=ranked,
        ... )

Caveats:
    - Requires Google Cloud project with Discovery Engine API enabled
    - Reranking adds latency - use for quality-critical applications
    - Models may silently fall back to default if name is not recognized

See Also:
    - Vertex AI Ranking API: https://cloud.google.com/generative-ai-app-builder/docs/ranking
    - RAG best practices: https://genkit.dev/docs/rag
"""

from genkit.plugins.google_genai.rerankers.reranker import (
    DEFAULT_MODEL_NAME,
    KNOWN_MODELS,
    RerankRequest,
    RerankResponse,
    VertexRerankerClientOptions,
    VertexRerankerConfig,
    _from_rerank_response,
    _to_reranker_doc,
    is_reranker_model_name,
    reranker_rank,
)

__all__ = [
    'DEFAULT_MODEL_NAME',
    'KNOWN_MODELS',
    'RerankRequest',
    'RerankResponse',
    'VertexRerankerClientOptions',
    'VertexRerankerConfig',
    '_from_rerank_response',
    '_to_reranker_doc',
    'is_reranker_model_name',
    'reranker_rank',
]
