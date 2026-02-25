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

"""Vertex AI Plugin for Genkit.

This plugin provides integration with Google Cloud's Vertex AI platform,
including Model Garden for accessing third-party models and Vector Search
for RAG applications.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Vertex AI           │ Google Cloud's AI platform. Like a shopping       │
    │                     │ mall where you can access many AI services.       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Model Garden        │ A catalog of AI models from different companies.  │
    │                     │ Like an app store but for AI models.              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Vector Search       │ Find similar items using math. Like asking        │
    │                     │ "show me documents similar to this one."          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ RAG                 │ Retrieval-Augmented Generation. Let AI search     │
    │                     │ your documents before answering. Like giving      │
    │                     │ AI a reference book to look things up.            │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ BigQuery            │ Google's data warehouse. Store and search         │
    │                     │ huge amounts of data super fast.                  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Firestore           │ Google's NoSQL database. Store documents          │
    │                     │ as flexible JSON-like data.                       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Embeddings          │ Turn text into numbers for comparison.            │
    │                     │ Like converting words to GPS coordinates.         │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow (Vector Search)::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  HOW VECTOR SEARCH FINDS SIMILAR DOCUMENTS              │
    │                                                                         │
    │    Your Query: "How do I reset my password?"                            │
    │         │                                                               │
    │         │  (1) Query converted to embedding (numbers)                   │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Embedder       │   Text → [0.12, -0.45, 0.78, ...]                │
    │    │  (Gemini)       │   (hundreds of numbers)                          │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Search for similar embeddings                        │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Vector Index   │   Find documents with similar                    │
    │    │  (BigQuery or   │   number patterns                                │
    │    │   Firestore)    │                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) Return matching documents                            │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Results        │   "Password Reset Guide" (95% match)             │
    │    │                 │   "Account Recovery FAQ" (87% match)             │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Vertex AI Plugin                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Plugin Entry Point (__init__.py)                                       │
    │  ├── ModelGardenPlugin - Access third-party models via Model Garden     │
    │  ├── Vector Search Retrievers (BigQuery, Firestore)                     │
    │  └── Helper functions for defining vector search                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  model_garden/modelgarden_plugin.py - Model Garden Integration          │
    │  ├── ModelGardenPlugin class                                            │
    │  └── Access to Anthropic, Llama, Mistral via Vertex AI                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  model_garden/client.py - API Client                                    │
    │  └── Google Cloud client initialization                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  model_garden/anthropic.py - Anthropic Models                           │
    │  └── Claude models via Vertex AI Model Garden                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  vector_search.py - Vector Search Integration                           │
    │  ├── BigQueryRetriever - Vector search with BigQuery backend            │
    │  ├── FirestoreRetriever - Vector search with Firestore backend          │
    │  └── RetrieverOptionsSchema - Configuration for retrievers              │
    └─────────────────────────────────────────────────────────────────────────┘

Key Components:
    - ModelGardenPlugin: Access third-party models (Anthropic, Meta, Mistral)
      through Vertex AI Model Garden
    - BigQueryRetriever: Vector similarity search using BigQuery
    - FirestoreRetriever: Vector similarity search using Firestore

Example:
    ```python
    from genkit import Genkit
    from genkit.plugins.vertex_ai import (
        ModelGardenPlugin,
        define_vertex_vector_search_firestore,
    )

    # Model Garden for third-party models
    ai = Genkit(
        plugins=[ModelGardenPlugin(project='my-project', location='us-central1')],
    )

    # Vector Search with Firestore
    store = define_vertex_vector_search_firestore(
        ai,
        name='my_store',
        collection='documents',
        embedder='vertexai/text-embedding-005',
    )
    ```

Caveats:
    - Requires Google Cloud credentials (ADC or explicit)
    - Model Garden requires models to be deployed in your project
    - Vector Search requires appropriate index configuration

See Also:
    - Vertex AI Model Garden: https://cloud.google.com/vertex-ai/docs/model-garden
    - Vertex AI Vector Search: https://cloud.google.com/vertex-ai/docs/vector-search
    - Genkit documentation: https://genkit.dev/
"""

from genkit.plugins.vertex_ai.model_garden.modelgarden_plugin import (
    ModelGardenPlugin,
)
from genkit.plugins.vertex_ai.vector_search import (
    BigQueryRetriever,
    FirestoreRetriever,
    RetrieverOptionsSchema,
    define_vertex_vector_search_big_query,
    define_vertex_vector_search_firestore,
)


def package_name() -> str:
    """Get the package name for the Vertex AI plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit.plugins.vertex_ai'


__all__ = [
    'BigQueryRetriever',
    'FirestoreRetriever',
    'ModelGardenPlugin',
    'RetrieverOptionsSchema',
    'define_vertex_vector_search_big_query',
    'define_vertex_vector_search_firestore',
    'package_name',
]
