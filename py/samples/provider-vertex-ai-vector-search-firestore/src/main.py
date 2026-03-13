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

"""Vertex AI Vector Search with Firestore sample.

This sample demonstrates how to use Vertex AI Vector Search with Firestore
as the document store for enterprise-scale vector similarity search.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Vertex AI Vector    │ Google's enterprise vector search. Handles         │
    │ Search              │ billions of items with fast nearest-neighbor.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Firestore           │ Stores the actual document content. Vector Search  │
    │                     │ returns IDs, Firestore returns full text.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Deployed Index      │ Your vector index running on Google's servers.     │
    │                     │ Ready to answer similarity queries 24/7.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Endpoint            │ The address where your index listens for queries.  │
    │                     │ Like a phone number for your search service.       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Distance            │ How "far apart" two vectors are. Lower = more      │
    │                     │ similar. 0 = identical match.                      │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow (Enterprise RAG)::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │        HOW VERTEX VECTOR SEARCH + FIRESTORE WORK TOGETHER               │
    │                                                                         │
    │    Query: "What movies are about time travel?"                          │
    │         │                                                               │
    │         │  (1) Convert to embedding                                     │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Embedder       │   Query → [0.3, -0.2, 0.7, ...]                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Send to Vector Search endpoint                       │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Vertex AI      │   Returns: ["doc_123", "doc_456", ...]           │
    │    │  Vector Search  │   (IDs only, ranked by similarity)               │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) Fetch full documents from Firestore                  │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Firestore      │   Returns: full document content                 │
    │    │  (by ID)        │   "Back to the Future is a 1985 film..."        │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (4) Sorted results returned                              │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Your App       │   Complete docs, ranked by relevance             │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Firestore Vector Search Definition      | `define_vertex_vector_search_firestore`|
| Firestore Async Client Integration      | `firestore.AsyncClient()`           |
| Document Retrieval                      | `ai.retrieve()`                     |
| Result Ranking                          | Custom sorting by distance          |

Testing This Demo
=================
1. **Prerequisites** - Set up GCP resources:
   ```bash
   # Required environment variables
   export LOCATION=us-central1
   export PROJECT_ID=your_project_id
   export FIRESTORE_COLLECTION=your_collection_name
   export VECTOR_SEARCH_DEPLOYED_INDEX_ID=your_deployed_index_id
   export VECTOR_SEARCH_INDEX_ENDPOINT_PATH=your_endpoint_path
   export VECTOR_SEARCH_API_ENDPOINT=your_api_endpoint

   # Authenticate with GCP
   gcloud auth application-default login
   ```

2. **GCP Setup Required**:
   - Create Vertex AI Vector Search index
   - Deploy index to an endpoint
   - Create Firestore collection with documents
   - Ensure documents have matching IDs in both services

3. **Run the demo**:
   ```bash
   cd py/samples/provider-vertex-ai-vector-search-firestore
   ./run.sh
   ```

4. **Open DevUI** at http://localhost:4000

5. **Test the flows**:
   - [ ] `retrieve_documents` - Vector similarity search
   - [ ] Check results are ranked by distance
   - [ ] Verify Firestore document metadata is returned

6. **Expected behavior**:
   - Query is embedded and sent to Vector Search
   - Similar vectors are found and IDs returned
   - Firestore is queried for full document content
   - Results sorted by similarity distance
"""

import os
import time

from google.cloud import aiplatform, firestore
from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.blocks.document import Document
from genkit.core.logging import get_logger
from genkit.core.typing import RetrieverResponse
from genkit.plugins.google_genai import VertexAI
from genkit.plugins.vertex_ai import define_vertex_vector_search_firestore
from samples.shared.logging import setup_sample

setup_sample()

LOCATION = os.environ['LOCATION']
PROJECT_ID = os.environ['PROJECT_ID']

FIRESTORE_COLLECTION = os.environ['FIRESTORE_COLLECTION']

VECTOR_SEARCH_DEPLOYED_INDEX_ID = os.environ['VECTOR_SEARCH_DEPLOYED_INDEX_ID']
VECTOR_SEARCH_INDEX_ENDPOINT_PATH = os.environ['VECTOR_SEARCH_INDEX_ENDPOINT_PATH']
VECTOR_SEARCH_API_ENDPOINT = os.environ['VECTOR_SEARCH_API_ENDPOINT']

firestore_client = firestore.AsyncClient(project=PROJECT_ID)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

logger = get_logger(__name__)

ai = Genkit(plugins=[VertexAI()])

# Define Vertex AI Vector Search with Firestore
define_vertex_vector_search_firestore(
    ai,
    name='my-vector-search',
    embedder='vertexai/gemini-embedding-001',
    embedder_options={
        'task': 'RETRIEVAL_DOCUMENT',
        'output_dimensionality': 128,
    },
    firestore_client=firestore_client,
    collection_name=FIRESTORE_COLLECTION,
)


class QueryFlowInputSchema(BaseModel):
    """Input schema."""

    query: str = Field(default='document 1', description='Search query text')
    k: int = Field(default=5, description='Number of results to return')


class QueryFlowOutputSchema(BaseModel):
    """Output schema."""

    result: list[dict[str, object]]
    length: int
    time: int


@ai.flow(name='queryFlow')
async def query_flow(_input: QueryFlowInputSchema) -> QueryFlowOutputSchema:
    """Executes a vector search with VertexAI Vector Search."""
    start_time = time.time()

    query_document = Document.from_text(text=_input.query)
    query_document.metadata = {
        'api_endpoint': VECTOR_SEARCH_API_ENDPOINT,
        'index_endpoint_path': VECTOR_SEARCH_INDEX_ENDPOINT_PATH,
        'deployed_index_id': VECTOR_SEARCH_DEPLOYED_INDEX_ID,
    }

    result: RetrieverResponse = await ai.retrieve(
        retriever='my-vector-search',
        query=query_document,
        options={'limit': 10},
    )

    end_time = time.time()

    duration = int(end_time - start_time)

    result_data = []
    for doc in result.documents:
        metadata = doc.metadata or {}
        result_data.append({
            'id': metadata.get('id'),
            'text': doc.content[0].root.text if doc.content and doc.content[0].root.text else '',
            'distance': metadata.get('distance', 0.0),
        })

    result_data = sorted(result_data, key=lambda x: x['distance'])

    return QueryFlowOutputSchema(
        result=result_data,
        length=len(result_data),
        time=duration,
    )


async def main() -> None:
    """Main function."""
    query_input = QueryFlowInputSchema(
        query='Content for doc',
        k=3,
    )

    result = await query_flow(query_input)
    await logger.ainfo(str(result))


if __name__ == '__main__':
    ai.run_main(main())
