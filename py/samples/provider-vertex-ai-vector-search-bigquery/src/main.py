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

"""Vertex AI Vector Search with BigQuery sample.

This sample demonstrates how to use Vertex AI Vector Search with BigQuery
as the document store for large-scale analytics-friendly vector search.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ BigQuery            │ Google's data warehouse. Store and query           │
    │                     │ massive amounts of data (petabytes!) fast.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Vertex AI Vector    │ Google's enterprise vector search. Handles         │
    │ Search              │ billions of vectors with sub-second latency.       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Dataset/Table       │ BigQuery organization. Dataset = folder,           │
    │                     │ Table = spreadsheet with your data.                │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Deployed Index      │ Your vector index running in the cloud.            │
    │                     │ Ready 24/7 to answer similarity queries.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Analytics + AI      │ Combine SQL analytics with AI search.              │
    │                     │ The best of both worlds!                           │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow (BigQuery + Vector Search)::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │          HOW BIGQUERY + VERTEX VECTOR SEARCH WORK TOGETHER              │
    │                                                                         │
    │    Query: "Find similar products to this one"                           │
    │         │                                                               │
    │         │  (1) Convert to embedding                                     │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Embedder       │   Query → [0.3, -0.2, 0.7, ...]                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Search Vector Index                                  │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Vertex AI      │   Returns IDs: ["prod_1", "prod_2", ...]         │
    │    │  Vector Search  │   (ranked by similarity)                         │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) Fetch from BigQuery (with extra analytics)           │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  BigQuery       │   JOIN with sales data, filter by region,        │
    │    │  (SQL query)    │   add business logic, return full records        │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (4) Rich results with analytics context                  │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Your App       │   Products + sales data + similarity scores      │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| BigQuery Vector Search Definition       | `define_vertex_vector_search_big_query`|
| BigQuery Client Integration             | `bigquery.Client()`                 |
| Document Retrieval with Filters         | `ai.retrieve(..., options={'limit': ...})`|
| Performance Metrics                     | Duration tracking                   |

Testing This Demo
=================
1. **Prerequisites** - Set up GCP resources:
   ```bash
   # Required environment variables
   export LOCATION=us-central1
   export PROJECT_ID=your_project_id
   export BIGQUERY_DATASET_NAME=your_dataset
   export BIGQUERY_TABLE_NAME=your_table
   export VECTOR_SEARCH_DEPLOYED_INDEX_ID=your_deployed_index_id
   export VECTOR_SEARCH_INDEX_ENDPOINT_PATH=your_endpoint_path
   export VECTOR_SEARCH_API_ENDPOINT=your_api_endpoint

   # Authenticate with GCP
   gcloud auth application-default login
   ```

2. **GCP Setup Required**:
   - Create Vertex AI Vector Search index
   - Deploy index to an endpoint
   - Create BigQuery dataset and table with embeddings
   - Ensure table schema matches expected format

3. **Run the demo**:
   ```bash
   cd py/samples/provider-vertex-ai-vector-search-bigquery
   ./run.sh
   ```

4. **Open DevUI** at http://localhost:4000

5. **Test the flows**:
   - [ ] `retrieve_documents` - Vector similarity search
   - [ ] Test with limit options
   - [ ] Check performance metrics in output

6. **Expected behavior**:
   - Query is embedded and sent to Vector Search
   - Similar vectors are found and IDs returned
   - BigQuery is queried for full document content
   - Duration metrics show performance
"""

import os
import time

from google.cloud import aiplatform, bigquery
from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.blocks.document import Document
from genkit.core.logging import get_logger
from genkit.plugins.google_genai import VertexAI
from genkit.plugins.vertex_ai import define_vertex_vector_search_big_query
from samples.shared.logging import setup_sample

setup_sample()

LOCATION = os.getenv('LOCATION')
PROJECT_ID = os.getenv('PROJECT_ID')

BIGQUERY_DATASET_NAME = os.getenv('BIGQUERY_DATASET_NAME')
BIGQUERY_TABLE_NAME = os.getenv('BIGQUERY_TABLE_NAME')

VECTOR_SEARCH_DEPLOYED_INDEX_ID = os.getenv('VECTOR_SEARCH_DEPLOYED_INDEX_ID')
VECTOR_SEARCH_INDEX_ENDPOINT_PATH = os.getenv('VECTOR_SEARCH_INDEX_ENDPOINT_PATH')
VECTOR_SEARCH_API_ENDPOINT = os.getenv('VECTOR_SEARCH_API_ENDPOINT')

bq_client = bigquery.Client(project=PROJECT_ID)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

logger = get_logger(__name__)

ai = Genkit(plugins=[VertexAI()])

# Define Vertex AI Vector Search with BigQuery
define_vertex_vector_search_big_query(
    ai,
    name='my-vector-search',
    embedder='vertexai/gemini-embedding-001',
    embedder_options={
        'task': 'RETRIEVAL_DOCUMENT',
        'output_dimensionality': 128,
    },
    bq_client=bq_client,
    dataset_id=BIGQUERY_DATASET_NAME or 'default_dataset',
    table_id=BIGQUERY_TABLE_NAME or 'default_table',
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

    response = await ai.retrieve(
        retriever='my-vector-search',
        query=query_document,
        options={'limit': 10},
    )

    end_time = time.time()

    duration = int(end_time - start_time)

    result_data = []
    for doc in response.documents:
        metadata = doc.metadata or {}
        result_data.append({
            'id': metadata.get('id'),
            'text': doc.content[0].root.text if doc.content and doc.content[0].root.text else '',
            'distance': metadata.get('distance'),
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
