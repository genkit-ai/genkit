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

"""Example of using Genkit to fill VertexAI Index for Vector Search with BigQuery."""

import json
import os
from typing import Any

import structlog
from google.cloud import aiplatform, aiplatform_v1, bigquery

from genkit.ai import Genkit
from genkit.core.typing import DocumentPart
from genkit.plugins.google_genai import VertexAI
from genkit.plugins.vertex_ai import define_vertex_vector_search_big_query
from genkit.types import Document, TextPart

# Environment Variables
LOCATION = os.environ['LOCATION']
PROJECT_ID = os.environ['PROJECT_ID']
EMBEDDING_MODEL = 'gemini-embedding-001'

BIGQUERY_DATASET_NAME = os.environ['BIGQUERY_DATASET_NAME']
BIGQUERY_TABLE_NAME = os.environ['BIGQUERY_TABLE_NAME']

VECTOR_SEARCH_INDEX_ID = os.environ['VECTOR_SEARCH_INDEX_ID']

bq_client = bigquery.Client(project=PROJECT_ID)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

logger = structlog.get_logger(__name__)

ai = Genkit(plugins=[VertexAI()])

define_vertex_vector_search_big_query(
    ai,
    name='bigquery-vector-search',
    embedder=f'vertexai/{EMBEDDING_MODEL}',
    bq_client=bq_client,
    dataset_id=BIGQUERY_DATASET_NAME,
    table_id=BIGQUERY_TABLE_NAME,
)


@ai.flow(name='generateEmbeddings')
async def generate_embeddings() -> None:
    """Generates document embeddings and upserts them to the Vertex AI Vector Search index.

    This flow retrieves data from BigQuery, generates embeddings for the documents,
    and then upserts these embeddings to the specified Vector Search index.
    """
    toy_documents = [
        {
            'id': 'doc1',
            'content': {'title': 'Document 1', 'body': 'This is the content of document 1.'},
            'metadata': {'author': 'Alice', 'date': '2024-01-15'},
        },
        {
            'id': 'doc2',
            'content': {'title': 'Document 2', 'body': 'This is the content of document 2.'},
            'metadata': {'author': 'Bob', 'date': '2024-02-20'},
        },
        {
            'id': 'doc3',
            'content': {'title': 'Document 3', 'body': 'Content for doc 3'},
            'metadata': {'author': 'Charlie', 'date': '2024-03-01'},
        },
    ]

    create_bigquery_dataset_and_table(
        PROJECT_ID,
        LOCATION,
        BIGQUERY_DATASET_NAME,
        BIGQUERY_TABLE_NAME,
        toy_documents,
    )

    results_dict = get_data_from_bigquery(
        bq_client=bq_client,
        project_id=PROJECT_ID,
        dataset_id=BIGQUERY_DATASET_NAME,
        table_id=BIGQUERY_TABLE_NAME,
    )

    genkit_documents = [Document(content=[DocumentPart(root=TextPart(text=text))]) for text in results_dict.values()]

    embed_response = await ai.embed_many(
        embedder=f'vertexai/{EMBEDDING_MODEL}',
        content=genkit_documents,
        options={'task': 'RETRIEVAL_DOCUMENT', 'output_dimensionality': 128},
    )

    embeddings = [emb.embedding for emb in embed_response]

    ids = list(results_dict.keys())[: len(embeddings)]
    data_embeddings = list(zip(ids, embeddings, strict=True))

    upsert_data = [(str(id), embedding) for id, embedding in data_embeddings]
    upsert_index(PROJECT_ID, LOCATION, VECTOR_SEARCH_INDEX_ID, upsert_data)


def create_bigquery_dataset_and_table(
    project_id: str,
    location: str,
    dataset_id: str,
    table_id: str,
    documents: list[dict[str, Any]],
) -> None:
    """Creates a BigQuery dataset and table, and inserts documents.

    Args:
        project_id: The ID of the Google Cloud project.
        location: The location for the BigQuery resources.
        dataset_id: The ID of the BigQuery dataset.
        table_id: The ID of the BigQuery table.
        documents: A list of dictionaries, where each dictionary represents a document
            with 'id', 'content', and 'metadata' keys.  'content' and 'metadata'
            are expected to be JSON serializable.
    """
    client = bigquery.Client(project=project_id)
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = location

    try:
        dataset = client.create_dataset(dataset, exists_ok=True)
        logger.debug('Dataset %s.%s created.', client.project, dataset.dataset_id)
    except Exception as e:
        logger.exception('Error creating dataset: %s', e)
        raise e

    schema = [
        bigquery.SchemaField('id', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('content', 'JSON'),
        bigquery.SchemaField('metadata', 'JSON'),
    ]

    table_ref = dataset_ref.table(table_id)
    table = bigquery.Table(table_ref, schema=schema)
    try:
        table = client.create_table(table, exists_ok=True)
        logger.debug(
            'Table %s.%s.%s created.',
            table.project,
            table.dataset_id,
            table.table_id,
        )
    except Exception as e:
        logger.exception('Error creating table: %s', e)
        raise e

    rows_to_insert = [
        {
            'id': doc['id'],
            'content': json.dumps(doc['content']),
            'metadata': json.dumps(doc['metadata']),
        }
        for doc in documents
    ]

    errors = client.insert_rows_json(table, rows_to_insert)
    if errors:
        logger.error('Errors inserting rows: %s', errors)
        raise Exception(f'Failed to insert rows: {errors}')
    else:
        logger.debug('Inserted %s rows into BigQuery.', len(rows_to_insert))


def get_data_from_bigquery(
    bq_client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str,
) -> dict[str, str]:
    """Retrieves data from a BigQuery table.

    Args:
        bq_client:  The BigQuery client.
        project_id: The ID of the Google Cloud project.
        dataset_id: The ID of the BigQuery dataset.
        table_id: The ID of the BigQuery table.

    Returns:
        A dictionary where keys are document IDs and values are JSON strings
        representing the document content.
    """
    table_ref = bigquery.TableReference.from_string(f'{project_id}.{dataset_id}.{table_id}')
    query = f'SELECT id, content FROM `{table_ref}`'  # noqa: S608 - table ref from trusted config
    query_job = bq_client.query(query)
    rows = query_job.result()

    results = {row['id']: json.dumps(row['content']) for row in rows}
    logger.debug('Found %s rows with different ids into BigQuery.', len(results))

    return results


def upsert_index(
    project_id: str,
    region: str,
    index_name: str,
    data: list[tuple[str, list[float]]],
) -> None:
    """Upserts data points to a Vertex AI Index using batch processing.

    Args:
        project_id: The ID of your Google Cloud project.
        region: The region where the Index is located.
        index_name: The name of the Vertex AI Index.
        data: A list of tuples, where each tuple contains (id, embedding).
            id should be a string, and embedding should be a list of floats.
    """
    aiplatform.init(project=project_id, location=region)

    index_client = aiplatform_v1.IndexServiceClient(
        client_options={'api_endpoint': f'{region}-aiplatform.googleapis.com'}
    )

    index_path = index_client.index_path(project=project_id, location=region, index=index_name)

    datapoints = [aiplatform_v1.IndexDatapoint(datapoint_id=id, feature_vector=embedding) for id, embedding in data]

    logger.debug('Attempting to insert %s rows into Index %s', len(datapoints), index_path)

    upsert_request = aiplatform_v1.UpsertDatapointsRequest(index=index_path, datapoints=datapoints)

    index_client.upsert_datapoints(request=upsert_request)
    logger.info('Upserted %s datapoints.', len(datapoints))


async def main() -> None:
    """Main function."""
    await logger.ainfo(await generate_embeddings())


if __name__ == '__main__':
    ai.run_main(main())
