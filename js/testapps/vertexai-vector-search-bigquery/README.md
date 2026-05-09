# Sample Vertex AI Plugin Retriever and Indexer with BigQuery

This sample app demonstrates the use of the Vertex AI plugin retriever and indexer with BigQuery for storing document content and metadata. This guide will walk you through setting up and running the sample.

## Prerequisites

Before running this sample, ensure you have the following:

1. **Node.js** installed.
2. **PNPM** (Node Package Manager) installed.
3. A deployed index to an index endpoint in **Vertex AI Vector Search**.
4. A BigQuery dataset and table with columns `id` (string), `content` (JSON), `metadata` (JSON).

## Getting Started

### Step 1: Clone the Repository and Install Dependencies

Clone this repository to your local machine, and follow the instructions in the root README.md to build
the core packages. This sample uses `workspace:*` dependencies, so they will need to be accessible.

Then

```bash
cd js/testapps/vertex-vector-search-bigquery && pnpm i
```

### Step 3: Set Up Environment Variables

Ensure you have a deployed index in Vertex AI Vector Search.

**Important**: This plugin only supports streaming update indexes, not batch update indexes.

Create a `.env` file in the root directory and set the following variables (see the .env.example as well if needed)

```plaintext
PROJECT_ID=your-google-cloud-project-id
LOCATION=your-vertex-ai-location
BIGQUERY_DATASET=your_bigquery_dataset_here
BIGQUERY_TABLE=your_bigquery_table_here
VECTOR_SEARCH_PUBLIC_DOMAIN_NAME=your-vector-search-public-domain-name
VECTOR_SEARCH_INDEX_ENDPOINT_ID=your-index-endpoint-id
VECTOR_SEARCH_INDEX_ID=your-index-id
VECTOR_SEARCH_DEPLOYED_INDEX_ID=your-deployed-index-id
GOOGLE_APPLICATION_CREDENTIALS=path-to-your-service-account-key.json
```

### Step 4: Run the Sample

Start the Genkit server:

```bash
genkit start
```

## Sample Explanation

### Overview

This sample demonstrates how to define a custom document indexer and retriever using BigQuery. It integrates with Vertex AI for indexing and retrieval of documents.

### Key Components

- **BigQuery Document Indexer**: Stores documents in a BigQuery table
- **BigQuery Document Retriever**: Retrieves documents from the BigQuery table based on neighbor IDs.
- **Genkit Configuration**: Configures Genkit with the Vertex AI plugin, setting up the project, location, and vector search index options.
- **Indexing Flow**: Defines a flow for indexing documents.
- **Query Flow**: Defines a flow for querying indexed documents.

### BigQuery Document Indexer

The `bigQueryDocumentIndexer` function writes documents from BigQuery, returning generated datapoint ids for Vertex AI Vector Search.

```typescript
const bigQueryDocumentIndexer: DocumentIndexer = getBigQueryDocumentIndexer(
  bq,
  BIGQUERY_TABLE,
  BIGQUERY_DATASET
);
```

### BigQuery Retriever

The `bigQueryDocumentRetriever` function queries BigQuery, and retrieves documents based on neighbor IDs:

```typescript
const bigQueryDocumentRetriever: DocumentRetriever =
  getBigQueryDocumentRetriever(bq, BIGQUERY_TABLE, BIGQUERY_DATASET);
```

### Defining Flows

Two flows are defined: `indexFlow` for indexing documents and `queryFlow` for querying documents.

- **Index Flow**: Converts text inputs to documents and indexes them.
- **Query Flow**: Retrieves documents based on a query and returns the results sorted by distance.

### Running the Server

The server is started using the `startFlowsServer` function, which sets up the Genkit server to handle flow requests.

```typescript
startFlowsServer();
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Conclusion

This sample provides a basic demonstration of using Vertex AI plugins with Genkit for document indexing and retrieval. It can be extended and adapted to suit more complex use cases and integrations with other data sources and services.

For more information, please refer to the official [official Genkit documentation](https://genkit.dev/docs/get-started/).
