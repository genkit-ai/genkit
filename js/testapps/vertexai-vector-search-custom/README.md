# Sample Vertex AI Plugin Retriever and Indexer with Local File

This sample app demonstrates the use of the Vertex AI plugin retriever and indexer with a local file for demonstration purposes. This guide will walk you through setting up and running the sample.

## Prerequisites

Before running this sample, ensure you have the following:

1. **Node.js** installed.
2. **PNPM** (Node Package Manager) installed.
3. A deployed index to an index endpoint in **Vertex AI Vector Search**.

## Getting Started

### Step 1: Clone the Repository and Install Dependencies

Clone this repository to your local machine, and follow the instructions in the root README.md to build
the core packages. This sample uses `workspace:*` dependencies, so they will need to be accessible.

Then

```bash
cd js/testapps/vertex-vector-search-custom && pnpm i
```

### Step 3: Set Up Environment Variables

Ensure you have a deployed index in Vertex AI Vector Search.

Create a `.env` file in the root directory and set the following variables (see the .env.example as well if needed)

```plaintext
PROJECT_ID=your-google-cloud-project-id
LOCATION=your-vertex-ai-location
LOCAL_DIR=./data
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

This sample demonstrates how to define a custom document indexer and retriever using local JSON files. It integrates with Vertex AI for indexing and retrieval of documents.

### Key Components

- **Custom Document Indexer**: Stores documents in a local JSON file.
- **Custom Document Retriever**: Retrieves documents from the local JSON file based on neighbor IDs.
- **Genkit Configuration**: Configures Genkit with the Vertex AI plugin, setting up the project, location, and vector search index options.
- **Indexing Flow**: Defines a flow for indexing documents.
- **Query Flow**: Defines a flow for querying indexed documents.

### Custom Document Indexer

The `localDocumentIndexer` function reads existing documents from a local file, adds new documents, and writes them back to the file:

```typescript
const localDocumentIndexer: DocumentIndexer = async (documents: Document[]) => {
  const content = await fs.promises.readFile(localFilePath, 'utf-8');
  const currentLocalFile = JSON.parse(content);
  const docsWithIds = Object.fromEntries(
    documents.map((doc) => [
      generateRandomId(),
      { content: JSON.stringify(doc.content) },
    ])
  );
  const newLocalFile = { ...currentLocalFile, ...docsWithIds };
  await fs.promises.writeFile(
    localFilePath,
    JSON.stringify(newLocalFile, null, 2)
  );
  return Object.keys(docsWithIds);
};
```

### Custom Document Retriever

The `localDocumentRetriever` function reads the local file and retrieves documents based on neighbor IDs:

```typescript
const localDocumentRetriever: DocumentRetriever = async (
  neighbors: Neighbor[]
) => {
  const content = await fs.promises.readFile(localFilePath, 'utf-8');
  const currentLocalFile = JSON.parse(content);
  const ids = neighbors
    .map((neighbor) => neighbor.datapoint?.datapointId)
    .filter(Boolean) as string[];
  const docs = ids
    .map((id) => {
      const doc = currentLocalFile[id];
      if (!doc || !doc.content) return null;
      const parsedContent = JSON.parse(doc.content);
      const text = parsedContent[0]?.text;
      return text ? Document.fromText(text) : null;
    })
    .filter(Boolean) as Document[];
  return docs;
};
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
