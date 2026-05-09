# Sample Vertex AI Plugin Reranker with Fake Document Content

This sample app demonstrates the use of the Vertex AI plugin for reranking a set of documents based on a query using fake document content. This guide will walk you through setting up and running the sample.

## Prerequisites

Before running this sample, ensure you have the following:

1. **Node.js** installed.
2. **PNPM** (Node Package Manager) installed.
3. A **Vertex AI** project with appropriate permissions for reranking models.

## Getting Started

### Step 1: Clone the Repository and Install Dependencies

Clone this repository to your local machine and navigate to the project directory. Then, install the necessary dependencies:

```bash
pnpm install
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the root directory and set the following variables. You can use the provided `.env.example` as a reference.

```plaintext
PROJECT_ID=your_project_id_here
LOCATION=your_location_here
```

These variables are required to configure the Vertex AI project and location for reranking.

### Step 3: Run the Sample

Start the Genkit server:

```bash
genkit start
```

This will launch the server that hosts the reranking flow.

## Sample Explanation

### Overview

This sample demonstrates how to use the Vertex AI plugin to rerank a predefined list of fake document content based on a query input. It utilizes a semantic reranker model from Vertex AI.

### Key Components

- **Fake Document Content**: A hardcoded array of strings representing document content.
- **Rerank Flow**: A flow that reranks the fake documents based on the provided query.
- **Genkit Configuration**: Configures Genkit with the Vertex AI plugin, setting up the project and reranking model.

### Rerank Flow

The `rerankFlow` function takes a query as input, reranks the predefined document content using the Vertex AI semantic reranker, and returns the documents sorted by relevance score.

```typescript
export const rerankFlow = defineFlow(
{
name: 'rerankFlow',
inputSchema: z.object({ query: z.string() }),
outputSchema: z.array(
z.object({
text: z.string(),
score: z.number(),
})
),
},
async ({ query }) => {
const documents = FAKE_DOCUMENT_CONTENT.map((text) =>
Document.fromText(text)
);
const reranker = 'vertexai/reranker';

    const rerankedDocuments = await rerank({
      reranker,
      query: Document.fromText(query),
      documents,
    });

    return rerankedDocuments.map((doc) => ({
      text: doc.text,
      score: doc.metadata.score,
    }));

}
);
```

### Running the Server

The server is started using the `startFlowsServer` function, which sets up the Genkit server to handle flow requests.

```typescript
startFlowsServer();
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Conclusion

This sample provides a basic demonstration of using the Vertex AI plugin with Genkit for reranking documents based on a query. It can be extended and adapted to suit more complex use cases and integrations with other data sources and services.

For more information, please refer to the official [official Genkit documentation](https://genkit.dev/docs/get-started/).
