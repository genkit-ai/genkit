# Cloud SQL for PostgreSQL plugin for Genkit

## Installing the plugin

```bash
npm i --save genkitx-cloud-sql-pg
```

## Using the plugin

### Initialize the Postgres engine

```ts
import { PostgresEngine } from '@genkit-ai/cloud-sql-pg';

// Initialize the engine
const engine = await PostgresEngine.fromInstance(
  'your-project-id',
  'your-region',
  'your-instance-name',
  'your-database-name',
  {
    user: 'your-username',
    password: 'your-password',
  }
);
```

### Indexer usage

```ts
import { genkit } from 'genkit';
import {
  postgres,
  postgresRetrieverRef,
  postgresIndexerRef,
} from '@genkit-ai/cloud-sql-pg';
import { Document } from 'genkit/retriever';

const ai = genkit({
  plugins: [
    postgres([
      {
        tableName: 'my-documents',
        engine: engine,
        embedder: textEmbedding004,
        schemaName: 'public',
        contentColumn: 'content',
        embeddingColumn: 'embedding',
        idColumn: 'custom_id',
        metadataColumns: ['source', 'category'],
        ignoreMetadataColumns: ['created_at', 'updated_at'],
        metadataJsonColumn: 'metadata',
      },
    ]),
  ],
});

// To index documents:
export const myDocumentsIndexer = postgresIndexerRef({
  tableName: 'my-documents',
});

// Example of indexing documents with different options
const documents = [
  new Document({
    content: [{ text: 'This is a test document' }],
    metadata: { source: 'test', category: 'example' },
  }),
  new Document({
    content: [{ text: 'Another test document' }],
    metadata: { source: 'test', category: 'example' },
  }),
];

// Index with default options
await ai.index({
  indexer: myDocumentsIndexer,
  documents,
});

// Index with custom batch size
await ai.index({
  indexer: myDocumentsIndexer,
  documents,
  options: { batchSize: 10 },
});

// Index with custom ID from metadata
const docWithCustomId = new Document({
  content: [{ text: 'Document with custom ID' }],
  metadata: {
    source: 'test',
    customId: 'custom-123', // This will be used as the document ID
  },
});
await ai.index({
  indexer: myDocumentsIndexer,
  documents: [docWithCustomId],
});
```

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [Genkit documentation](https://genkit.dev/docs/plugins/cloud-sql-pg/).

License: Apache 2.0
