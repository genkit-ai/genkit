# genkitx-valkey

Genkit AI framework plugin for Valkey vector database.

This plugin provides `Indexer` and `Retriever` implementations backed by
[Valkey](https://valkey.io/) with the
[valkey-search](https://github.com/valkey-io/valkey-search) module loaded.

## Prerequisites

Your Valkey instance must have the `valkey-search` module loaded. This enables
the `FT.CREATE` and `FT.SEARCH` commands used for vector indexing and KNN
retrieval.

- **Docker**: use `valkey/valkey-bundle` which bundles the module.
- **ElastiCache**: engine versions that include `valkey-search` work
  out-of-the-box.
- **Manual**: pass `--loadmodule /path/to/libsearch.so` at startup.

## Installation

```bash
pnpm add genkitx-valkey
```

## Usage

```typescript
import { genkit } from 'genkit';
import { valkeyPlugin, valkeyIndexer, valkeyRetriever } from 'genkitx-valkey';

const ai = genkit({
  plugins: [
    valkeyPlugin([
      {
        indexName: 'docs',
        embedder: 'googleai/gemini-embedding-001',
        dimension: 768,
        clientConfig: {
          addresses: [{ host: 'localhost', port: 6379 }],
        },
      },
    ]),
  ],
});

// Index documents
await ai.index({
  indexer: valkeyIndexer({ indexName: 'docs' }),
  documents: [
    { content: [{ text: 'Valkey is an open-source key-value store.' }] },
  ],
});

// Retrieve documents
const docs = await ai.retrieve({
  retriever: valkeyRetriever({ indexName: 'docs' }),
  query: 'What is Valkey?',
  options: { k: 3 },
});
```

## Configuration

Each entry in the plugin params array accepts:

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `indexName` | `string` | Yes | Name of the FT index in Valkey |
| `embedder` | `EmbedderArgument` | Yes | Genkit embedder reference |
| `dimension` | `number` | Yes | Embedding vector dimension |
| `clientConfig` | `GlideClientConfiguration` | Yes | Connection config for `@valkey/valkey-glide` |
| `embedderOptions` | `object` | No | Options passed to the embedder |
| `prefix` | `string` | No | Hash key prefix (defaults to `indexName`) |
| `distanceMetric` | `'COSINE' \| 'L2' \| 'IP'` | No | HNSW distance metric (defaults to `COSINE`) |
| `metadataFields` | `ValkeyMetadataField[]` | No | Metadata fields to index for query-time filtering |

## License

Apache-2.0
