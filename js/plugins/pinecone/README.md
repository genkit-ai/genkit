# Pinecone plugin for Genkit

## Installing the plugin

```bash
npm i --save genkitx-pinecone
```

## Using the plugin

```ts
import { genkit } from 'genkit';
import {
  pinecone,
  pineconeRetrieverRef,
  pineconeIndexerRef,
} from 'genkitx-pinecone';

const ai = genkit({
  plugins: [
    pinecone([
      {
        indexId: 'bob-facts',
        embedder: textEmbedding004,
      },
    ]),
  ],
});

export const bobFactsIndexer = pineconeIndexerRef({
  indexId: 'bob-facts',
});
await ai.index({ indexer: bobFactsIndexer, documents });

// To specify an index:
export const bobFactsRetriever = pineconeRetrieverRef({
  indexId: 'bob-facts',
});

// To use the index you configured when you loaded the plugin:
let docs = await ai.retrieve({ retriever: pineconeRetrieverRef, query });
```

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [official Genkit documentation](https://genkit.dev/docs/get-started/).

License: Apache 2.0
