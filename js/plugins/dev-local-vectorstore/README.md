# Dev Local Vector Store for Genkit

This is a simple implementation of a vector store that can be used to local development and testing.

> **Building with a coding agent? Install the Genkit JS skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-js
> ```
>
> It teaches your agent the current Genkit JS APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

This plugin is not meant to be used in production.

## Installing the plugin

```bash
npm i --save @genkit-ai/dev-local-vectorstore
```

## Using the plugin

```ts
import { Document, genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';
import {
  devLocalVectorstore,
  devLocalIndexerRef,
  devLocalRetrieverRef,
} from '@genkit-ai/dev-local-vectorstore';

const ai = genkit({
  plugins: [
    googleAI(),
    devLocalVectorstore([
      {
        indexName: 'BobFacts',
        embedder: googleAI.embedder('gemini-embedding-001'),
      },
    ]),
  ],
  model: googleAI.model('gemini-flash-latest'),
});

// Reference to a local vector database storing Genkit documentation
const indexer = devLocalIndexerRef('BobFacts');
const retriever = devLocalRetrieverRef('BobFacts');

async function main() {
  // Add documents to the index. Only do it once.
  await ai.index({
    indexer: indexer,
    documents: [
      Document.fromText('Bob lives on the moon.'),
      Document.fromText('Bob is 42 years old.'),
      Document.fromText('Bob likes bananas.'),
      Document.fromText('Bob has 11 cats.'),
    ],
  });

  const question = 'How old is Bob?';

  // Consistent API to retrieve most relevant documents based on semantic similarity to query
  const docs = await ai.retrieve({
    retriever: retriever,
    query: question,
  });

  const result = await ai.generate({
    prompt: `Use the provided context from the Genkit documentation to answer this query: ${question}`,
    docs, // Pass retrieved documents to the model
  });

  console.log(result.text);
}

main();
```

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [official Genkit documentation](https://genkit.dev/docs/get-started/).

License: Apache 2.0
