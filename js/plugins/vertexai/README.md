# Vertex AI plugin for Genkit

This package provides Vertex AI integrations for [Genkit](https://github.com/genkit-ai/genkit), including Model Garden, Rerankers, Evaluation, and Vector Search.

> **⚠️ Deprecation notice:** The main `vertexAI` plugin export (Gemini, Imagen, and embedder models) is **deprecated**. Please migrate to [`@genkit-ai/google-genai`](https://www.npmjs.com/package/@genkit-ai/google-genai):
>
> ```ts
> // Before (deprecated)
> import { vertexAI } from '@genkit-ai/vertexai';
>
> // After
> import { vertexAI } from '@genkit-ai/google-genai';
> ```

## Installation

```bash
npm i --save @genkit-ai/vertexai
```

## Sub-packages

This package provides the following sub-package imports:

| Import | Description |
|---|---|
| `@genkit-ai/vertexai/modelgarden` | Third-party models (Claude, Mistral, Llama) via Vertex AI Model Garden |
| `@genkit-ai/vertexai/rerankers` | Vertex AI Rerankers API |
| `@genkit-ai/vertexai/evaluation` | Vertex AI evaluation metrics (BLEU, ROUGE, SAFETY, etc.) |
| `@genkit-ai/vertexai/vectorsearch` | Vertex AI Vector Search with BigQuery and Firestore backends |

### Model Garden

Access third-party models (Anthropic Claude, Mistral, Llama) hosted on Vertex AI Model Garden:

```ts
import { genkit } from 'genkit';
import { vertexModelGarden } from '@genkit-ai/vertexai/modelgarden';

const ai = genkit({
  plugins: [
    vertexModelGarden({ projectId: 'my-project', location: 'us-central1' }),
  ],
});

const { text } = await ai.generate({
  model: vertexModelGarden.model('claude-sonnet-4'),
  prompt: 'Write a haiku about cloud computing',
});

console.log(text);
```

### Rerankers

Use Vertex AI's reranking API to reorder documents by relevance:

```ts
import { genkit } from 'genkit';
import { vertexRerankers } from '@genkit-ai/vertexai/rerankers';

const ai = genkit({
  plugins: [
    vertexRerankers({ projectId: 'my-project', location: 'us-central1' }),
  ],
});
```

### Evaluation

Evaluate AI output quality using Vertex AI's built-in metrics:

```ts
import { vertexAIEvaluation } from '@genkit-ai/vertexai/evaluation';
import { VertexAIEvaluationMetricType } from '@genkit-ai/vertexai/evaluation';

const ai = genkit({
  plugins: [
    vertexAIEvaluation({
      projectId: 'my-project',
      location: 'us-central1',
      metrics: [
        VertexAIEvaluationMetricType.BLEU,
        VertexAIEvaluationMetricType.ROUGE,
        VertexAIEvaluationMetricType.SAFETY,
        VertexAIEvaluationMetricType.GROUNDEDNESS,
      ],
    }),
  ],
});
```

### Vector Search

Use Vertex AI Vector Search for retrieval-augmented generation (RAG) with BigQuery or Firestore document stores:

```ts
import { vertexAIVectorSearch } from '@genkit-ai/vertexai/vectorsearch';

const ai = genkit({
  plugins: [
    vertexAIVectorSearch({
      projectId: 'my-project',
      location: 'us-central1',
      vectorSearchOptions: [
        {
          publicDomainName: 'my-public-endpoint.vdb.vertexai.goog',
          indexEndpointId: 'my-index-endpoint-id',
          indexId: 'my-index-id',
          deployedIndexId: 'my-deployed-index-id',
          documentRetriever: myDocRetriever,
          documentIndexer: myDocIndexer,
          embedder: myEmbedder,
        },
      ],
    }),
  ],
});
```

## More information

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [official Genkit documentation](https://genkit.dev/docs/get-started/).

License: Apache 2.0
