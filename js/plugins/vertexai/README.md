# Vertex AI plugin for Genkit

This package provides Vertex AI integrations for [Genkit](https://github.com/genkit-ai/genkit), including Model Garden and Evaluation.

> **Building with a coding agent? Install the Genkit JS skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-js
> ```
>
> It teaches your agent the current Genkit JS APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

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
| `@genkit-ai/vertexai/evaluation` | Vertex AI evaluation metrics (BLEU, ROUGE, SAFETY, etc.) |

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

## More information

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [official Genkit documentation](https://genkit.dev/docs/get-started/).

License: Apache 2.0
