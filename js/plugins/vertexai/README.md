# A Vertex AI plugin for Genkit

## Installing the plugin

```bash
npm i --save @genkit-ai/vertexai
```

## Using the plugin

```ts
import { genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/vertexai';

const ai = genkit({
  plugins: [vertexAI()],
  model: vertexAI.model('gemini-2.5-flash'),
});

async () => {
  const { text } = ai.generate('hi Gemini!');
  console.log(text);
};
```

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [official Genkit documentation](https://genkit.dev/docs/get-started/).

License: Apache 2.0
