# OpenAI Compatible API plugin for Genkit

**`@genkit-ai/compat-oai`** is a plugin for using OpenAI Compatible APIs with [Genkit](https://genkit.dev).

This Genkit plugin allows to use OpenAI models through their official APIs.

Official OpenAI-compatible provider documentation:

- [OpenAI](https://genkit.dev/docs/integrations/openai/)
- [xAI (Grok)](https://genkit.dev/docs/integrations/xai/)
- [DeepSeek](https://genkit.dev/docs/integrations/deepseek/)
- [Other compatible APIs](https://genkit.dev/docs/integrations/openai-compatible/)

## Supported models

This plugin also supports OpenAI models, and custom models from other model providers.

## Installation

Install the plugin in your project with your favorite package manager:

- `npm install @genkit-ai/compat-oai`
- `yarn add @genkit-ai/compat-oai`
- `pnpm add @genkit-ai/compat-oai`

## Usage

### Initialize

Import the OpenAI plugin from `@genkit-ai/compat-oai/openai` (not the package root). The package root exports `openAICompatible` for custom OpenAI-compatible providers.

```typescript
import dotenv from 'dotenv';
import { genkit } from 'genkit';
import { openAI } from '@genkit-ai/compat-oai/openai';

dotenv.config();

const ai = genkit({
  plugins: [openAI({ apiKey: process.env.OPENAI_API_KEY })],
  // specify a default model if not provided in generate params:
  model: openAI.model('gpt-3.5-turbo'),
});
```

### Basic examples

The simplest way to generate text is by using the `generate` method:

```typescript
const response = await ai.generate({
  model: openAI.model('gpt-4o'),
  prompt: 'Tell me a joke.',
});

console.log(response.text);
```

### Multi-modal prompt

```typescript
const response = await ai.generate({
  model: openAI.model('gpt-4o'),
  prompt: [
    { text: 'What animal is in the photo?' },
    { media: { url: imageUrl } },
  ],
  config: {
    // control of the level of visual detail when processing image embeddings
    // Low detail level also decreases the token usage
    visualDetailLevel: 'low',
  },
});
console.log(response.text);
```

### Text Embeddings

```typescript
const embedding = await ai.embed({
  embedder: openAI.embedder('text-embedding-ada-002'),
  content: 'Hello world',
});

console.log(embedding);
```

### Within a flow

```typescript
import { z } from 'genkit';

export const jokeFlow = ai.defineFlow(
  {
    name: 'jokeFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (subject) => {
    const llmResponse = await ai.generate({
      prompt: `tell me a joke about ${subject}`,
    });
    return llmResponse.text;
  }
);
```

### Tool use

```typescript
import { z } from 'genkit';

// ...initialize genkit (as shown above)

const createReminder = ai.defineTool(
  {
    name: 'createReminder',
    description: 'Use this to create reminders for things in the future',
    inputSchema: z.object({
      time: z
        .string()
        .describe('ISO timestamp string, e.g. 2024-04-03T12:23:00Z'),
      reminder: z.string().describe('the content of the reminder'),
    }),
    outputSchema: z.number().describe('the ID of the created reminder'),
  },
  (reminder) => Promise.resolve(3)
);

const result = await ai.generate({
  tools: [createReminder],
  prompt: `
  You are a reminder assistant.
  If you create a reminder, describe in text the reminder you created as a response.

  Query: I have a meeting with Anna at 3 for dinner - can you set a reminder for the time?
  `,
});

console.log(result.text);
```

### Custom models & other Cloud providers

For non-OpenAI OpenAI-compatible APIs, use `openAICompatible` from the package root and provide a plugin `name`:

```typescript
import { genkit, z } from 'genkit';
import {
  compatOaiModelRef,
  defineCompatOpenAIModel,
  openAICompatible,
} from '@genkit-ai/compat-oai';

const ai = genkit({
  plugins: [
    openAICompatible({
      name: 'openrouter',
      apiKey: process.env.OPENROUTER_API_KEY,
      baseURL: 'https://openrouter.ai/api/v1',
      async initializer(client) {
        return [
          defineCompatOpenAIModel({
            name: 'openrouter/z-ai/glm-4.5-air:free',
            client,
            modelRef: compatOaiModelRef({
              name: 'openrouter/z-ai/glm-4.5-air:free',
            }),
          }),
        ];
      },
    }),
  ],
});

export const customModelFlow = ai.defineFlow(
  {
    name: 'customModelFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (subject) => {
    const llmResponse = await ai.generate({
      prompt: `tell me a joke about ${subject}`,
      model: compatOaiModelRef({
        name: 'openrouter/z-ai/glm-4.5-air:free',
      }),
    });
    return llmResponse.text;
  }
);
```

## Contributing

Want to contribute to the project? That's awesome! Head over to our [Contribution Guidelines](https://github.com/genkit-ai/genkit/blob/main/CONTRIBUTING.md).

## Need help?

 - [Genkit Discord](https://genkit.dev/discord)
 - [Github Issues](https://github.com/genkit-ai/genkit/issues)

## License

This project is licensed under the [Apache 2.0 License](https://github.com/genkit-ai/genkit/blob/main/LICENSE).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202%2E0-lightgrey.svg)](https://github.com/genkit-ai/genkit/blob/main/LICENSE)
