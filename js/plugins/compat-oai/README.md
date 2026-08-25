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

```typescript
import dotenv from 'dotenv';
import { genkit } from 'genkit';
import openAI, { gpt35Turbo } from '@genkit-ai/compat-oai';

dotenv.config();

const ai = genkit({
  plugins: [openAI({ apiKey: process.env.OPENAI_API_KEY })],
  // specify a default model if not provided in generate params:
  model: gpt35Turbo,
});
```

### Basic examples

The simplest way to generate text is by using the `generate` method:

```typescript
const response = await ai.generate({
  model: gpt4o
  prompt: 'Tell me a joke.',
});

console.log(response.text);
```

### Multi-modal prompt

```typescript
const response = await ai.generate({
  model: gpt4o,
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
import { textEmbeddingAda002 } from '@genkit-ai/compat-oai';

const embedding = await ai.embed({
  embedder: textEmbeddingAda002,
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

### OpenAI Responses API

Models that OpenAI serves only over the [Responses API](https://platform.openai.com/docs/api-reference/responses), such as `gpt-5-pro`, `o1-pro`, `o3-pro` and the codex family, are routed there automatically:

```typescript
import { openAI } from '@genkit-ai/compat-oai/openai';

const response = await ai.generate({
  model: openAI.model('gpt-5-pro'),
  prompt: 'Tell me a joke.',
});
```

Models available on both transports default to Chat Completions. Opt a request into the Responses API with `openAI.responsesModel()`:

```typescript
const response = await ai.generate({
  model: openAI.responsesModel('gpt-4o'),
  prompt: 'Tell me a joke.',
});
```

or from dotprompt frontmatter, or any other place that carries config:

```yaml
model: openai/gpt-4o
config:
  transport: responses
```

Built-in OpenAI tools such as web search pass through `config.tools`:

```typescript
const response = await ai.generate({
  model: openAI.responsesModel('gpt-4o'),
  prompt: 'What happened in the news today?',
  config: { tools: [{ type: 'web_search' }] },
});
```

By default requests are stateless: the plugin pins `store: false` and asks OpenAI to return encrypted reasoning, so multi-turn conversations replay the full history from Genkit messages and nothing is retained server-side. To use server-side threading instead, opt into storage and thread responses yourself:

```typescript
const first = await ai.generate({
  model: openAI.responsesModel('gpt-4o'),
  prompt: 'Pick a number between 1 and 10.',
  config: { store: true },
});

const second = await ai.generate({
  model: openAI.responsesModel('gpt-4o'),
  prompt: 'Double it.',
  config: { store: true, previousResponseId: (first.raw as { id: string }).id },
});
```

When threading with `previousResponseId`, send only the new user turn; the server already holds the earlier context.

### Custom models & other Cloud providers

```typescript
import { GenerationCommonConfigSchema, genkit, z } from 'genkit';
import { ModelInfo } from 'genkit/model';
import openAI from '@genkit-ai/compat-oai';

const modelInfo: ModelInfo = {
  label: 'Claude - Claude 3.7 Sonnet',
  supports: {
    multiturn: true,
    tools: true,
    media: false,
    systemRole: true,
    output: ['json', 'text'],
  },
};
const schema = GenerationCommonConfigSchema.extend({});

const ai = genkit({
  plugins: [
    openAI({
      apiKey: process.env.ANTHROPIC_API_KEY,
      baseURL: 'https://api.anthropic.com/v1/',
      models: [
        { name: 'claude-3-7-sonnet', info: modelInfo, configSchema: schema },
      ],
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
      model: 'openai/claude-3-7-sonnet',
      config: {
        version: 'claude-3-7-sonnet-20250219',
      },
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
