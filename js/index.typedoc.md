Genkit is a framework for building AI-powered applications. This is the API reference for the Genkit JS libraries. For tutorials and guides, visit [genkit.dev](https://genkit.dev/docs/js/).

## Quick Start

Install the following Genkit dependencies to use Genkit in your project:

- `genkit` — Genkit core capabilities.
- A model plugin, e.g. `@genkit-ai/google-genai` for Google AI Gemini models.

```posix-terminal
npm install genkit @genkit-ai/google-genai
```

Set up your API key:

```posix-terminal
export GOOGLE_API_KEY=your-api-key
```

Make your first request:

```ts
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({ plugins: [googleAI()] });

const { text } = await ai.generate({
  model: googleAI.model('gemini-flash-latest'),
  prompt: 'Why is Genkit awesome?',
});

console.log(text);
```

## Packages

This reference documents the following packages:

| Package                                                                               | Description                                                              |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **[genkit](modules/genkit.html)**                                                     | Core framework — generation, flows, tools, prompts, streaming, and more. |
| **[@genkit-ai/google-genai](modules/_genkit-ai_google-genai.html)**                   | Google AI (Gemini) model plugin.                                         |
| **[@genkit-ai/vertexai](modules/_genkit-ai_vertexai.html)**                           | Vertex AI model plugin.                                                  |
| **[@genkit-ai/firebase](modules/_genkit-ai_firebase.html)**                           | Firebase integration (auth, Firestore, Cloud Functions).                 |
| **[@genkit-ai/express](modules/_genkit-ai_express.html)**                             | Serve flows as Express endpoints.                                        |
| **[@genkit-ai/google-cloud](modules/_genkit-ai_google-cloud.html)**                   | Google Cloud monitoring and telemetry.                                   |
| **[@genkit-ai/next](modules/_genkit-ai_next.html)**                                   | Next.js integration.                                                     |
| **[@genkit-ai/checks](modules/_genkit-ai_checks.html)**                               | Google Checks safety evaluation plugin.                                  |
| **[@genkit-ai/dev-local-vectorstore](modules/_genkit-ai_dev-local-vectorstore.html)** | Local vector store for development.                                      |
| **[@genkit-ai/evaluators](modules/_genkit-ai_evaluator.html)**                        | Built-in evaluators for testing AI output quality.                       |
| **[@genkit-ai/ollama](modules/genkitx-ollama.html)**                                  | Ollama local model plugin.                                               |
| **[@genkit-ai/chroma](modules/genkitx-chromadb.html)**                                | ChromaDB vector store plugin.                                            |
| **[@genkit-ai/pinecone](modules/genkitx-pinecone.html)**                              | Pinecone vector store plugin.                                            |
| **[@genkit-ai/mcp](modules/_genkit-ai_mcp.html)**                                     | Model Context Protocol (MCP) plugin.                                     |
| **[@genkit-ai/anthropic](modules/_genkit-ai_anthropic.html)**                         | Anthropic (Claude) model plugin.                                         |
| **[@genkit-ai/compat-oai](modules/_genkit-ai_compat-oai.html)**                       | OpenAI-compatible model plugin.                                          |
| **[@genkit-ai/fetch](modules/_genkit-ai_fetch.html)**                                 | HTTP fetch utilities for plugins.                                        |
| **[@genkit-ai/middleware](modules/_genkit-ai_middleware.html)**                       | Model middleware plugin (retry, caching, etc.).                          |

The `genkit` package also provides subpath imports for specific functionality:

| Import                    | Purpose                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------- |
| `genkit`                  | Main entry — `Genkit` class, `generate`, `defineFlow`, `defineTool`, schemas, types |
| `genkit/beta`             | Beta features including interrupts (`defineInterrupt`)                              |
| `genkit/beta/client`      | Client-side helpers (`runFlow`, `streamFlow`)                                       |
| `genkit/model/middleware` | Model middleware (`retry`, `fallback`, `augmentWithContext`, etc.)                  |
| `genkit/plugin`           | Plugin authoring utilities (`model`, `embedder`, `retriever`, etc.)                 |
| `genkit/model`            | Model types and helpers                                                             |
| `genkit/embedder`         | Embedder types                                                                      |
| `genkit/retriever`        | Retriever and indexer types                                                         |
| `genkit/reranker`         | Reranker types                                                                      |
| `genkit/evaluator`        | Evaluator types                                                                     |
| `genkit/tool`             | Tool types                                                                          |
| `genkit/schema`           | Schema utilities                                                                    |

## Key Features

### Structured Output

Generate strongly-typed, schema-validated output using Zod schemas:

```ts
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({ plugins: [googleAI()] });

const RecipeSchema = z.object({
  title: z.string(),
  ingredients: z.array(z.string()),
  instructions: z.array(z.string()),
});

const { output } = await ai.generate({
  model: googleAI.model('gemini-flash-latest'),
  prompt: 'Invent a new pasta recipe',
  output: { schema: RecipeSchema },
});

console.log(output?.title); // fully typed
```

### Streaming

Stream responses in real time with `generateStream`:

```ts
const { response, stream } = ai.generateStream({
  model: googleAI.model('gemini-flash-latest'),
  prompt: 'Write a short story about a robot',
});

for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}
```

### Tools (Function Calling)

Define tools that models can call automatically to access external data or perform actions:

```ts
const getWeather = ai.defineTool(
  {
    name: 'getWeather',
    description: 'Gets the current weather for a given city',
    inputSchema: z.object({ city: z.string() }),
    outputSchema: z.object({ temperature: z.number(), condition: z.string() }),
  },
  async ({ city }) => {
    // your implementation here
    return { temperature: 72, condition: 'sunny' };
  }
);

const { text } = await ai.generate({
  model: googleAI.model('gemini-flash-latest'),
  prompt: 'What should I wear in Tokyo today?',
  tools: [getWeather],
});
```

### Interrupts (Human-in-the-Loop)

> **Beta feature:** Interrupts require importing from `genkit/beta` instead of `genkit`:
>
> ```ts
> import { genkit } from 'genkit/beta';
> ```

Interrupts pause model processing and return control to the caller, enabling human-in-the-loop workflows. There are two patterns:

#### Basic Interrupts

Use `defineInterrupt` to create a tool that always pauses. The caller provides a response with `.respond()`:

```ts
const confirmAction = ai.defineInterrupt({
  name: 'confirmAction',
  description: 'Confirm an action with the user before proceeding',
  inputSchema: z.object({ action: z.string(), reason: z.string() }),
  outputSchema: z.object({ approved: z.boolean() }),
});

let response = await ai.generate({
  model: googleAI.model('gemini-flash-latest'),
  prompt: 'Book a table for 2 at 7pm tonight',
  tools: [confirmAction],
});

// The model triggered an interrupt — get user approval
if (response.interrupts.length) {
  const interrupt = response.interrupts[0];
  console.log(interrupt.toolRequest.input); // { action: '...', reason: '...' }

  // Resume with the user's response (bypasses tool execution)
  response = await ai.generate({
    model: googleAI.model('gemini-flash-latest'),
    messages: response.messages,
    tools: [confirmAction],
    resume: {
      respond: confirmAction.respond(interrupt, { approved: true }),
    },
  });
}
```

#### Restartable Tools

Regular tools can conditionally interrupt using `interrupt()` and be re-executed with `.restart()`. The `resumed` flag lets the tool know it's been approved:

```ts
const sendEmail = ai.defineTool(
  {
    name: 'sendEmail',
    description: 'Sends an email',
    inputSchema: z.object({ to: z.string(), body: z.string() }),
    outputSchema: z.object({ sent: z.boolean() }),
  },
  async (input, { interrupt, resumed }) => {
    if (!resumed) {
      interrupt({ message: `Send email to ${input.to}?` });
    }
    // Approved — proceed with sending
    return { sent: true };
  }
);

let response = await ai.generate({
  model: googleAI.model('gemini-flash-latest'),
  prompt: 'Send a hello email to alice@example.com',
  tools: [sendEmail],
});

if (response.interrupts.length) {
  const interrupt = response.interrupts[0];
  // Restart re-executes the tool, this time with resumed=true
  response = await ai.generate({
    model: googleAI.model('gemini-flash-latest'),
    messages: response.messages,
    tools: [sendEmail],
    resume: { restart: [sendEmail.restart(interrupt)] },
  });
}
```

### Prompts (Dotprompt)

Manage prompts as code with embedded schemas, model configuration, and Handlebars templating:

```
---
model: googleai/gemini-flash-latest
input:
  schema:
    topic: string
output:
  schema:
    title: string
    summary: string
---
Write a blog post about {{topic}}.
```

```ts
const blogPrompt = ai.prompt('blog');
const { output } = await blogPrompt({ topic: 'AI safety' });
```

### Flows

Build strongly typed, fully observable workflows that can be served as APIs and accessed from the client:

```ts
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-flash-latest'),
});

const RecipeSchema = z.object({
  title: z.string(),
  ingredients: z.array(z.string()),
  instructions: z.array(z.string()),
});

export const recipeFlow = ai.defineFlow(
  {
    name: 'recipeFlow',
    inputSchema: z.object({ ingredient: z.string() }),
    outputSchema: RecipeSchema,
  },
  async (input) => {
    const { output } = await ai.generate({
      prompt: `Create a recipe using ${input.ingredient}`,
      output: { schema: RecipeSchema },
    });
    if (!output) throw new Error('Failed to generate recipe');
    return output;
  }
);
```

Serve flows as an API:

```ts
import { startFlowServer } from '@genkit-ai/express'; // npm i @genkit-ai/express

startFlowServer({ flows: [recipeFlow] });
```

Access from the client:

```ts
import { streamFlow } from 'genkit/beta/client';

const { stream } = streamFlow({
  url: 'http://localhost:3500/recipeFlow',
  input: { ingredient: 'avocado' },
});

for await (const chunk of stream) {
  console.log(chunk);
}
```

### Middleware

Add common functionality to your AI requests with middleware (available in `genkit/model/middleware` and `@genkit-ai/middleware`):

```ts
import { retry } from 'genkit/model/middleware';

const { text } = await ai.generate({
  model: googleAI.model('gemini-flash-latest'),
  prompt: 'Why is Genkit awesome?',
  use: [
    retry({
      maxRetries: 3,
      initialDelayMs: 1000,
      backoffFactor: 2,
    }),
  ],
});
```

## More Resources

- **Full documentation:** [genkit.dev](https://genkit.dev/docs/js/)
- **Developer tools:** [CLI and Developer UI](https://genkit.dev/docs/js/devtools/)
- **Browse plugins:** [npmjs.com/search?q=keywords:genkit-plugin](https://www.npmjs.com/search?q=keywords:genkit-plugin)
- **Deployment:** [Express](https://genkit.dev/docs/js/deployment/any-platform/) · [Firebase](https://genkit.dev/docs/js/deployment/firebase/) · [Cloud Run](https://genkit.dev/docs/js/deployment/cloud-run/)
- **Community:** [Discord](https://discord.gg/qXt5zzQKpc) · [GitHub Issues](https://github.com/genkit-ai/genkit/issues)
