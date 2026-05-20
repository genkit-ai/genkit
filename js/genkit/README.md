# Genkit

Genkit is a framework for building AI-powered applications. It provides open source libraries for Node.js and Go, along with tools to help you debug and iterate quickly.

## Quick Start

Install the following Genkit dependencies to use Genkit in your project:

- `genkit` - Genkit core capabilities.
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

// The model triggered an interrupt - get user approval
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
    // Approved - proceed with sending
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

The `toolApproval` middleware from `@genkit-ai/middleware` automates this pattern, interrupting any tool not in an approved list:

```ts
import { toolApproval } from '@genkit-ai/middleware';
import { restartTool } from 'genkit';

let response = await ai.generate({
  model: googleAI.model('gemini-flash-latest'),
  prompt: 'Send a hello email to alice@example.com',
  tools: [sendEmail, readInbox],
  use: [toolApproval({ approved: ['readInbox'] })], // sendEmail not approved
});

// sendEmail was interrupted - get user approval, then restart
if (response.interrupts.length) {
  response = await ai.generate({
    model: googleAI.model('gemini-flash-latest'),
    messages: response.messages,
    tools: [sendEmail, readInbox],
    resume: {
      restart: response.interrupts.map((i) =>
        restartTool(i, { toolApproved: true })
      ),
    },
    use: [toolApproval({ approved: ['readInbox'] })],
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

## Middleware

The [`@genkit-ai/middleware`](https://www.npmjs.com/package/@genkit-ai/middleware) package provides ready-made middleware to add common functionality to your AI requests:

- **`retry`** - Automatically retry failed requests with exponential backoff.
- **`fallback`** - Fall back to alternative models on specific error statuses.
- **`toolApproval`** - Restrict tool execution to an approved list, interrupting unapproved calls for review.
- **`filesystem`** - Give the model sandboxed read/write access to a directory on the filesystem.
- **`skills`** - Scan for skill definitions and inject them as available tools.

```posix-terminal
npm install @genkit-ai/middleware
```

```ts
import { retry } from '@genkit-ai/middleware';

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

## Developer Tools

Genkit comes with a powerful CLI and Developer UI for locally testing, debugging, and iterating on your AI features:

```posix-terminal
npx genkit start -- npx tsx src/index.ts
```

The Developer UI lets you visually test flows, inspect traces, and experiment with prompts - all in your browser.

## Plugins

Genkit supports a growing ecosystem of plugins for model providers, vector stores, and more:

| Category | Plugins |
|---|---|
| **Models** | `@genkit-ai/google-genai`, `@genkit-ai/vertexai`, `@genkit-ai/compat-oai`, `genkitx-anthropic`, `genkitx-ollama` |
| **Deployment** | `@genkit-ai/express`, `@genkit-ai/fetch`, `@genkit-ai/firebase`, `@genkit-ai/cloud-run` |
| **Monitoring** | `@genkit-ai/google-cloud` |

Browse all plugins: [npmjs.com/search?q=keywords:genkit-plugin](https://www.npmjs.com/search?q=keywords:genkit-plugin)

## Deployment

Genkit flows can be deployed anywhere Node.js runs:

- **Express** - [Deploy to Node.js](https://genkit.dev/docs/js/deployment/any-platform/)
- **Firebase** - [Deploy to Firebase](https://genkit.dev/docs/js/deployment/firebase/)
- **Cloud Run** - [Deploy to Cloud Run](https://genkit.dev/docs/js/deployment/cloud-run/)

## Next Steps

- [Developer tools](https://genkit.dev/docs/js/devtools/): Set up and use Genkit's CLI and developer UI.
- [Generating content](https://genkit.dev/docs/js/models/): Use Genkit's unified generation API.
- [Creating flows](https://genkit.dev/docs/js/flows/): Build observable workflows with rich debugging.
- [Managing prompts](https://genkit.dev/docs/js/dotprompt/): Manage prompts and configuration as code.

Learn more at [genkit.dev](https://genkit.dev)

License: Apache 2.0
