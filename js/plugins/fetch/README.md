# Genkit Web Fetch Plugin

This plugin provides utilities for exposing Genkit actions (flows, models, etc.) over the **Web Fetch API** (`Request` / `Response`). Use it with any runtime or framework that supports the standard Fetch API (such as Hono, Bun, Cloudflare Workers, Deno, Node (18+), Vercel Edge, Netlify Edge, Elysia, SvelteKit, etc). Express-like API: pass the action first, then call the returned handler with the request.

No framework-specific dependencies; only `genkit` and the standard Web APIs.

## Installation

```bash
npm i @genkit-ai/fetch
```

## Usage (Hono)

### Single action with `fetchHandler`

```ts
import { fetchHandler } from '@genkit-ai/fetch';
import { Hono } from 'hono';

const simpleFlow = ai.defineFlow('simpleFlow', async (input, { sendChunk }) => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-2.0-flash'),
    prompt: input,
    onChunk: (c) => sendChunk(c.text),
  });
  return text;
});

const app = new Hono();
app.all('/simpleFlow', (c) => fetchHandler(simpleFlow)(c.req.raw));
```

For a model, resolve it from the plugin then pass to `fetchHandler`:

```ts
const gai = googleAI();
const model = await gai.model('gemini-2.0-flash');
app.post('/models/gemini-flash', (c) => fetchHandler(model)(c.req.raw));
```

### Multiple actions with `fetchHandlers`

Mount several actions (flows, models, etc.) under one path; the action is selected by the request path (e.g. `/api/hello` runs the action named `hello`):

```ts
import { fetchHandlers } from '@genkit-ai/fetch';

const actions = [helloFlow, greetingFlow, streamingFlow];

app.all('/api/*', (c) => fetchHandlers(actions, '/api')(c.req.raw));
```

Clients call `POST /api/<actionName>` with body `{ "data": <input> }`.

### Auth with context providers

Use a context provider (e.g. for auth) and attach it to an action with `withActionOptions`:

```ts
import { UserFacingError } from 'genkit';
import type { ContextProvider, RequestData } from 'genkit/context';
import { fetchHandler, fetchHandlers, withActionOptions } from '@genkit-ai/fetch';

const authContext: ContextProvider<{ userId: string }> = (req: RequestData) => {
  if (req.headers['authorization'] !== 'Bearer open-sesame') {
    throw new UserFacingError('PERMISSION_DENIED', 'not authorized');
  }
  return { userId: 'authenticated-user' };
};

// Single action with auth
app.all('/secureFlow', (c) =>
  fetchHandler(secureFlow, { contextProvider: authContext })(c.req.raw)
);

// Or wrap the action for use with fetchHandlers
const actions = [
  publicFlow,
  withActionOptions(secureFlow, { contextProvider: authContext }),
];
app.all('/api/*', (c) => fetchHandlers(actions, '/api')(c.req.raw));
```

### Durable streaming (Beta)

You can configure actions to use a `StreamManager` so stream state is persisted. Clients can disconnect and reconnect without losing the stream.

Provide a `streamManager` in the options. For development, use `InMemoryStreamManager`:

```ts
import { InMemoryStreamManager } from 'genkit/beta';
import { fetchHandler, fetchHandlers, withActionOptions } from '@genkit-ai/fetch';

app.all('/myDurableFlow', (c) =>
  fetchHandler(myFlow, {
    streamManager: new InMemoryStreamManager(),
  })(c.req.raw)
);

// Or with fetchHandlers
const actions = [
  withActionOptions(myFlow, {
    streamManager: new InMemoryStreamManager(),
  }),
];
app.all('/api/*', (c) => fetchHandlers(actions, '/api')(c.req.raw));
```

For production, use a durable implementation such as `FirestoreStreamManager` or `RtdbStreamManager` from `@genkit-ai/firebase`, or a custom `StreamManager`.

Clients can reconnect using the `streamId`:

```ts
import { streamFlow } from 'genkit/beta/client';

// Start a new stream
const result = streamFlow({
  url: 'http://localhost:3780/api/myDurableFlow',
  input: 'tell me a long story',
});
const streamId = await result.streamId; // save for reconnect

// Reconnect later
const reconnected = streamFlow({
  url: 'http://localhost:3780/api/myDurableFlow',
  streamId,
});
```

### Calling actions from the client

Use `runFlow` and `streamFlow` from `genkit/beta/client` (same protocol as the Express plugin):

```ts
import { runFlow, streamFlow } from 'genkit/beta/client';

const result = await runFlow({
  url: 'http://localhost:3780/api/hello',
  input: 'world',
});
console.log(result);

// With auth headers
const result = await runFlow({
  url: 'http://localhost:3780/api/secureGreeting',
  headers: { Authorization: 'Bearer open-sesame' },
  input: { name: 'Alex' },
});

// Streaming
const result = streamFlow({
  url: 'http://localhost:3780/api/streaming',
  input: { prompt: 'Say hello in chunks' },
});
for await (const chunk of result.stream) {
  console.log(chunk);
}
console.log(await result.output);
```

## API summary

| Export              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `fetchHandler(action, options?)` | Returns a handler `(request) => Promise<Response>` for a single action (flow, model, etc.). |
| `fetchHandlers(actions, pathPrefix?)` | Returns a handler that dispatches by path to one of the given actions.   |
| `withActionOptions(action, options)`  | Wraps an action with `contextProvider`, `streamManager`, or custom `path`. |
| `ActionWithOptions` | Type for an action plus options.                                            |
| `FetchHandlerOptions` | Options for `fetchHandler`: `contextProvider`, `streamManager`.            |

Request body must be JSON with a `data` field: `{ "data": <input> }`. For streaming, use `Accept: text/event-stream` or query `?stream=true`.

## Contributing

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests there.

More details are in the [Genkit documentation](https://genkit.dev/docs/get-started/).

## License

Licensed under the Apache 2.0 License.
