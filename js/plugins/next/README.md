# Genkit Next.js Plugin

See [official documentation](https://genkit.dev/docs/frameworks/nextjs/) for more.

This plugin provides utilities for conveninetly exposing Genkit flows and actions via Next.js app routs for REST APIs.

```ts
// /genkit/simpleFlow.ts
const simpleFlow = ai.defineFlow(
  'simpleFlow',
  async (input, streamingCallback) => {
    const { text } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: input,
      streamingCallback: (chunk) => streamingCallback(chunk.text),
    });
    return text;
  }
);
```

```ts
// /app/api/simpleFlow/route.ts
import { simpleFlow } from '@/genkit/simpleFlow';
import { appRoute } from '@genkit-ai/next';

export const POST = appRoute(simpleFlow);
```

### Durable Streaming (Beta)

You can configure flows to use a `StreamManager` to persist their state. This allows clients to disconnect and reconnect to a stream without losing its state.

To enable durable streaming, provide a `streamManager` in the `appRoute` options. The `InMemoryStreamManager` is useful for development and testing:

```ts
// /app/api/myDurableFlow/route.ts
import { myFlow } from '@/genkit/myFlow';
import { appRoute } from '@genkit-ai/next';
import { InMemoryStreamManager } from 'genkit/beta';

export const POST = appRoute(myFlow, {
  streamManager: new InMemoryStreamManager(),
});
```

For production environments, you should use a durable `StreamManager` implementation, such as `FirestoreStreamManager` or `RtdbStreamManager` from the `@genkit-ai/firebase` plugin, or a custom implementation.

Clients can then connect and reconnect to the stream using the `streamId`:

```ts
// Start a new stream
const result = streamFlow({
  url: `/api/myDurableFlow`,
  input: 'tell me a long story',
});
const streamId = await result.streamId; // Save this ID

// ... later, reconnect if needed ...
const reconnectedResult = streamFlow({
  url: `/api/myDurableFlow`,
  streamId: streamId,
});
```

APIs can be called with the generic `genkit/beta/client` library, or `@genkit-ai/next/client`

```ts
import { runFlow, streamFlow } from '@genkit-ai/next/client';
import { simpleFlow } from '@/genkit/simpleFlow';

const result = await runFlow<typeof simpleFlow>({
  url: '/api/simpleFlow',
  input: 'say hello',
});

console.log(result); // hello

// set auth headers (when using auth policies)
const result = await runFlow<typeof simpleFlow>({
  url: `/api/simpleFlow`,
  headers: {
    Authorization: 'open sesame',
  },
  input: 'say hello',
});

console.log(result); // hello

// and streamed
const { stream, output } = streamFlow<typeof simpleFlow>({
  url: '/api/simpleFlow',
  input: 'say hello',
});
for await (const chunk of stream) {
  console.log(chunk.output);
}
console.log(await output); // output is a promise, must be awaited
```

### Initialization Data

If your flow or action accepts initialization data (defined via `initSchema`), you can pass it using the `init` option in the client:

```ts
const result = await runFlow<typeof myFlow>({
  url: '/api/myFlow',
  input: 'say hello',
  init: { sessionId: 'abc123', config: { temperature: 0.7 } },
});

// Also works with streaming
const { stream, output } = streamFlow<typeof myFlow>({
  url: '/api/myFlow',
  input: 'say hello',
  init: { sessionId: 'abc123' },
});
for await (const chunk of stream) {
  console.log(chunk.output);
}
console.log(await output);
```

The `init` data is sent in the request body alongside `data` and is validated against the action's `initSchema` on the server side.

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [official Genkit documentation](https://genkit.dev/docs/get-started/).

License: Apache 2.0
