# Genkit Express Plugin

This plugin provides utilities for conveninetly exposing Genkit flows and actions via Express HTTP server as REST APIs.

See [official documentation](https://genkit.dev/docs/frameworks/express/) for more.

## Installation

To use this plugin, install it in your project:

```bash
npm i @genkit-ai/express
```

## Usage

```ts
import { expressHandler } from '@genkit-ai/express';
import express from 'express';

const simpleFlow = ai.defineFlow('simpleFlow', async (input, { sendChunk }) => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: input,
    onChunk: (c) => sendChunk(c.text),
  });
  return text;
});

const app = express();
app.use(express.json());

app.post('/simpleFlow', expressHandler(simpleFlow));

app.listen(8080);
```

You can also handle auth using context providers:

```ts
import { UserFacingError } from 'genkit';
import { ContextProvider, RequestData } from 'genkit/context';

const context: ContextProvider<Context> = (req: RequestData) => {
  if (req.headers['authorization'] !== 'open sesame') {
    throw new UserFacingError('PERMISSION_DENIED', 'not authorized');
  }
  return {
    auth: {
      user: 'Ali Baba',
    },
  };
};

app.post(
  '/simpleFlow',
  authMiddleware,
  expressHandler(simpleFlow, { context })
);
```

### Durable Streaming (Beta)

You can configure flows to use a `StreamManager` to persist their state. This allows clients to disconnect and reconnect to a stream without losing its state.

To enable durable streaming, provide a `streamManager` in the `expressHandler` options. The `InMemoryStreamManager` is useful for development and testing:

```ts
import { InMemoryStreamManager } from 'genkit/beta';

// ...

app.post(
  '/myDurableFlow',
  expressHandler(myFlow, {
    streamManager: new InMemoryStreamManager(),
  })
);
```

For production environments, you should use a durable `StreamManager` implementation, such as `FirestoreStreamManager` or `RtdbStreamManager` from the `@genkit-ai/firebase` plugin, or a custom implementation.

Clients can then connect and reconnect to the stream using the `streamId`:

```ts
// Start a new stream
const result = streamFlow({
  url: `http://localhost:8080/myDurableFlow`,
  input: 'tell me a long story',
});
const streamId = await result.streamId; // Save this ID

// ... later, reconnect if needed ...
const reconnectedResult = streamFlow({
  url: `http://localhost:8080/myDurableFlow`,
  streamId: streamId,
});
```

Flows and actions exposed using the `expressHandler` function can be accessed using `genkit/beta/client` library:

```ts
import { runFlow, streamFlow } from 'genkit/beta/client';

const result = await runFlow({
  url: `http://localhost:${port}/simpleFlow`,
  input: 'say hello',
});

console.log(result); // hello

// set auth headers (when using auth policies)
const result = await runFlow({
  url: `http://localhost:${port}/simpleFlow`,
  headers: {
    Authorization: 'open sesame',
  },
  input: 'say hello',
});

console.log(result); // hello

// and streamed
const result = streamFlow({
  url: `http://localhost:${port}/simpleFlow`,
  input: 'say hello',
});
for await (const chunk of result.stream) {
  console.log(chunk);
}
console.log(await result.output);
```

### Initialization Data

If your flow or action accepts initialization data (defined via `initSchema`), you can pass it using the `init` option in the client:

```ts
const result = await runFlow({
  url: `http://localhost:${port}/myFlow`,
  input: 'say hello',
  init: { sessionId: 'abc123', config: { temperature: 0.7 } },
});

// Also works with streaming
const streamed = streamFlow({
  url: `http://localhost:${port}/myFlow`,
  input: 'say hello',
  init: { sessionId: 'abc123' },
});
for await (const chunk of streamed.stream) {
  console.log(chunk);
}
console.log(await streamed.output);
```

The `init` data is sent in the request body alongside `data` and is validated against the action's `initSchema` on the server side.

You can also use `startFlowServer` to quickly expose multiple flows and actions:

```ts
import { startFlowServer } from '@genkit-ai/express';
import { genkit } from 'genkit';

const ai = genkit({});

export const menuSuggestionFlow = ai.defineFlow(
  {
    name: 'menuSuggestionFlow',
  },
  async (restaurantTheme) => {
    // ...
  }
);

startFlowServer({
  flows: [menuSuggestionFlow],
});
```

You can also configure the server:

```ts
startFlowServer({
  flows: [menuSuggestionFlow],
  port: 4567,
  cors: {
    origin: '*',
  },
});
```

## Contributing

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [official Genkit documentation](https://genkit.dev/docs/get-started/).

## License

Licensed under the Apache 2.0 License.
