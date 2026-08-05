# Genkit Fastify Plugin

This plugin provides utilities for conveniently exposing Genkit flows and actions via a [Fastify](https://fastify.dev/) HTTP server as REST APIs.

Fastify is not Web Fetch native and has no built-in Genkit integration, so wiring a flow into Fastify by hand requires bridging Fastify's `request`/`reply` to the Genkit action protocol (including Server-Sent Events for streaming). This plugin does that bridging for you, so a flow mounts in one line, just like [`@genkit-ai/express`](https://www.npmjs.com/package/@genkit-ai/express) does for Express.

See the [official documentation](https://genkit.dev/docs/frameworks/) for more.

## Installation

```bash
npm i @genkit-ai/fastify
```

## Usage

Mount a single flow with `fastifyHandler`:

```ts
import Fastify from 'fastify';
import { fastifyHandler } from '@genkit-ai/fastify';

const simpleFlow = ai.defineFlow('simpleFlow', async (input, { sendChunk }) => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: input,
    onChunk: (c) => sendChunk(c.text),
  });
  return text;
});

const app = Fastify();

app.post('/simpleFlow', fastifyHandler(simpleFlow));

await app.listen({ port: 8080 });
```

The handler reads `{ data }` from the JSON body, runs the flow, and streams chunks back as Server-Sent Events when the caller sends `Accept: text/event-stream` (or `?stream=true`) — exactly what the `runFlow` and `streamFlow` clients in `genkit/beta/client` expect.

### Expose multiple flows with the plugin

Register `genkitFastify` to mount several flows at once. Each flow is exposed at `/<flowName>` (override with `withFlowOptions`):

```ts
import Fastify from 'fastify';
import { genkitFastify, withFlowOptions } from '@genkit-ai/fastify';

const app = Fastify();

await app.register(genkitFastify, {
  pathPrefix: '/api',
  flows: [
    simpleFlow,
    withFlowOptions(secureFlow, { contextProvider }),
  ],
});

await app.listen({ port: 8080 });
```

### Handle auth with context providers

```ts
import { UserFacingError } from 'genkit';
import { ContextProvider, RequestData } from 'genkit/context';

const contextProvider: ContextProvider<Context> = (req: RequestData) => {
  if (req.headers['authorization'] !== 'open sesame') {
    throw new UserFacingError('PERMISSION_DENIED', 'not authorized');
  }
  return { auth: { user: 'Ali Baba' } };
};

app.post('/secureFlow', fastifyHandler(secureFlow, { contextProvider }));
```

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [Genkit documentation](https://genkit.dev/docs/get-started).

License: Apache 2.0
