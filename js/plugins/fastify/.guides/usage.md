Genkit's Fastify integration makes it easy to expose Genkit flows as Fastify API endpoints:

```ts
import Fastify from 'fastify';
import { fastifyHandler } from '@genkit-ai/fastify';
import { simpleFlow } from './flows/simple-flow.js';

const app = Fastify();

app.post('/simpleFlow', fastifyHandler(simpleFlow));

await app.listen({ port: 8080 });
```

You can also handle auth using context providers:

```ts
import { UserFacingError } from 'genkit';
import { ContextProvider, RequestData } from 'genkit/context';

const contextProvider: ContextProvider<Context> = (req: RequestData) => {
  if (req.headers['authorization'] !== 'open sesame') {
    throw new UserFacingError('PERMISSION_DENIED', 'not authorized');
  }
  return {
    auth: {
      user: 'Ali Baba',
    },
  };
};

app.post('/simpleFlow', fastifyHandler(simpleFlow, { contextProvider }));
```

Flows and actions exposed using the `fastifyHandler` function can be accessed using the `genkit/beta/client` library:

```ts
import { runFlow, streamFlow } from 'genkit/beta/client';

const result = await runFlow({
  url: `http://localhost:${port}/simpleFlow`,
  input: 'say hello',
});

console.log(result); // hello
```

```ts
// set auth headers (when using auth policies)
const result = await runFlow({
  url: `http://localhost:${port}/simpleFlow`,
  headers: {
    Authorization: 'open sesame',
  },
  input: 'say hello',
});

console.log(result); // hello
```

```ts
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

You can register the `genkitFastify` plugin to quickly expose multiple flows and actions:

```ts
import Fastify from 'fastify';
import { genkitFastify } from '@genkit-ai/fastify';
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

const app = Fastify();
await app.register(genkitFastify, {
  flows: [menuSuggestionFlow],
});
await app.listen({ port: 8080 });
```

You can prefix the exposed routes and attach per-flow options:

```ts
import { genkitFastify, withFlowOptions } from '@genkit-ai/fastify';

await app.register(genkitFastify, {
  pathPrefix: '/api',
  flows: [withFlowOptions(menuSuggestionFlow, { contextProvider })],
});
```
