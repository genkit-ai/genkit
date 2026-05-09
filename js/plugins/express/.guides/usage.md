Genkit's Express integration makes it easy to expose Genkit flows as Express API endpoints:

```ts
import express from 'express';
import { expressHandler } from '@genkit-ai/express';
import { simpleFlow } from './flows/simple-flow.js';

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

Flows and actions exposed using the `expressHandler` function can be accessed using `genkit/beta/client` library:

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

You can use `startFlowServer` to quickly expose multiple flows and actions:

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
