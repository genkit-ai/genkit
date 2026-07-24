# @genkit-ai/a2ui

A Genkit plugin that adds [A2UI](https://a2ui.org/) ("Agent to UI") — a
transport-agnostic, JSON-based **streaming UI protocol** — to Genkit agents.

An A2UI-enabled agent can stream not just prose, but rich, interactive UI
**surfaces** that a client renders incrementally.

> Status: experimental.

## Design principle: one representation

A2UI rides on its own part channel — a Genkit `data` part carrying the mime type
`application/a2ui+json` whose `data` is an object `{ envelopes }` wrapping an
**array of A2UI envelope messages**. This maps 1:1 onto the A2A binding of the
A2UI spec, so an A2A/MCP binding can drop in later for free.

- A **mixed** turn is a message whose content is `[textPart, a2uiPart, …]`.
- A **pure-surface** turn is the special case with no text parts.
- Downstream consumers (client transport, `@a2ui/web_core`) only ever see a2ui
  parts.

## Server: the `a2ui()` middleware

The whole server-side integration is the `a2ui()` model middleware. Add it to an
agent's (or a one-shot `generate`'s) `use` array. Nothing else changes.

```ts
import { genkit } from 'genkit/beta';
import { googleAI } from '@genkit-ai/google-genai';
import { a2ui } from '@genkit-ai/a2ui';

const ai = genkit({ plugins: [googleAI()] });

export const uiAgent = ai.defineAgent({
  name: 'uiAgent',
  model: 'googleai/gemini-flash-latest',
  system: 'You help users. Render UI when it is clearer than prose.',
  use: [a2ui()], // <- A2UI support (defaults to the bundled 'basic' catalog)
});
```

Works the same on a one-shot generate:

```ts
const res = await ai.generate({
  prompt: 'Show me the weather in Tokyo',
  use: [a2ui()],
});
```

### Options

| Option         | Default    | Description                                                                                                                     |
| -------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `catalog`      | `'basic'`  | The id of the catalog describing what the agent may render.                                                                    |
| `instructions` | `'system'` | Where to inject catalog capabilities. `'none'` injects nothing.                                                                |
| `validate`     | `'warn'`   | Validate emitted envelopes against the catalog. `'warn'` logs and drops bad blocks; `'strict'` throws; `'off'` skips checking. |
| `surfaceId`    | fresh UUID | Surface id policy. Defaults to a new UUID per surface; pass a fixed string to reuse one id for every surface.                  |
| `version`      | `'v0.9'`   | Protocol version stamped on envelopes.                                                                                         |

The middleware injects the catalog's capabilities into the system prompt, then
intercepts model output (streamed chunks **and** the final message), extracts
`a2ui` fenced blocks, validates them, and rewrites them into a2ui data parts.

### Catalogs

`catalog` is a **catalog id** resolved from the Genkit registry. The bundled
`'basic'` catalog (mirroring `@a2ui/web_core`'s basic catalog) is the default and
needs no registration.

To define and use a custom catalog (e.g. matching your own layout elements and
design system), register it with `loadCatalog` and reference it by id.

#### Catalog format & structure

An A2UI catalog describes the list of visual or interactive components the model
is allowed to emit. It consists of:

- `id`: A globally unique URI identifying the catalog (used as `catalogId` on
  `createSurface`).
- `components`: An array of components, where each component contains:
  - `name`: The component type name, matching the renderer type (e.g.
    `CustomCard`, `Text`).
  - `description`: A clear, one-line summary of what the component is and when to
    use it.
  - `props`: A compact, model-facing text description of its properties (kept as
    a simple, human-readable string to minimize system prompt token usage).

#### Option A: load from a JSON file

Create a JSON file (e.g. `./my-catalog.json`) following this format:

```json
{
  "id": "https://my-app.org/catalogs/custom.json",
  "components": [
    {
      "name": "Banner",
      "description": "Displays a prominent alert banner at the top of a section.",
      "props": "title: string (required); severity?: info|warning|error."
    },
    {
      "name": "Text",
      "description": "Displays a plain or inline-markdown text run.",
      "props": "text: string (required); variant?: body|caption."
    }
  ]
}
```

Then register it under a lookup identifier (e.g. `'my-catalog'`) on the server:

```ts
import { loadCatalog } from '@genkit-ai/a2ui';

await loadCatalog(ai, { id: 'my-catalog', file: './my-catalog.json' });
```

#### Option B: in-memory definition

You can construct and register an `A2uiCatalog` directly in-memory:

```ts
import { loadCatalog, type A2uiCatalog } from '@genkit-ai/a2ui';

const myCatalog: A2uiCatalog = {
  id: 'https://my-app.org/catalogs/custom.json',
  components: [
    {
      name: 'Banner',
      description: 'Displays a prominent alert banner at the top of a section.',
      props: 'title: string (required); severity?: info|warning|error.',
    },
    {
      name: 'Text',
      description: 'Displays a plain or inline-markdown text run.',
      props: 'text: string (required); variant?: body|caption.',
    },
  ],
};

await loadCatalog(ai, { id: 'my-catalog', catalog: myCatalog });
```

#### Using the registered catalog in agents

Once registered, reference your catalog lookup id in your `a2ui()` options:

```ts
export const uiAgent = ai.defineAgent({
  name: 'uiAgent',
  model: 'googleai/gemini-flash-latest',
  use: [a2ui({ catalog: 'my-catalog' })],
});
```

Catalogs live in the registry (value type `a2ui-catalog`) so the middleware can
resolve them by id and, in the future, tooling can list them.

## Client

`@genkit-ai/a2ui/client` is browser-safe (no Node deps). Consume the agent with
`remoteAgent` from `genkit/beta/client` and pull A2UI envelopes off each chunk
with `a2uiEnvelopesFromParts`, feeding them to a renderer such as
[`@a2ui/lit`](https://www.npmjs.com/package/@a2ui/lit):

```ts
import { MessageProcessor } from '@a2ui/web_core/v0_9';
import { basicCatalog } from '@a2ui/lit/v0_9';
import '@a2ui/lit/v0_9'; // registers <a2ui-surface> + basic components
import { a2uiEnvelopesFromParts } from '@genkit-ai/a2ui/client';
import { remoteAgent } from 'genkit/beta/client';

const processor = new MessageProcessor([basicCatalog]);
processor.onSurfaceCreated((s) => {
  document.querySelector('a2ui-surface').surface = s;
});

const chat = remoteAgent({ url: '/api/uiAgent' }).chat();
const turn = chat.sendStream('weather in Tokyo');
for await (const chunk of turn.stream) {
  if (chunk.text) appendProse(chunk.text);
  const envelopes = a2uiEnvelopesFromParts(chunk.raw.modelChunk?.content);
  if (envelopes.length) processor.processMessages(envelopes);
}
```

If you're not using the full agent client, `@genkit-ai/a2ui/client` also ships a
lightweight `streamA2uiAgent({ url, message, sessionId })` async-generator helper
that yields `{ type: 'text' }` / `{ type: 'envelopes' }` events.

### Sending user actions back to the agent

When the user interacts with a surface (e.g. presses a `Button`), the renderer
emits an `A2uiClientAction`. Turn it into an agent input with `actionToMessage`
and send it as the next turn:

```ts
import { actionToMessage, type A2uiClientAction } from '@genkit-ai/a2ui/client';

const processor = new MessageProcessor([basicCatalog], (action) => {
  const turn = chat.sendStream({ message: actionToMessage(action) });
  // …consume turn.stream like above…
});
```

The action's `name` is sent as the user message; the full action (including its
`context`) is attached as an a2ui data part so the agent can react to it.

**Forms:** input components (`TextField`, `CheckBox`, `Slider`) do **not** send
their values automatically. To capture what the user entered, the model must
(1) bind each input's `value` to a data-model path (`{ "path": "/email" }`) and
(2) echo those same paths in the submit `Button`'s `action.event.context`. The
catalog capabilities injected into the system prompt already instruct the model
to do this; without both, the action arrives with an empty `context`.

> Renderer note: the `@a2ui/lit` basic catalog needs two host-side pieces to
> render fully — a **MarkdownRenderer** provided via Lit context (e.g. backed by
> `@a2ui/markdown-it`; `Text` heading variants are rendered as Markdown), and the
> **Material Symbols Outlined** font (the `Icon` component renders names as font
> ligatures). Without them, headings show as literal `##` and icons as literal
> names. See `js/testapps/a2ui` for the wiring.

See `js/testapps/a2ui` for a complete runnable sample.

## License

Apache-2.0
