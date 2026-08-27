# @genkit-ai/a2ui

A Genkit plugin that brings [A2UI](https://a2ui.org/) ("Agent to UI"), a
transport-agnostic, JSON-based streaming UI protocol, to Genkit agents.

An A2UI-enabled agent can stream more than prose. It streams rich, interactive UI
**surfaces** (cards, lists, forms, buttons) that a client renders incrementally
as the model responds. The whole server-side integration is a single model
middleware: add `a2ui()` to an agent's `use` array and nothing else changes.

> Status: experimental.

## Installation

Install the plugin in your project:

```bash
npm i @genkit-ai/a2ui
```

To render surfaces in the browser you will also want a renderer and its
supporting packages. A2UI ships renderers for several frameworks:
[`@a2ui/lit`](https://www.npmjs.com/package/@a2ui/lit),
[`@a2ui/react`](https://www.npmjs.com/package/@a2ui/react), and
[`@a2ui/angular`](https://www.npmjs.com/package/@a2ui/angular). The examples
below use the Lit renderer:

```bash
npm i @a2ui/lit @a2ui/web_core @a2ui/markdown-it
```

You will also load the **Material Symbols Outlined** font on the client (the
basic catalog's `Icon` component renders names as font ligatures). See the
[renderer note](#renderer-requirements) below.


## Quickstart

### 1. Add the middleware on the server

Add `a2ui()` to your agent's `use` array. That is the entire server-side setup.

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

It works the same on a one-shot `generate`:

```ts
const res = await ai.generate({
  prompt: 'Show me the weather in Tokyo',
  use: [a2ui()],
});
```

### 2. Render surfaces on the client

`@genkit-ai/a2ui/client` is browser-safe (no Node dependencies). Consume the
agent with `remoteAgent` from `genkit/beta/client`, pull A2UI envelopes off each
chunk with `a2uiEnvelopesFromParts`, and feed them to a renderer. The example
below uses [`@a2ui/lit`](https://www.npmjs.com/package/@a2ui/lit), but the same
approach works with the [`@a2ui/react`](https://www.npmjs.com/package/@a2ui/react)
and [`@a2ui/angular`](https://www.npmjs.com/package/@a2ui/angular) renderers:

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

If you are not using the full agent client, `@genkit-ai/a2ui/client` also ships a
lightweight `streamA2uiAgent({ url, message, sessionId })` async-generator helper
that yields `{ type: 'text' }` and `{ type: 'envelopes' }` events.

> See [`js/testapps/a2ui`](../../testapps/a2ui) for a complete, runnable sample
> (Express backend plus a Vite + Lit frontend).

## Options

Pass options to `a2ui()` to control the catalog, prompt injection, and
validation:

| Option         | Default    | Description                                                                                                                     |
| -------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `catalog`      | `'basic'`  | The id of the catalog describing what the agent may render.                                                                    |
| `instructions` | `'system'` | Where to inject catalog capabilities. `'none'` injects nothing.                                                                |
| `validate`     | `'warn'`   | Validate emitted envelopes against the catalog. `'warn'` logs and drops bad blocks; `'strict'` throws; `'off'` skips checking. |
| `surfaceId`    | fresh UUID | Surface id policy. Defaults to a new UUID per surface; pass a fixed string to reuse one id for every surface.                  |
| `version`      | `'v0.9'`   | Protocol version stamped on envelopes.                                                                                         |

## Handling user actions

When a user interacts with a surface (for example, presses a `Button`), the
renderer emits an `A2uiClientAction`. Turn it into an agent input with
`actionToMessage` and send it as the next turn:

```ts
import { actionToMessage, type A2uiClientAction } from '@genkit-ai/a2ui/client';

const processor = new MessageProcessor([basicCatalog], (action) => {
  const turn = chat.sendStream({ message: actionToMessage(action) });
  // …consume turn.stream like above…
});
```

The action's `name` is sent as the user message; the full action (including its
`context`) is attached as an a2ui data part so the agent can react to it.

### Forms

Input components (`TextField`, `CheckBox`, `Slider`) do **not** send their values
automatically. To capture what the user entered, the model must:

1. Bind each input's `value` to a data-model path (`{ "path": "/email" }`).
2. Echo those same paths in the submit `Button`'s `action.event.context`.

The catalog capabilities injected into the system prompt already instruct the
model to do this. Without both steps, the action arrives with an empty `context`.

### Renderer requirements

The `@a2ui/lit` basic catalog needs two host-side pieces to render fully:

- A **MarkdownRenderer** provided via Lit context (for example, backed by
  `@a2ui/markdown-it`). `Text` heading variants are rendered as Markdown.
- The **Material Symbols Outlined** font. The `Icon` component renders names as
  font ligatures.

Without them, headings show as literal `##` and icons show as literal names. See
[`js/testapps/a2ui`](../../testapps/a2ui) for the wiring.

## Custom catalogs

The `catalog` option is a **catalog id** resolved from the Genkit registry. The
bundled `'basic'` catalog (mirroring `@a2ui/web_core`'s basic catalog) is the
default and needs no registration.

To match your own layout elements and design system, define a custom catalog,
register it with `loadCatalog`, and reference it by id.

### Catalog format

An A2UI catalog describes the components the model is allowed to emit:

- `id`: A globally unique URI identifying the catalog (used as `catalogId` on
  `createSurface`).
- `components`: An array of components, where each has:
  - `name`: The component type name, matching the renderer type (for example
    `CustomCard`, `Text`).
  - `description`: A clear, one-line summary of what the component is and when to
    use it.
  - `props`: A compact, model-facing text description of its properties (kept as
    a simple, human-readable string to minimize system prompt token usage).

### Option A: load from a JSON file

Create a JSON file (for example `./my-catalog.json`) following this format:

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

Then register it under a lookup identifier (for example `'my-catalog'`) on the
server:

```ts
import { loadCatalog } from '@genkit-ai/a2ui';

await loadCatalog(ai, { id: 'my-catalog', file: './my-catalog.json' });
```

### Option B: in-memory definition

You can construct and register an `A2uiCatalog` directly:

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

### Using a registered catalog

Once registered, reference the lookup id in your `a2ui()` options:

```ts
export const uiAgent = ai.defineAgent({
  name: 'uiAgent',
  model: 'googleai/gemini-flash-latest',
  use: [a2ui({ catalog: 'my-catalog' })],
});
```

Catalogs live in the registry (value type `a2ui-catalog`) so the middleware can
resolve them by id and, in the future, tooling can list them.

## Security and the trust boundary

Generative UI moves model output into the DOM, so treat every surface an agent
emits as **untrusted input**. The `a2ui()` middleware's `validate` option
(including `'strict'`) checks envelope structure and component *type names*
against the catalog only. It does **not** validate component props or data-model
values: model-controlled values such as `Image.url` and `Text` (inline Markdown,
which a renderer may turn into HTML) pass through untouched. `'strict'` is a
well-formedness check, not a security boundary.

A prompt-injected or simply mistaken model can therefore emit an arbitrary remote
image URL, or Markdown that a renderer turns into HTML. To keep that safe:

- **The renderer/catalog owns prop sanitization.** Whatever renders a surface
  (for example `@a2ui/lit` plus your Markdown renderer) is responsible for
  escaping and sanitizing prop values before they reach the DOM. If you ship a
  custom catalog, its renderer must sanitize its own components' props.
- **Restrict remote sources at the host.** Serve the app with a Content Security
  Policy that limits `img-src` (and other fetch directives) to origins you trust,
  so a model-supplied image or link URL cannot exfiltrate data or load
  unexpected content.
- **Do not put secrets in the data model.** Anything bound into a surface's data
  model can be echoed back through an action's `context`.

If you need server-side control over props (for example, allow-listing image
hosts), add your own model middleware after `a2ui()` to inspect and rewrite the
emitted a2ui parts.

## How it works

### One representation

A2UI rides on its own part channel: a Genkit `data` part carrying the mime type
`application/a2ui+json` whose `data` is an object `{ envelopes }` wrapping an
array of A2UI envelope messages. This maps 1:1 onto the A2A binding of the A2UI
spec, so an A2A or MCP binding can drop in later for free.

- A **mixed** turn is a message whose content is `[textPart, a2uiPart, …]`.
- A **pure-surface** turn is the special case with no text parts.
- Downstream consumers (client transport, `@a2ui/web_core`) only ever see a2ui
  parts. "Pure vs mixed" is a prompting choice, not a separate code path.

### The middleware pipeline

On each model call inside the agent's tool loop, `a2ui()`:

1. Injects the catalog's capabilities into the system prompt so the model knows
   what UI it may render (unless `instructions: 'none'`).
2. Intercepts the model's output, both the streamed chunks and the final
   aggregated message.
3. Extracts `a2ui` fenced code blocks from the model's text.
4. Validates them against the catalog (per the `validate` option).
5. Rewrites them into canonical a2ui data parts.

Inbound a2ui parts (for example, a surface action sent back as the next turn, or
replayed history) are summarized into plain text before the underlying model sees
them, so a model that does not understand the a2ui mime type can still reason
about prior surfaces and user actions.

## License

Apache-2.0
