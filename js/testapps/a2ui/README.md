# A2UI testapp

A runnable demo of [`@genkit-ai/a2ui`](../../plugins/a2ui): a Genkit agent that
streams generative **A2UI** surfaces, rendered live in the browser with the
[`@a2ui/lit`](https://www.npmjs.com/package/@a2ui/lit) renderer.

- **Backend** (`src/index.ts`) — an `ai.defineAgent({ use: [a2ui()], store })`
  agent with a `getWeather` tool, served over HTTP with `expressHandler` at
  `/api/uiAgent`.
- **Frontend** (`web/`) — a tiny Vite + Lit chat UI that talks to the agent with
  `remoteAgent` from `genkit/beta/client` and renders each surface via
  `@a2ui/web_core`'s `MessageProcessor`.

Two host-side pieces are required for the basic catalog to render fully (and are
wired up in this sample):

- A **MarkdownRenderer** provided via Lit context — the basic catalog's `Text`
  component converts heading `variant`s to Markdown and renders them through a
  renderer pulled from context. This app provides one backed by
  `@a2ui/markdown-it`; without it, headings show as literal `##`.
- The **Material Symbols Outlined** font — `Icon` renders names as font
  ligatures. `web/index.html` loads it from Google Fonts; without it, icon names
  show as literal text.

## Run it

You need a Gemini API key in `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

From `js/testapps/a2ui`:

```bash
# terminal 1 — backend on :8080
. ~/init-gemini.sh   # or: export GEMINI_API_KEY=...
pnpm server

# terminal 2 — Vite dev server on :5173 (proxies /api -> :8080)
pnpm web:dev
```

Open http://localhost:5173 and try the suggested prompts (weather, comparisons,
a signup form). The agent replies with brief prose and a rendered UI surface.

### One-shot production-style run

```bash
pnpm start   # builds the web frontend, then serves everything from :8080
```

Then open http://localhost:8080.

### With the Genkit Dev UI

```bash
pnpm genkit:dev   # genkit start -- tsx src/index.ts
```

## How it works

The whole A2UI integration is the `a2ui()` middleware in the agent's `use`
array. It injects the catalog's capabilities into the system prompt, then
rewrites the model's `a2ui` fenced blocks (in both the stream and the final
message) into canonical A2UI data parts. The browser client (`remoteAgent`)
streams the agent, filters those parts out with `a2uiEnvelopes`, and feeds whole
envelopes to the Lit renderer. Button presses on a surface are sent back to the
agent as the next turn; server-managed sessions (the agent's
`InMemorySessionStore`) keep the conversation history.
