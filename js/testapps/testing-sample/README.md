# testing-sample

Demonstrates `genkit/testing` — unit-testing Genkit flows, prompts, tools, and
streaming deterministically, without a live model, network, or API key.

`src/menu.ts` defines a normal app: a `recommendDish` flow that calls
`ai.generate` with a `dailySpecial` tool. The default model is referenced by
name only (`'menuModel'`), so tests register a mock under that name and the flow
code runs unchanged.

`tests/menu_test.ts` shows the patterns:

- **`mockModel`** — script the model response, then assert flow output and
  inspect what the model was called with: `model.lastMessage` (a genkit
  `Message`, so `.text` / `.media` work like on a response), plus
  `model.lastRequest`, `model.requests`, `model.requestCount`.
- **Tool calls** — return `toolRequests` from the mock; the framework dispatches
  the real tool and calls the model again with the result.
- **Streaming** — emit chunks via `sendChunk` and assert on the stream.
- **`echoModel`** — zero-config model that echoes the rendered request, for
  asserting prompt/message assembly.

`genkit/testing` is runner-agnostic. The same patterns are shown twice:
`tests/menu_test.ts` under Node's built-in `node:test`, and
`tests/menu.vitest.test.ts` under vitest.

## Run

```sh
pnpm i
pnpm test          # node:test
pnpm test:vitest   # vitest
```
