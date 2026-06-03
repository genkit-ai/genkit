# testing-sample

Demonstrates `genkit/testing` — unit-testing Genkit flows, prompts, tools, and
streaming deterministically, without a live model, network, or API key.

`src/menu.ts` defines a normal app with logic worth testing:

- a `recommendPrompt` dotprompt (Handlebars template + system instruction + the
  `dailySpecial` tool);
- a `recommendDish` flow that asks for a **structured** recommendation, then
  applies its own logic — validating the result and deriving `withinBudget`;
- a `streamRecommendation` flow that streams the model's tokens out through the
  flow's own stream.

`createMenuApp()` is a factory so each test builds a fresh, isolated app. The
default model is referenced by name only (`'menuModel'`), so tests register a
mock under that name and the app code runs unchanged.

`tests/menu_test.ts` shows the patterns:

- **Structured output + your own logic** — the mock returns JSON; the test pins
  down the flow's derivation (`withinBudget`) and validation. The same model
  output with a lower budget flips `withinBudget` — proving the test exercises
  *your* code, not the model.
- **Error paths** — script an invalid response and assert the flow rejects it.
- **Tool round-trip** — return `toolRequests` from the mock; the framework
  dispatches the real tool and calls the model again with the result.
- **Streaming through a flow** — drive `flow.stream(...)`, emit chunks via
  `sendChunk`, and assert chunks arrive through the flow plus the final output.
- **`echoModel`** — a zero-config model that echoes the whole assembled request
  (system + Handlebars-rendered messages), for asserting prompt assembly. Use it
  for text paths; `mockModel` for structured output.
- **Inspection** — `model.lastRequest` / `lastRequestMessage` (a genkit
  `Message`, so `.text` / `.media` work like on a response) / `requests` /
  `requestCount`.

`genkit/testing` is runner-agnostic. The same patterns are shown twice:
`tests/menu_test.ts` under Node's built-in `node:test`, and
`tests/menu.vitest.test.ts` under vitest.

## Run

```sh
pnpm i
pnpm test          # node:test
pnpm test:vitest   # vitest
```
