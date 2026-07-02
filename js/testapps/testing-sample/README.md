# testing-sample

Demonstrates `genkit/testing` — unit-testing Genkit flows, prompts, tools, and
streaming deterministically, without a live model, network, or API key.

`src/menu.ts` defines a normal app with logic worth testing:

- a `recommendPrompt` dotprompt (Handlebars template + system instruction + the
  `dailySpecial` tool);
- a `recommendDish` flow that asks for a **structured** recommendation, then
  applies its own logic — validating the result and deriving `withinBudget`;
- a `streamRecommendation` flow that streams the model's tokens out through the
  flow's own stream;
- a `confirmBooking` tool that `interrupt()`s for human confirmation
  (human-in-the-loop). This is why the app imports `genkit/beta` — interrupts
  are a beta feature; everything else works identically on the stable `genkit`.

The app is a standard module-level Genkit singleton. Its default model is
referenced by name only (`'menuModel'`), so tests register a mock under that
name and the app code runs unchanged. Each test *file* gets its own process or
module graph (and so a fresh registry) from the runner — `node --test` spawns
a child process per file; Jest and Vitest isolate each file's module registry —
so a file registers its mock **once** and tests share it:
`reset()` in `beforeEach` clears recorded history, and each test scripts its
own behavior with `respondWith(...)`.

`tests/menu_test.ts` shows the patterns:

- **Structured output + your own logic** — the mock returns JSON; the test pins
  down the flow's derivation (`withinBudget`) and validation. The same model
  output with a lower budget flips `withinBudget` — proving the test exercises
  *your* code, not the model.
- **Error paths** — script an invalid response and assert the flow rejects it.
- **Tool round-trip** — return `toolRequests` from the mock; the framework
  dispatches the real tool and calls the model again with the result. Assert
  what ran via `model.toolResponses` (the tool results fed back to the model).
- **Scripting with a queue** — pass `respondWith([...])` an array consumed one
  item per call (last repeating), so a multi-turn tool interaction needs no
  branching callback.
- **Injected failures** — a queued `Error` (`respondWith([new Error(...)])`) is
  thrown when reached, to test how a flow handles a model failure.
- **Human-in-the-loop** — the model requests `confirmBooking`, the tool
  interrupts, generation pauses (`response.interrupts`), then `restart` resumes
  it with the human's decision so the tool books the dish.
- **Streaming through a flow** — drive `flow.stream(...)`, emit chunks via
  `sendChunk`, and assert chunks arrive through the flow plus the final output.
- **`echoModel`** (`tests/prompt_test.ts`) — a zero-config model that echoes
  the whole assembled request (system + Handlebars-rendered messages), for
  asserting prompt assembly. It lives in its own test file because
  `menu_test.ts` already claims `'menuModel'` with a `mockModel`, and per-file
  process isolation keeps the registrations apart. It's a *text*-path preset:
  if the request carries an output schema it throws an explanatory error (text
  can't satisfy a schema) — use `mockModel` there.
- **Inspection** — `model.lastRequest` / `lastRequestMessage` (a genkit
  `Message`, so `.text` / `.media` work like on a response) / `lastRequestText`
  (the full assembled conversation as a string — works on *any* mock, including
  structured-output paths where `echoModel` can't be used) / `toolResponses` /
  `requests` / `requestCount`.

`genkit/testing` is runner-agnostic. The same patterns are shown twice: under
Node's built-in `node:test` (`tests/menu_test.ts`, `tests/prompt_test.ts`) and
under vitest (`tests/menu.vitest.test.ts`, `tests/prompt.vitest.test.ts`).

## Run

```sh
pnpm i
pnpm test          # node:test
pnpm test:vitest   # vitest
```
