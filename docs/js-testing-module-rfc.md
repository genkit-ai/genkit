# RFC: [JS] First-class testing module (`genkit/testing`)

> **Status:** Draft — open for discussion.
> **Service:** Managed DX. **Author:** Invertase. **Date:** 2026-06-02.

## Summary

Promote Genkit's internal model-mocking helpers into a public, documented `genkit/testing` module, so app developers can unit-test flows, prompts, tools, and chat deterministically — without a live model, network, or API key.

## Motivation

Testing a Genkit app means testing code that calls `ai.generate(...)`, which means substituting the model. Genkit's own suite does this with `defineProgrammableModel` and `defineEchoModel` (`js/ai/tests/helpers.ts`), used across dozens of internal tests — but neither is exported. The only public testing export, `testModels` (`genkit/testing`), is a plugin-conformance harness for model *plugin* authors, not a mocking tool for *app* authors.

So app developers hand-roll a fake plugin per project, and the streaming, tool-call, and structured-output cases — the parts that are easy to get wrong — have no blessed helper to copy. "How do I test my flow?" has no documented answer.

### We already pay this cost internally

The same fake-model pattern is re-implemented in at least five places in this monorepo, and the copies have drifted:

| File | Helper |
|---|---|
| `js/ai/tests/helpers.ts` | `defineProgrammableModel` / `defineEchoModel` (takes `Registry`) |
| `js/genkit/tests/helpers.ts` | `defineProgrammableModel` / `defineEchoModel` (takes `Genkit`) |
| `js/plugins/express/tests/express_test.ts` | inline fake |
| `js/plugins/mcp/tests/fakes.ts` | inline fake |
| `js/plugins/fetch/tests/web_test.ts` | inline fake |

The two `defineEchoModel`s don't even share a signature (one takes a `Registry`, the other a `Genkit`), and both expose call-inspection through an untyped escape hatch:

```ts
// js/ai/tests/helpers.ts:74 — inspection bolted on via `any`
(model as any).__test__lastRequest = request;
(model as any).__test__lastStreamingCallback = streamingCallback;
```

If the maintainers who wrote the model abstraction maintain five drifting copies and an `any`-cast inspection hack, an app developer starting from zero will not do better. That duplication is the signal the shape is real.

### What an app developer writes today

With nothing exported, the answer to "how do I test my flow" is: hand-roll a fake plugin and re-derive the streaming/inspection envelope.

```ts
// fake-model.ts — copied into every project
import { genkit, type GenerateRequest } from 'genkit';

export function makeFakeModel(ai: ReturnType<typeof genkit>) {
  let lastRequest: GenerateRequest | undefined;
  const model = ai.defineModel({ name: 'fake' }, async (request, sendChunk) => {
    lastRequest = request;                         // no typed way to expose this
    sendChunk?.({ content: [{ text: 'chunk' }] }); // envelope shape easy to get wrong
    return { message: { role: 'model', content: [{ text: 'canned' }] }, finishReason: 'stop' };
  });
  return { model, get lastRequest() { return lastRequest; } };
}
```

The proposal replaces this with one import.

## Current state

`genkit/testing` (public) exports only `testModels` — a conformance test for a model *plugin*:

```ts
import { testModels } from 'genkit/testing';
```

Internal-only, in `js/ai/tests/helpers.ts`:

- `defineProgrammableModel(registry)` → a model whose responses (including streamed chunks and tool requests) are set per call via a `handleResponse` hook.
- `defineEchoModel(registry)` → a model that echoes the request back as text, for asserting prompt rendering / message assembly.

## Goals

- A public way to mock a model and assert on flow/prompt/tool output.
- Support for the cases an app developer otherwise gets wrong: streaming chunks, tool/function calls, structured (JSON) output.
- Typed inspection of what the model was called with (rendered messages, config, tools).
- Test-runner agnostic (`node:test`, vitest, jest, bun test).
- A documented pattern plus a sample.
- A global guard to fail any test that attempts a real model call (cf. Pydantic AI `ALLOW_MODEL_REQUESTS=False`).

## Non-goals

- Replacing `testModels` (plugin conformance) — complementary, stays.
- Recording/replaying real provider traffic (possible later; see Open questions).
- Evaluators / quality scoring (separate, handled by evaluator plugins).

## Proposal

Add a public `genkit/testing` surface that promotes the internal helpers, renamed for an app-author audience, with thin assertion ergonomics.

### Three tiers

The surface offers three fake models trading control against effort. Pick the lightest one that proves what the test is about.

| Tier | Helper | You provide | Output | Asserts |
|---|---|---|---|---|
| Programmable | `mockModel` | a `respond(req)` per call | what you return (text, structured, tool requests, stream) | flow logic given a known response |
| Echo | `echoModel` | nothing | the rendered request, as text | prompt/message assembly |
| Zero-config | `autoModel` *(Open Q5)* | nothing | schema-valid output and tool calls, derived from `output.schema` and registered tools | that the flow wires up — tools fire, output parses |

The zero-config tier takes no `respond` callback and no canned data: it inspects the request and synthesizes a structurally valid answer. For structured output it reads `output.schema` and returns a conforming value (correct shape, dummy values); if tools are registered it emits a schema-valid call to one. This is Pydantic AI's `TestModel` — it answers "does my flow hang together end-to-end?" without a hand-written fixture per turn. `echoModel` is the trivial floor of the idea; `autoModel` is the schema-satisfying version. Whether `autoModel` ships in v1 is Open Question 5.

The rest of this sketch covers the programmable tier, the workhorse.

### API sketch

```ts
import { genkit } from 'genkit';
import { mockModel, echoModel } from 'genkit/testing';

const ai = genkit({});

// Drive each call's response, inspect each call's input. `respond` may return
// a string, a `{ text | toolRequests | content }` object (auto-wrapped into a
// full GenerateResponseData envelope for you), or a complete
// GenerateResponseData — pick the lightest shape the test needs.
const model = mockModel(ai, {
  respond: (req) => ({ text: 'hello world' }),
});

// Streaming and tool-call shapes are first-class. `sendChunk` takes a string
// shorthand (or a full GenerateResponseChunkData if you need media/custom parts):
mockModel(ai, {
  respond: (req, { sendChunk }) => {
    sendChunk('hel');
    sendChunk('lo');
    return { text: 'hello' };
  },
});

mockModel(ai, {
  respond: () => ({ toolRequests: [{ name: 'lookup', input: { id: 1 } }] }),
});

// Inspection:
model.lastMessage;        // final message of the last *request* (the input
                          //   prompt, not the response), as a Genkit Message —
                          //   read it with `.text` / `.media` / `.toolRequests`
model.lastRequest;        // the full resolved request the model received
model.requests;           // every call this run
```

`echoModel(ai)` returns a model that echoes the rendered request as text.

### Usage in a flow test

```ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { genkit } from 'genkit';
import { mockModel } from 'genkit/testing';

test('summarize flow returns the model text', async () => {
  const ai = genkit({});
  const model = mockModel(ai, { respond: () => ({ text: 'a summary' }) });

  const summarize = ai.defineFlow('summarize', async (doc: string) =>
    (await ai.generate({ model, prompt: `Summarize: ${doc}` })).text
  );

  assert.equal(await summarize('long text'), 'a summary');
  assert.match(model.lastMessage!.text, /Summarize: long text/);
});
```

### Implementation

Move the logic from `js/ai/tests/helpers.ts` into `js/ai/src/testing/` and re-export from `genkit/testing`. The mechanics already exist and are exercised by Genkit's own suite, so this is mostly promotion, naming, and docs.

## Cross-SDK parity

This RFC is JS-scoped, but the same gap exists in Go, Python, and Dart (tracked as a cross-SDK theme). Land the JS shape first as the reference, then port. Keeping the surface aligned across SDKs (a programmable mock, an echo model, call inspection) is a design constraint, not a follow-up.

## Prior art

How other agent/LLM frameworks support testing.

### Vercel AI SDK — `ai/test`

`MockLanguageModelV2` (and `V1`) is a mock model passed straight to `generateText` / `streamText`. `simulateReadableStream` builds a streamed response with configurable `initialDelayInMs` / `chunkDelayInMs`; `mockId` / `mockValues` give deterministic ids. Mocks at the model-abstraction layer, not HTTP.
Docs: <https://ai-sdk.dev/docs/ai-sdk-core/testing>

### LangChain.js — `@langchain/core/utils/testing`

`FakeListChatModel` (ordered predefined responses, `sleep` to simulate latency/streaming), plus `FakeChatModel`, `FakeStreamingLLM`, `GenericFakeChatModel`. Used through the normal `.invoke()` / `.stream()` surface.
Docs: <https://docs.langchain.com/oss/javascript/integrations/chat/fake>

### Pydantic AI — `TestModel` + `FunctionModel`

Two tiers: `TestModel` is zero-config, auto-generating schema-valid tool calls and outputs from the agent's registered schemas; `FunctionModel` drives responses via a function for precise control. Ships a global `ALLOW_MODEL_REQUESTS=False` switch that hard-fails accidental real calls.
Docs: <https://ai.pydantic.dev/testing/>

### OpenAI / Anthropic SDKs

No model-level test double. The community mocks the HTTP layer (msw / nock, respx), which couples tests to wire formats and breaks on transport changes — the approach this RFC avoids.

### Adoption

Public GitHub files referencing each testing API (code search, June 2026):

| Framework | Signal | Files |
|---|---|---|
| Vercel AI SDK | `MockLanguageModelV2` | ~566 |
| LangChain.js | `FakeListChatModel` | ~1,284 |
| Pydantic AI | `from pydantic_ai.models.test import TestModel` | ~1,364 |

Hundreds to low thousands of public repos import the testing helpers specifically; testing is a mainstream part of the workflow in every comparator. Genkit shipping no public equivalent is a parity gap.

_Code search covers only public default branches and de-dupes, so these are floors._

### Design conclusions

- Two tiers is the norm: a programmable model and a zero-config one. `mockModel` covers the first; `echoModel` is the minimum second tier, `autoModel` the stronger one.
- Streaming gets a dedicated helper everywhere (Vercel `simulateReadableStream`, LangChain `sleep`). `sendChunk` is the equivalent.
- A "no real calls" guard (Pydantic `ALLOW_MODEL_REQUESTS=False`) is cheap and worth adopting.
- Mock at the model abstraction, not HTTP — the layer every framework that did this well chose.

## Alternatives considered

**Model middleware (`ai.generate({ use: [...] })`).** The strongest alternative. A middleware that never calls `next()` and returns its own `GenerateResponseData` short-circuits the real model, and it sees `req.tools` / `req.output.schema`, so it can fake tool calls and schema-valid output. It is not the right surface for this, for three reasons:

- Streaming silently breaks in the obvious form. The 2-arg `(req, next)` middleware never receives the streaming callback — `ctx` (carrying `onChunk`) is passed only to the 3-arg `(req, ctx, next)` form; the dispatcher selects between them by `Function.length` (`js/ai/src/generate.ts:414-433`). The natural `(req, next) => …` mock compiles, passes a non-streaming test, then drops every chunk under `generateStream`.
- No typed inspection. Middleware returns only `Promise<GenerateResponseData>`; capturing "what was the model called with, how many times" needs closure-held mutable state — the pattern this RFC removes. `model.lastRequest` / `model.requests` is what middleware does not give you.
- You rebuild the response envelope (`{ message: { role, content }, finishReason }`) and re-derive structured/tool shapes from `req` by hand.

A dedicated fake model avoids all three by construction: it owns its own streaming channel (no `Function.length` dispatch to get wrong), inspection is a property of the model object you hold (`lastRequest` / `requests`), and the response envelope is the model's natural output rather than something you reassemble. That is why `genkit/testing` should offer a fake *model*, not a documented middleware recipe.

**Document the copy-paste mock** instead of exporting one — rejected: pushes framework-internal envelopes onto every app, the error-prone part.

**A separate `@genkit-ai/testing` package** — possible, but `genkit/testing` already exists; extending it is lower-friction. (Open question.)

**Generic DI / manual fake plugin** — works, but high-ceremony; the point is a one-liner.

## Open questions

1. Naming: keep internal names (`defineProgrammableModel` / `defineEchoModel`) or app-friendly (`mockModel` / `echoModel`)? Aliases?
2. Subpath `genkit/testing` vs a dedicated `@genkit-ai/testing` package (tree-shaking, dep weight).
3. Ship inspection helpers now, or the models first?
4. Record/replay of real provider responses — in scope later, or never?
5. Ship the zero-config `autoModel` (cf. Pydantic `TestModel`) in v1, or start with `echoModel` + `mockModel` only?
6. Mock-by-name / registry override: code under test often references models by string (`ai.generate({ model: 'googleai/...' })`) rather than passing a reference. Should `mockModel` register itself under a given name and shadow an existing registered model to support these non-DI patterns?
7. Global guard API: what mechanism for the "no real calls" switch — env var (e.g. `GENKIT_ALLOW_MODEL_REQUESTS=false`), or a programmatic call on the `genkit` instance / testing module?
8. Inspection naming + lifecycle: `lastMessage` reads the request (input) — rename to `lastRequestMessage` to avoid confusion with the response? And should the mock expose `reset()` to clear `requests` between tests that reuse one instance?

## Rollout

1. RFC PR (this doc) → discussion.
2. Implementation PR: `genkit/testing` exports + tests.
3. Docs page + a sample app demonstrating flow/tool/streaming tests.
4. Cross-SDK ports tracked separately.
