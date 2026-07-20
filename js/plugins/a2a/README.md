# @genkit-ai/a2a

A [Genkit](https://github.com/genkit-ai/genkit) plugin that exposes a Genkit
**Agent** over the [A2A (Agent2Agent) protocol](https://a2a-protocol.org/)
using the [`@a2a-js/sdk`](https://www.npmjs.com/package/@a2a-js/sdk).

It provides `GenkitA2ARequestHandler`, a custom implementation of the SDK's
`A2ARequestHandler` that runs a Genkit agent turn for each incoming A2A
message, streams the agent's output back as A2A task events, and derives the
agent's `AgentCard` automatically.

## Installation

```bash
npm install @genkit-ai/a2a @a2a-js/sdk
```

### Peer dependencies

| Package | Required | Notes |
| ------- | -------- | ----- |
| `genkit` | ✅ | `>=1.0.0` — the agent API lives in the `genkit/beta` subpath. |
| `@a2a-js/sdk` | ✅ | `^0.3.0` — provides the A2A server handler interfaces and Express adapters. |

## Quick start

```ts
import {
  agentCardHandler,
  jsonRpcHandler,
  UserBuilder,
} from '@a2a-js/sdk/server/express';
import { GenkitA2ARequestHandler } from '@genkit-ai/a2a';
import { genkit } from 'genkit/beta';
import { googleAI } from '@genkit-ai/google-genai';
import express from 'express';

const ai = genkit({ plugins: [googleAI()] });

const weatherAgent = ai.defineAgent({
  name: 'weatherAgent',
  description: 'Tells you the weather.',
  model: 'googleai/gemini-flash-latest',
  tools: [/* your tools here */],
});

const PORT = 3000;

// The card is derived automatically from the agent's name and description.
// Only `url` is needed so the card knows where the agent is hosted. An explicit
// `card` may be passed to override or extend the derived card.
const a2aHandler = new GenkitA2ARequestHandler({
  agent: weatherAgent,
  url: `http://localhost:${PORT}`,
});

const app = express();
app.use(express.json());

// A2A JSON-RPC endpoint (primary A2A transport)
app.use(
  '/',
  jsonRpcHandler({
    requestHandler: a2aHandler,
    userBuilder: UserBuilder.noAuthentication,
  }) as any // eslint-disable-line @typescript-eslint/no-explicit-any
);

// Agent card discovery
app.use(
  '/.well-known/agent-card.json',
  agentCardHandler({ agentCardProvider: a2aHandler }) as any // eslint-disable-line @typescript-eslint/no-explicit-any
);

app.listen(PORT);
```

## How it works

`GenkitA2ARequestHandler` implements the `@a2a-js/sdk` `A2ARequestHandler`
interface directly (no `DefaultRequestHandler` / `AgentExecutor`). For each
incoming message it:

1. Starts (or resumes) a Genkit agent chat keyed by the A2A `contextId`.
2. Streams the agent turn, translating Genkit stream chunks into A2A task
   events.
3. Emits a terminal status update derived from the turn's finish reason.

### Identifiers

| A2A | Genkit | Notes |
| --- | --- | ----- |
| `contextId` | `sessionId` | A server-managed agent (one with a `store`) resumes its session across A2A tasks that share a `contextId`. |
| `taskId` | one agent turn | Each task maps to a single turn. The handler keeps an in-memory record of tasks so `getTask` works. |

> Conversation state is owned by the **agent's own `SessionStore`**, not by
> A2A. The handler's task map is only for `getTask` and interrupt-resume
> detection.

### Streaming event model

For each turn the handler emits:

1. A `Task` (state `submitted`).
2. A `TaskStatusUpdateEvent` (state `working`).
3. A `TaskArtifactUpdateEvent` per model chunk of **user-facing** content
   (text / reasoning / media), all sharing one `artifactId` (`append: false`
   for the first chunk, `append: true` thereafter). Internal tool-call
   mechanics (`toolRequest` / `toolResponse` / `data`) are **not** streamed as
   artifacts; interrupts are surfaced separately on the terminal
   `input-required` status (see below).
4. A terminal `TaskStatusUpdateEvent` (`final: true`) whose state is derived
   from the Genkit finish reason:

| Genkit `finishReason` | A2A terminal state |
| --------------------- | ------------------ |
| `stop` / `length` / `other` / `unknown` | `completed` (final message echoed in `status.message`) |
| `interrupted` | `input-required` (interrupt tool requests carried in `status.message.parts`) |
| `failed` | `failed` (error text in `status.message`) |
| `aborted` | `canceled` |

### Part mapping

A2A's part model (`text` | `file` | `data`) is narrower than Genkit's, so the
extra Genkit semantics are encoded in part `metadata` under a `genkit:`
namespace. This keeps a **Genkit ↔ Genkit** round-trip lossless while staying
interoperable with generic A2A clients (which see plain text / file / data
parts and ignore the metadata).

| Genkit part | A2A part |
| ----------- | -------- |
| `text` | `TextPart` |
| `reasoning` | `TextPart` + `metadata['genkit:reasoning'] = true` |
| `media` (remote url) | `FilePart` (`FileWithUri`) |
| `media` (base64 `data:` url) | `FilePart` (`FileWithBytes`) |
| `toolRequest` | `DataPart` + `metadata['genkit:type'] = 'toolRequest'` |
| `toolResponse` | `DataPart` + `metadata['genkit:type'] = 'toolResponse'` |
| `data` / `custom` | `DataPart` + `metadata['genkit:type'] = 'data' \| 'custom'` |

On the way back in, the `genkit:type` discriminator reconstructs the exact
Genkit part; parts from non-Genkit clients fall back to a structural
interpretation (text → text, file → media, data → data).

### Interrupts (human-in-the-loop)

When an agent turn pauses on an interrupt, the terminal event is
`input-required` and its `status.message.parts` carries the interrupt
`toolRequest`(s) as data parts (tagged `metadata['genkit:interrupt']`).

To resume, send a follow-up A2A message **targeting the same `taskId`** with:

- a **respond** — a data part tagged `genkit:type = 'toolResponse'` whose
  `data` is the `{ name, ref, output }` of the resolved tool; or
- a **restart** — a data part tagged `genkit:type = 'toolRequest'` plus
  `metadata['genkit:restart']` (the value becomes the tool's `resumed`
  argument), instructing the agent to re-run the tool.

The handler detects that the referenced task is in `input-required` and
converts these into the Genkit `resume` payload (`{ respond, restart }`).

## API

### `GenkitA2ARequestHandler`

```ts
new GenkitA2ARequestHandler({
  agent,           // the Genkit agent (from ai.defineAgent)
  url,             // base URL where the agent is hosted (for the card)
  card,            // optional: partial/full AgentCard to override/extend
  version,         // optional: the agent's version string (default '0.0.0')
});
```

Implements the full `A2ARequestHandler` interface. `sendMessage`,
`sendMessageStream`, `getTask`, `getAgentCard`, and
`getAuthenticatedExtendedAgentCard` are supported. `cancelTask`, push
notification configuration, and `resubscribe` currently throw
("not supported").

### Mapping utilities

The package also exports the low-level mapping helpers, useful for building
custom handlers or A2A clients that talk to Genkit agents:

| Function | Direction | Description |
| -------- | --------- | ----------- |
| `genkitPartToA2A` / `genkitPartsToA2A` | Genkit → A2A | Map Genkit parts to A2A parts |
| `a2aPartToGenkit` / `a2aPartsToGenkit` | A2A → Genkit | Map A2A parts to Genkit parts |
| `a2aMessageToGenkit` | A2A → Genkit | Map an A2A message to a Genkit `MessageData` |
| `genkitMessageToA2AParts` | Genkit → A2A | Map a Genkit message's content to A2A parts |
| `a2aMessageToResumeInput` | A2A → Genkit | Build the Genkit `AgentInput` (message or resume) for an incoming A2A message |
| `genkitRoleToA2A` | Genkit → A2A | Map a Genkit role to an A2A role |

Plus the `A2A_METADATA` key constants and the `GenkitPartType` discriminator
values.

### Agent card

| Function | Description |
| -------- | ----------- |
| `deriveAgentCard(agent, options)` | Build an `AgentCard` from a Genkit agent and options (`url`, `version`, partial `card`) |
| `getAgentName(agent)` / `getAgentDescription(agent)` | Read the agent's registered name / description |

## Limitations

- **Task cancellation**, **push notifications**, and **stream resubscription**
  are not yet supported.
- The task store is **in-memory** (per handler instance); durable conversation
  state lives in the agent's own `SessionStore`.
- Genkit `customPatch` (live custom-state) chunks are **not** surfaced over
  A2A, as A2A has no direct equivalent.

## Requirements

- Genkit `>=1.0.0` (agent API via `genkit/beta`)
- `@a2a-js/sdk` `>=0.3.0`
- Node.js `>=20`

## License

Apache-2.0 — see [LICENSE](./LICENSE).
