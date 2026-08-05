# a2a-testapp

An end-to-end demo of [`@genkit-ai/a2a`](../../plugins/a2a): a Genkit **agent**
exposed over the [A2A (Agent2Agent) protocol](https://a2a-protocol.org/) and
driven by the real [`@a2a-js/sdk`](https://www.npmjs.com/package/@a2a-js/sdk)
client.

The sample is a **travel concierge** agent that is intentionally rich enough to
exercise every part of the A2A mapping:

| Feature | What it demonstrates |
| ------- | -------------------- |
| `getWeather` / `searchFlights` tools + streamed text | Genkit model chunks → A2A `artifact-update` events |
| `confirmBooking` interrupt | A turn pausing → A2A terminal `input-required`, then **resuming the same task** |
| Snapshot-native task identity | An A2A `taskId` **is** the originating turn's Genkit snapshot id; `getTask` reads straight from the agent's `SessionStore` |
| Shared `contextId` across turns | A2A `contextId` ↔ Genkit `sessionId`, so the agent remembers the conversation |
| Agent card discovery | `deriveAgentCard` served at `/.well-known/agent-card.json` |

## Prerequisites

Set a Gemini API key (the agent uses `googleai/gemini-flash-latest`):

```bash
export GEMINI_API_KEY=...   # or GOOGLE_API_KEY
```

## Run it

In one terminal, start the A2A server:

```bash
pnpm server
```

In another terminal, run the client demo:

```bash
pnpm client
```

The client will:

1. Discover the agent card and print the agent's name/description.
2. **Turn 1** — plan a trip to Tokyo (you'll see the streamed weather/flight
   results arrive as artifact updates).
3. **Turn 2** — ask to book; the agent pauses and the task ends in
   `input-required`, carrying the `confirmBooking` interrupt.
4. **Turn 3** — resume the *same task* with a `toolResponse` data part approving
   the booking; the task streams to `completed` with a confirmation code.
5. **getTask** — fetch the resumed task by id, showing the handler rebuilds it
   from the agent's snapshot store (see *Task identity* below).
6. **Turn 4** — a follow-up question proving the session remembers the trip.

## Files

| File | Purpose |
| ---- | ------- |
| `src/genkit.ts` | Genkit instance + model config |
| `src/concierge-agent.ts` | The agent, its tools, and the `confirmBooking` interrupt |
| `src/server.ts` | Express server wiring `GenkitA2ARequestHandler` to the A2A JSON-RPC + card endpoints |
| `src/client.ts` | A2A client demo driver |

## Dev UI

You can also explore the server with the Genkit Dev UI:

```bash
pnpm genkit:dev
```

## How resume works on the wire

When Turn 2 ends in `input-required`, the terminal status message carries the
paused tool request as a `data` part tagged `genkit:interrupt`:

```jsonc
{
  "kind": "data",
  "data": { "name": "confirmBooking", "ref": "...", "input": { /* ... */ } },
  "metadata": { "genkit:type": "toolRequest", "genkit:interrupt": true }
}
```

To resume, the client sends a follow-up message **targeting the same `taskId`**
with the resolved tool output as a `toolResponse` data part:

```jsonc
{
  "kind": "data",
  "data": { "name": "confirmBooking", "ref": "...", "output": { "confirmed": true } },
  "metadata": { "genkit:type": "toolResponse" }
}
```

The handler detects the task is `input-required`, converts this into the Genkit
`resume.respond` payload, and continues the turn.

## Task identity & the task store

Because `conciergeAgent` is **server-managed** (it has a `SessionStore`), the
A2A handler is snapshot-native: an A2A `taskId` is the Genkit snapshot id of the
turn that originated it (reserved up front and surfaced on the turn's
`turnStart` stream chunk). `getTask` and interrupt-resume therefore read
straight from the agent's own store via `agent.getSnapshot` — there is no
separate copy of task state.

Resuming an interrupted turn (Turn 3) produces a **new** snapshot, so the task
"advances" past its originating snapshot. That single advancement is the only
thing recorded in the handler's `taskStore` (`taskId → { contextId, snapshotId
}`), which `src/server.ts` configures explicitly with the default
`InMemoryA2ATaskStore`. Supply a durable implementation to survive restarts or
span processes; durable conversation state itself lives in the agent's own
`SessionStore`.

