# Agent HTTP wire protocol (Python ↔ JS)

This document describes the **streamFlow SSE contract** used by `GenkitChatTransport`,
`remoteAgent`, and the Python FastAPI handler. Types are defined in
[`genkit-tools/common/src/types/agent.ts`](../../../genkit-tools/common/src/types/agent.ts).

## Mental model

```
Turn 1                          Turn 2                          Turn 3 (resume)
────────                        ────────                        ───────────────
POST /api/chat/weather          POST /api/chat/weather          POST /api/chat/banking
{ data, init: {sessionId} }     { data, init: {sessionId} }     { data: {resume}, init }
        │                               │                               │
        ▼                               ▼                               ▼
SSE: message*, result            SSE: message*, result            SSE: message*, result
(connection closed)              (connection closed)              (connection closed)
```

**Each turn is one HTTP request.** The client opens a connection, sends one input,
reads stream chunks until `result`, then the connection closes. Multi-turn chat
re-initializes the connection every time; continuity is **not** kept on the wire —
only in session state keyed by `sessionId` (server store) or `state` (client-managed).

---

## Request envelope

Every turn:

```http
POST /api/chat/<agentName> HTTP/1.1
Accept: text/event-stream
Content-Type: application/json
```

```json
{
  "data": { /* AgentInput */ },
  "init": { /* AgentInit */ }
}
```

### `init` (AgentInit)

| Field | When | JS (`GenkitChatTransport`) | Python (`_load_session`) |
|-------|------|---------------------------|--------------------------|
| `sessionId` | Server-managed store | Always sent (`useChat` `id` must be UUID) | Loads latest leaf snapshot for session |
| `snapshotId` | Optional pin | Supported by `remoteAgent` | Exact snapshot resume |
| `state` | Client-managed (no store) | Not used by transport | Full session blob from prior `result.state` |

### `data` (AgentInput)

**New user message** (normal turn):

```json
{
  "messages": [
    {
      "role": "user",
      "content": [{ "text": "Weather in Paris?" }]
    }
  ]
}
```

Only the **latest user message** is sent per turn (history lives in the store).

**Interrupt resume** (after `finishReason: "interrupted"`):

```json
{
  "resume": {
    "respond": [
      {
        "toolResponse": {
          "name": "userApproval",
          "ref": "<tool-call-id>",
          "output": { "approved": true }
        }
      }
    ],
    "restart": [
      {
        "toolRequest": {
          "name": "someTool",
          "ref": "<tool-call-id>",
          "input": { /* must match interrupted request exactly */ }
        },
        "metadata": { "resumed": true }
      }
    ]
  }
}
```

`respond` and `restart` can appear together. `GenkitChatTransport` derives these from
resolved tool parts in the UI message list.

---

## Response envelope (SSE)

Delimiter: `\n\n` (same as JS `streamFlow`).

Each event is one line:

```
data: {"message":{...}}

data: {"message":{...}}

data: {"result":{...}}
```

On error (inside the stream):

```
data: {"error":{"status":"INVALID_ARGUMENT","message":"..."}}
```

| Key | Type | Meaning |
|-----|------|---------|
| `message` | `AgentStreamChunk` | Streaming chunk (may repeat) |
| `result` | `AgentOutput` | Terminal success payload |
| `error` | `{ status, message, details? }` | Terminal failure |

Python serializes with **camelCase** keys (`modelChunk`, `turnEnd`, `finishReason`, …)
via Pydantic `alias_generator=to_camel`.

---

## Example: two-turn weather (full wire)

**Turn 1 — request**

```json
{
  "data": {
    "messages": [{ "role": "user", "content": [{ "text": "Weather in Paris?" }] }]
  },
  "init": { "sessionId": "550e8400-e29b-41d4-a716-446655440000" }
}
```

**Turn 1 — SSE (illustrative shapes)**

```json
data: {"message":{"modelChunk":{"role":"model","content":[{"toolRequest":{"name":"getWeather","ref":"...","input":{"city":"Paris"}}}]}}}
data: {"message":{"modelChunk":{"role":"tool","content":[{"toolRequest":{"name":"getWeather",...}}]}}}
data: {"message":{"modelChunk":{"role":"model","content":[{"text":"It's 18°C in Paris."}]}}}
data: {"message":{"turnEnd":{"snapshotId":"<uuid>","finishReason":"stop"}}}
data: {"result":{"snapshotId":"<uuid>","message":{"role":"model","content":[{"text":"..."}]},"finishReason":"stop","artifacts":[]}}
```

**Turn 2 — request** (new TCP connection, same `sessionId`)

```json
{
  "data": {
    "messages": [{ "role": "user", "content": [{ "text": "What city did I ask about?" }] }]
  },
  "init": { "sessionId": "550e8400-e29b-41d4-a716-446655440000" }
}
```

Server reloads messages from the store; the model sees turn-1 history.

---

## Example: banking interrupt + resume

**Turn 1** ends with `finishReason: "interrupted"` and a `userApproval` tool request
in the stream (model chunks + optional `role: "tool"` echo chunk).

**Turn 2 — request** (resume only, no `messages`):

```json
{
  "data": {
    "resume": {
      "respond": [{
        "toolResponse": {
          "name": "userApproval",
          "ref": "<ref-from-turn-1>",
          "output": { "approved": true, "feedback": "Looks good" }
        }
      }]
    }
  },
  "init": { "sessionId": "550e8400-e29b-41d4-a716-446655440000" }
}
```

---

## AgentStreamChunk fields (`define_agent` runtime)

Authoritative schema: `AgentStreamChunkSchema` in `agent.ts` (tools).

| Field | Emitted by `define_agent`? | JS source | Python source |
|-------|---------------------------|-----------|---------------|
| `modelChunk` | **Yes** — streaming generate tokens/tool parts | `definePromptAgent` → `sendChunk({ modelChunk })` | `define_agent` → `_on_chunk` → `ctx.send_chunk` |
| `modelChunk` (`role: "tool"`) | **Yes** — on interrupt | Extra chunk after `finishReason === 'interrupted'` | `_emit_interrupt_tool_chunk` |
| `customPatch` | **If** `sess.update_custom()` during turn | Runtime `session.on('customChanged')` | Runtime `session.on_custom_changed` → `_emit_custom_patch` |
| `artifact` | **If** `sess.add_artifacts()` during turn | Runtime `session.on('artifactAdded'/'artifactUpdated')` | Runtime `session.on_artifact_changed` → `_emit_artifact` |
| `turnEnd` | **Yes** — end of every turn (unless detached) | Runtime `onEndTurn` → `emitChunk({ turnEnd })` | Runtime `_emit_turn_end` via `SessionRunner.on_end_turn` |

`define_agent` itself only emits `modelChunk` (+ interrupt tool chunk). The other
three chunk types come from the **shared runtime** (`defineCustomAgent` / `_AgentRuntime`)
when session state mutates or a turn completes.

### `turnEnd` shape

```json
{
  "turnEnd": {
    "snapshotId": "<uuid when store configured>",
    "finishReason": "stop | interrupted | failed | ..."
  }
}
```

When `store` is set, `snapshotId` is included. When not, only `finishReason`.

### `result` (AgentOutput) shape

```json
{
  "snapshotId": "...",
  "message": { "role": "model", "content": [...] },
  "artifacts": [],
  "finishReason": "stop",
  "state": { /* only when no store */ }
}
```

On graceful turn failure (JS only today):

```json
{
  "finishReason": "failed",
  "error": { "status": "...", "message": "..." },
  "snapshotId": "...",
  "state": { /* last-good state */ }
}
```

---

## JS vs Python parity gaps (`define_agent`)

| Behavior | JS | Python | Status |
|----------|----|----|--------|
| `modelChunk` streaming | ✅ | ✅ | Match |
| Interrupt `role: "tool"` chunk | ✅ | ✅ | Match |
| `turnEnd` per turn | ✅ | ✅ | Match |
| `customPatch` on `update_custom` | ✅ | ✅ | Match |
| `artifact` on `add_artifacts` | ✅ | ✅ | Fixed (runtime listener) |
| `clientTransform.chunk` redaction | ✅ | ✅ | Match |
| Resume validation vs history | ✅ `validateResumeAgainstHistory` | ✅ `validate_resume_against_history` | Match |
| Graceful turn failure (`finishReason: failed` + last-good state) | ✅ `SessionRunner.lastGoodState` | ✅ `SessionRunner` + `_failed_agent_output` | Match |
| Abort via store → `abortSignal` | ✅ `onSnapshotStateChange` on detach | ✅ `_watch_snapshot_abort` on detach | Match |
| Detach early return | ✅ | ✅ | Match |
| Turn trace span name | `runTurn-N` (1-based) | `runTurn-N` (1-based) | Match |
| Root action subtype | `action/agent` | `action/agent` (`ActionKind.AGENT`) | Match |
| Root `agent:sessionId` metadata | set when store session loaded | set on root span when `session_id` present | Match |
| `render` span (prompt agents) | `promptTemplate` under turn | `promptTemplate` via `_prepare()` | Match |
| ToolApproval interrupt span | `flowStep` | `flowStep` | Match |
| Message persistence | Prompt-tag strip (`definePromptAgent`) | Prompt-tag strip via `_prepare` + `strip_preamble` | Match |
| `define_agent` vs `define_prompt_agent` | `definePrompt` + `definePromptAgent` | `ExecutablePrompt` + `define_prompt_agent` | Match |

---

## Runnable reference client

Each audit scenario has a dedicated script under `src/examples/basic_samples/` with
**inline** code (branching demos live in `src/examples/branching_samples/`):

| Script | Scenario |
|--------|----------|
| `basic_samples/01_define_agent_with_store.py` | Weather agent + `sessionId` |
| `basic_samples/02_define_agent_no_store.py` | Client passes `init.state` each turn |
| `basic_samples/03_interrupt_resume_with_store.py` | Banking interrupt + resume |
| `basic_samples/04_interrupt_resume_no_store.py` | Interrupt without store |
| `basic_samples/05_define_prompt_agent.py` | `define_prompt_agent` |
| `basic_samples/06_define_custom_agent.py` | Hand-written generate loop |
| `basic_samples/07_artifacts_custom_patch.py` | `customPatch` + `artifact` chunks |
| `basic_samples/08_graceful_failure.py` | `finishReason: failed` |
| `basic_samples/09_detach.py` | Detach via `conn.detach()` (in-process) |
| `basic_samples/10_abort.py` | `store.abort_snapshot()` after detach |
| `branching_samples/01_fork_sibling_snapshots.py` | Fork via `snapshotId` |

```bash
cd py/samples/agents
uv run uvicorn src.main:app --port 8080   # HTTP serving

uv run python src/examples/basic_samples/01_define_agent_with_store.py
uv run python src/examples/branching_samples/01_fork_sibling_snapshots.py
uv run python src/verify.py
```

Source: [`src/examples/basic_samples/`](src/examples/basic_samples/), [`src/examples/branching_samples/`](src/examples/branching_samples/)
