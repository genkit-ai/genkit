# RFC: Bidirectional agent streaming over WebSocket

**Status:** Proposal · **Audience:** product engineers building agent UIs · **Companion:** [`WIRE_PROTOCOL.md`](WIRE_PROTOCOL.md) (the HTTP/SSE contract this extends)

---

## 1. Why

Today an agent turn is one HTTP request. The client sends everything it has, calls
`close()`, and then *only reads* until the turn ends. That's half-duplex: **you cannot
talk to the agent while the agent is talking to you.** The whole interesting space of
real-time agents — barge-in / cancel, queuing feedback the agent chooses to absorb, live
voice, co-editing a document — is off the table.

The thing is, the *in-process* agent API is already full-duplex. `agent.stream_bidi()`
returns an `AgentConnection` with independent `send()` and `receive()`:

```python
conn = await agent.stream_bidi(AgentInit(session_id=sid))
await conn.send_text("Draft the launch post")
async for chunk in conn.receive():        # output streaming out...
    if looks_wrong(chunk):
        await conn.send_text("shorter, drop the intro")   # ...while input goes in
```

SSE can't carry that second `send_text`. **A WebSocket can.** This RFC proposes the wire
format and client SDKs to expose the duplex `AgentConnection` that already exists, over a
single persistent socket — *one socket = one conversation*, instead of one POST per turn.

### What stays the same

The agent author writes the **exact same bidi function**. Transport is swappable:
in-process, HTTP/SSE (half-duplex projection), or WebSocket (full duplex). No agent code
changes. The server-side chunk serialization (`message` / `result` / `error`) is reused
verbatim from the SSE handler.

---

## 2. Goals / non-goals

**Goals**

- A wire format for full-duplex agent sessions over one WebSocket.
- 1:1 mapping to the existing `AgentConnection` surface (`send`, `send_text`,
  `send_resume`, `detach`, `close`, `receive`, `output`).
- Mid-turn input (queue-and-absorb), cancel/barge-in, and reattach to a detached run.
- Python and JS client SDKs that are **drop-in** for the in-process connection.

**Non-goals**

- Replacing SSE. `useChat` and request/response stay on SSE; WS is opt-in for apps that
  need duplex.
- A new session/state model. Sessions, snapshots, stores, and resume semantics are
  unchanged — WS just keeps one session warm for the life of the socket.
- Transport-level reliability guarantees beyond what's in §7.
- Single-frame "steer" (atomic abort-and-redirect of the in-flight turn). Cancel +
  a follow-up input covers it in two frames today; steer is deferred (§11).

---

## 3. Mental model

```
HTTP / SSE (today)                         WebSocket (this RFC)
──────────────────                         ────────────────────
Turn 1  POST → SSE → close                 open socket ─┐
Turn 2  POST → SSE → close                  init        │  one session,
Turn 3  POST → SSE → close                  input ──▶    │  stays warm,
                                            ◀── message  │  duplex the
session reloaded from store each turn       input ──▶    │  whole time
                                            ◀── message  │
                                            ◀── turnEnd  │
                                            close ──▶    │
                                            ◀── result  ─┘
```

The single most important consequence: over SSE each turn is a fresh connection and the
session is reloaded from the store every time, so there's nowhere to put input mid-turn.
Over WS the connection *is* the session for its lifetime, so the input channel stays open
the whole time the output channel is streaming.

### Turn boundary vs invocation end

This distinction is collapsed on SSE (one socket = one turn) and **separated** on WS:

| Signal | Meaning | Frequency on WS |
|--------|---------|-----------------|
| `turnEnd` chunk | one agent turn finished | many per socket |
| `result` frame | the whole invocation finished | exactly one, right before close |

---

## 4. Connection lifecycle

```
1. Client opens   WS  /api/agents/<name>/ws
2. Client → init      { "init": AgentInit }
3. Server → ready     { "ready": { "traceId": "..." } }      (optional ack)
4. Client → input     { "input": AgentInput }                (one per turn, or mid-turn)
5. Server → message*  { "message": AgentStreamChunk }        (repeats; last is turnEnd)
   ... steps 4–5 repeat for as many turns as the client drives ...
6. Client → close     { "close": true }                      (no more inputs)
7. Server → result    { "result": AgentOutput }              (terminal)
8. Server closes the socket (code 1000).
```

Either side may end early: the client with a normal WS close, the server with an `error`
frame followed by close. A dropped socket without `close` is treated like a `detach` (see
§6.4) when a store is configured, otherwise the invocation is abandoned.

---

## 5. Wire format

WebSocket text frames, each a single JSON object with **one** top-level key acting as the
discriminator. Keys are **camelCase**, identical to the SSE payloads.

> An agent is a `BidiAction` (`define_agent` → `define_bidi_action(kind=AGENT)`), and the
> envelope below — `init` / `input` / `close` / `cancel` ↔ `message` / `result` / `error` —
> is just the serialization of `BidiAction.stream_bidi()`, which has nothing agent-specific
> in it. So the framing is the generic bidi-action wire protocol; **"agent" is the profile**
> that pins the slots to `AgentInit` / `AgentInput` / `AgentStreamChunk` / `AgentOutput` and
> layers on agent semantics (turn boundaries, resume, detach, sessions). It's the only
> profile today, so the rest of this RFC speaks in agent terms.

### 5.1 Client → server

| Frame | Shape | Maps to `AgentConnection` |
|-------|-------|---------------------------|
| **init** | `{ "init": AgentInit }` | the `stream_bidi(init)` handshake; first frame only |
| **input** | `{ "input": AgentInput }` | `send()` / `send_text()` / `send_resume()` |
| **cancel** | `{ "cancel": true }` | abort the running turn, keep the socket open |
| **detach** | `{ "input": { "detach": true } }` | `detach()` — background the run, then close |
| **close** | `{ "close": true }` | `close()` — no more inputs |

`AgentInput` is unchanged: `{ messages?, resume?, detach? }`. A new-message turn,
an interrupt resume, and a detach all flow through the existing type — see
[`WIRE_PROTOCOL.md` §`data`](WIRE_PROTOCOL.md#data-agentinput).

`init` must be the first frame. Sending `input` before `init` is `FAILED_PRECONDITION`.

### 5.2 Server → client

| Frame | Shape | Maps to |
|-------|-------|---------|
| **ready** | `{ "ready": { "traceId": string } }` | handshake ack; lets the client log/trace |
| **message** | `{ "message": AgentStreamChunk }` | `receive()` chunk — `modelChunk` / `customPatch` / `artifact` / `turnEnd` |
| **result** | `{ "result": AgentOutput }` | `output()` — sent once, right before close |
| **error** | `{ "error": { status, message, details? } }` | terminal failure |

`AgentStreamChunk`, `AgentOutput`, and the error envelope are **byte-for-byte the same** as
the SSE handler emits — only the framing differs (no `data: ` prefix; one JSON object per WS
message instead of `\n\n`-delimited lines).

### 5.3 Backpressure & ordering

- The server keeps the in-process queue semantics: the **input queue has depth 1** (a new
  `input` blocks at the protocol layer until the previous one is accepted), the **output
  queue is unbounded**. The client SDK surfaces this as an awaitable `send()`.
- Frames are ordered per direction (WS guarantees this). `cancel` is processed as soon as
  it's read, which may land mid-turn — that's the point.
- `result` is always the last server frame.

---

## 6. Behaviors that only exist on the duplex socket

### 6.1 Queue-and-absorb (the agent decides)

The client sends additional `input` frames while a turn runs. They land in the server's
input queue. The agent fn drains the queue **at its own checkpoints** (e.g. between tool
calls or generate steps) and folds pending user input into the next model call. No abort,
no restart — the agent chooses when to look.

### 6.2 Cancel / barge-in (the client forces it)

`{ "cancel": true }` trips the turn's `abort_signal`. The current generation stops and
last-good state is preserved (same machinery as graceful failure). The socket stays open,
so the client can immediately send a corrected `input` as the next turn — "stop, do this
instead" in two frames, like cutting off a model mid-sentence in voice mode. (Folding those
two frames into one atomic *steer* is deferred — see §11.)

The runtime plumbing already exists: `abort_signal` is an `asyncio.Event` threaded into
`generate_action`. Today only the store's abort watcher sets it; the WS handler sets the
same event when it reads a `cancel` frame. New API surface, not new machinery.

### 6.3 In-session resume (no second POST)

Interrupt → approve → resume happens on the **same socket**. The `userApproval` tool
request arrives as a `message` chunk; the client replies with
`{ "input": { "resume": { ... } } }`. The session never left memory, so there's no store
round-trip and no reload latency between approval and continuation.

### 6.4 Detach & reattach

`{ "input": { "detach": true } }` backgrounds the invocation. The runtime saves a pending
snapshot, sends a terminal `result` frame carrying that `snapshotId`, and **closes the
socket** — but the run keeps going server-side (further chunks are suppressed on the wire;
a background finalizer rewrites the snapshot when the fn finishes). You may detach *with* a
final payload (`messages` + `detach` in one `AgentInput`): it runs, then backgrounds.

**Detach requires a store** — without one there's nowhere to persist the backgrounded run,
so the server returns `FAILED_PRECONDITION`.

Later the client opens a **new** socket with `{ "init": { "snapshotId": ... } }` (or
`sessionId`) and resumes watching — and can inject new input into the still-running task.
"Tap the long-running agent on the shoulder." (`detach` is an existing `AgentConnection`
method, so the WS client gets it for free — see §8.)

---

## 7. Reliability notes for product engineers

- **Reconnect = new socket, same session.** Reopen with `init.sessionId` (server store) or
  `init.snapshotId` to pin an exact point. State lives in the store, never on the wire, so a
  dropped connection loses only un-snapshotted in-flight tokens, not the conversation.
- **No store?** The socket holds the only copy of session state. Dropping it loses the
  session — same trade-off as `init.state` over SSE, just for the socket's lifetime.
- **Idle/keepalive:** standard WS ping/pong. Server may close idle sockets; clients reopen
  with the same `init`.
- **Auth:** same header/cookie story as the SSE handler, applied at the WS upgrade request.

---

## 8. Client SDK — Python (`genkit.client`)

The client lives under a transport-neutral **`genkit.client`** namespace, not under a server
plugin: a client only knows the wire protocol and shouldn't import from
`genkit.plugins.fastapi` when the server might be Flask, Cloud Run, or anything else.

`genkit.client` hosts **both** agent transports behind two symmetric constructors that
return the **same connection protocol**, so swapping HTTP/SSE for WebSocket is a one-line
change:

```
genkit/client/__init__.py    # connect_ws_agent, connect_http_agent, AgentConnection, AgentClientError
genkit/client/_ws.py         # WebSocket duplex connection (this RFC)
genkit/client/_http.py       # HTTP/SSE connection, relocated from plugins/fastapi (see 8.5)
```

| Constructor | Transport | Shape |
|-------------|-----------|-------|
| `connect_ws_agent(url, …)` | WebSocket | persistent **full-duplex** session (this RFC) |
| `connect_http_agent(url, …)` | HTTP/SSE | **half-duplex** — one request per turn |

Both return an object implementing the `AgentConnection` protocol (§8.2) and are async
context managers. The half-duplex transport implements the same protocol but can't honor the
duplex-only methods — `send` while `receive`-ing, and `cancel` — which raise
`AgentClientError('UNIMPLEMENTED')`. Everything else (`send_text`, `send_resume`, `detach`,
`close`, `receive`, `output`) behaves the same on both, so app code that doesn't reach for
duplex runs unchanged across transports.

> Naming: `connect_*_agent` keeps the transport explicit and greppable. If you'd rather have
> one config-driven entry, `connect_agent(url, *, transport='ws' | 'http', …)` can wrap these
> two — see §11.

### 8.1 Constructors

```python
from genkit.client import connect_ws_agent, connect_http_agent
from genkit.agent import AgentInit

def connect_ws_agent(
    url: str,                       # ws(s)://…/ws endpoint (§11)
    *,
    init: AgentInit | None = None,
    headers: Headers | Callable[[], Awaitable[Headers]] | None = None,
    open_timeout: float = 30.0,
) -> AgentConnection:
    """Open a full-duplex agent session over WebSocket. Async context manager:
    entering connects, sends the `init` frame, and waits for the server's `ready`;
    exiting calls `close()` and shuts the socket."""

def connect_http_agent(
    url: str,                       # http(s):// agent endpoint
    *,
    init: AgentInit | None = None,
    headers: Headers | Callable[[], Awaitable[Headers]] | None = None,
    timeout: float = 120.0,
) -> AgentConnection:
    """Half-duplex agent session over HTTP/SSE — each turn is one POST that streams
    back SSE (the §`WIRE_PROTOCOL.md` contract). Same protocol as the WS connection,
    minus the duplex-only methods."""
```

`init` carries `sessionId` / `snapshotId` / `state` (the §5 `init` frame on WS; the request
`init` field on HTTP). `headers` is applied at the WS upgrade or on each POST — static or
resolved per connection — and is where auth goes.

### 8.2 The connection

`AgentConnection` is a **structural protocol** that the in-process
`genkit.agent.AgentConnection` already satisfies — same method names and types — so backend
code and client code read the same, and the in-process connection can be dropped in wherever
a client connection is expected (and vice versa). The wire types (`AgentInput`, `Resume`,
`AgentStreamChunk`, `AgentOutput`) are the existing ones from `genkit.agent`.

| Method | Signature | Wire frame (§5) | Notes |
|--------|-----------|-----------------|-------|
| `send` | `async (input: AgentInput) -> None` | `{ "input": … }` | raw input; one turn or mid-turn |
| `send_text` | `async (text: str) -> None` | `{ "input": { "messages": … } }` | convenience for a user turn |
| `send_resume` | `async (resume: Resume) -> None` | `{ "input": { "resume": … } }` | answer an interrupt |
| `detach` | `async () -> None` | `{ "input": { "detach": true } }` | background the run; server closes after `result` (§6.4) |
| `cancel` | `async () -> None` | `{ "cancel": true }` | **new** — abort the running turn, socket stays open (§6.2) |
| `close` | `async () -> None` | `{ "close": true }` | no more inputs; idempotent |
| `receive` | `(self) -> AsyncIterator[AgentStreamChunk]` | consumes `message` | yields until the terminal `result`, then stops |
| `output` | `async () -> AgentOutput` | consumes `result` | resolves once; raises `AgentClientError` on `error` |

Everything except `cancel` already exists on the in-process connection — a backend test
driving `agent.stream_bidi()` can be repointed at a server by swapping in `connect_ws_agent`,
nothing else.

### 8.3 Lifecycle & semantics

- **`receive()` spans the whole socket, not one turn.** It yields every `message` chunk
  (including a `turnEnd` per turn) and stops when the server sends the terminal `result`.
  `output()` returns that `AgentOutput`. After `receive()` ends, `output()` is already
  resolved.
- **`receive()` and `send*()` run concurrently** — that's the duplex point. A typical UI
  runs the `receive()` loop as one task and calls `send_text` / `cancel` from input handlers.
- **`close()`** tells the server no more inputs; the server finishes the in-flight/last turn,
  emits `result`, and closes. Sending after `close()` raises `AgentClientError`
  (`FAILED_PRECONDITION`).
- **`cancel()`** aborts the current turn; the next `message`/`turnEnd` reflects the cancelled
  turn and the socket stays open for a follow-up `send_text`.
- **`detach()`** ends the socket after one terminal `result` carrying the `snapshotId`;
  reconnect with `init=AgentInit(snapshot_id=…)` to keep watching.
- **Errors** arrive as an `error` frame and surface as `AgentClientError(status, message,
  details)` raised from `receive()` / `output()`. A dropped socket raises a transport error;
  reconnect with the same `init` (§7).

### 8.4 Examples

```python
from genkit.client import connect_ws_agent
from genkit.agent import AgentInit

async with connect_ws_agent(
    "wss://app/api/agents/coder/ws",
    init=AgentInit(session_id=sid),
) as conn:
    await conn.send_text("Refactor billing.py into smaller modules")

    async for chunk in conn.receive():          # one task drains output
        render(chunk)
        if user_typed():                        # duplex: send while receiving
            await conn.send_text(user_typed())  # queue — agent absorbs at a checkpoint
        if user_hit_stop():
            await conn.cancel()                 # barge-in

    print((await conn.output()).finish_reason)
```

Detach now, reattach later to the still-running task (§6.4):

```python
async with connect_ws_agent(url, init=AgentInit(session_id=sid)) as conn:
    await conn.send_text("Research the competitive landscape")
    await conn.detach()                         # background; socket closes after result
    snap = (await conn.output()).snapshot_id

# ...later, a brand-new socket reattaches...
async with connect_ws_agent(url, init=AgentInit(snapshot_id=snap)) as conn:
    await conn.send_text("focus on pricing, skip the company history")
    async for chunk in conn.receive():
        render(chunk)
```

### 8.5 Relocating the existing HTTP client

Today's HTTP client (`AgentHttpConnection` in `genkit.plugins.fastapi.client`) has zero
FastAPI dependency (just `httpx`), one caller, and **was never released** — so this is a
clean move with no back-compat shim. Relocate it to `genkit.client._http` and reshape its
entry to the `connect_http_agent` constructor so it conforms to the §8.2 `AgentConnection`
protocol (its existing `receive()` / `output()` already match; `send*` buffers the single
turn and `close()` fires the POST). `genkit.client` then pulls `httpx` (already common) and
`websockets` into core — or ships as a small standalone package if core must stay
dependency-light, with the same import path.

## 9. Client SDK — JS / TypeScript

Deferred. The JS client (`genkit/beta/client` `remoteAgent` + a `useBidiChat` React hook)
mirrors the Python surface in §8 once that's locked. Out of scope for this revision.

---

## 10. Motivating examples

### 10.1 Cancel a wrong generation and redirect

The agent is writing a 2,000-word doc, streaming token by token. Three sentences in you can
already tell it's too formal. You hit *Stop* (`cancel`) — generation halts immediately,
last-good state is kept — then type "way more casual, cut the preamble" and it picks up from
the warm session as the next turn. No waiting for a wrong 2,000 words to finish, no resend of
context, no cold reload. **Over SSE the only option is to let it finish and start over.**

### 10.2 Queue-and-absorb feedback

You ask for a feature. While the agent codes, you keep typing as things occur to you: "add
tests too" … "pytest not unittest" … "skip the integration test for now." Each lands in the
agent's input queue. At its next checkpoint (after the current tool call) it pulls all three
and folds them into the plan — *or*, if it's mid-edit, finishes the file first and absorbs
them on the next turn. The agent decides; you never block, and nothing is lost. This is the
chat behavior the brief calls out, made first-class on the wire.

### 10.3 Voice / live transcription duplex

Mic audio streams up as multimodal `input` parts continuously; partial transcripts and the
model's spoken response stream down as `message` chunks — *at the same time*. Talking over
the assistant sends `cancel` (barge-in), exactly like a phone call. One socket carries both
half-conversations; SSE would need a separate upload channel and couldn't interrupt cleanly.

### 10.4 Co-editing a live artifact

The agent maintains a document as an artifact, emitting `customPatch` / `artifact` chunks as
JSON patches stream down. The human edits the *same* document; their edits stream up as
inputs and patch the shared state. Two writers on one doc, live, over one socket — and the
chunk types that make it work (`customPatch`, `artifact`) **already exist** in the protocol.

### 10.5 Tap a background agent on the shoulder

Kick off a 10-minute research run, `detach`, close the laptop. Later, reopen a socket with
the same `snapshotId`, watch where it's at, and inject "focus on pricing, skip the company
history" — into the *still-running* task, without restarting it. Detached background work
plus a live input channel is a combination neither pure SSE nor pure background jobs give
you.

### 10.6 Zero-latency human-in-the-loop

A banking agent hits a `userApproval` interrupt. The approval UI is already connected on the
same warm socket, so clicking *Approve* sends one `resume` frame and the agent continues
instantly — no new POST, no session reload, no cold-start between "yes" and the transfer.

---

## 11. Open questions

1. **Endpoint shape** — `/api/agents/<name>/ws`, or content-negotiate an upgrade on the
   existing agent route? Leaning toward a distinct `/ws` suffix so SSE clients are untouched.
2. **Steer (deferred)** — fold `cancel` + a follow-up `input` into one atomic frame that
   aborts the in-flight turn and makes the new input the next turn. Two frames cover it today;
   revisit if the round-trip gap matters. Possibly also a soft "consider now, don't abort"
   nudge that wakes the agent's queue check early.
3. **Multiplexing** — one agent per socket (this RFC) vs. a `streamId` field to run several
   agents/sessions over one socket. Start with one; reserve a `streamId` slot if needed.
4. **Backpressure surfaced to UI** — should `queued` depth be a first-class server signal
   (a `queued` frame) rather than client-tracked?
5. **Binary frames** for audio/video parts vs. base64 in JSON `text` frames.
6. **Constructor style** — two explicit `connect_ws_agent` / `connect_http_agent` (current
   proposal) vs. one `connect_agent(url, *, transport=…)` config-driven entry. Explicit is
   greppable and lets each carry transport-specific kwargs; the config form is nicer when the
   transport is chosen by environment. Could ship both (the config form wrapping the two).
